// keelcpp/stats/src/stats_extra.cpp
// Pure zero-STL implementations of:
//   tiedrank, correlation exact/asymptotic p-values (Pearson, Kendall, Spearman AS89),
//   Grubbs & GESD outlier detection, 1-D Gaussian / Savitzky-Golay / LOWESS / Hampel
//   smoothers, Halton QMC point nets, and Lloyd k-means iteration.

#include "keel/stats.h"

#include "keel/algo.h"
#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"

namespace keel::stats {

namespace {

constexpr f64 kMadToStd = 1.4826022185056018;

inline f64 qnan() noexcept { return Limits<f64>::quiet_nan(); }
inline f64 pinf() noexcept { return Limits<f64>::infinity(); }

inline f64 rank_tail_from_lu(CorrTail tail, f64 lower, f64 upper) noexcept {
    switch (tail) {
    case CorrTail::kLeft:
        return min(1.0, lower);
    case CorrTail::kRight:
        return min(1.0, upper);
    case CorrTail::kBoth:
        return min(1.0, 2.0 * min(lower, upper));
    }
    return qnan();
}

f64 as89_upper(f64 d_stat, usize n, f64 n3c) noexcept {
    static constexpr f64 c[12] = {0.2274, 0.2531, 0.1745, 0.0758, 0.1033, 0.3932,
                                  0.0879, 0.0151, 0.0072, 0.0831, 0.0131, 0.00046};
    const f64 nd = as<f64>(n);
    const f64 x = (2.0 * (d_stat - 1.0) / n3c - 1.0) * sqrt(nd - 1.0);
    const f64 y = x * x;
    const f64 u = x * (c[0] + (c[1] + c[2] / nd) / nd +
                       y * (-c[3] + (c[4] + c[5] / nd) / nd -
                            y * (c[6] + c[7] / nd -
                                 y * (c[8] - c[9] / nd + y * (c[10] - c[11] * y) / nd))) /
                           nd) /
                  nd;
    const f64 pv = u / exp(0.5 * y) + 0.5 * erfc(x * kInvSqrt2);
    return min(1.0, max(0.0, pv));
}

f64 win_median_inplace(f64* w, usize n) noexcept {
    if (n == 0) {
        return qnan();
    }
    if (n == 1) {
        return w[0];
    }
    const usize mid = n / 2;
    nth_element(w, w + mid, w + n);
    if ((n & 1u) != 0) {
        return w[mid];
    }
    const f64 upper = w[mid];
    f64 lower = w[0];
    for (usize i = 1; i < mid; ++i) {
        if (w[i] > lower) {
            lower = w[i];
        }
    }
    return 0.5 * (upper + lower);
}

Array<i32> first_primes(usize d, Allocator* alloc) {
    Array<i32> ps(alloc);
    ps.reserve(d);
    i32 x = 2;
    while (ps.size() < d) {
        bool is_p = true;
        for (usize i = 0; i < ps.size(); ++i) {
            const i32 q = ps[i];
            if (q * q > x) {
                break;
            }
            if (x % q == 0) {
                is_p = false;
                break;
            }
        }
        if (is_p) {
            ps.push_back(x);
        }
        ++x;
    }
    return ps;
}

Array<i32> rr2_permutation(i32 p, Allocator* alloc) {
    i32 k = 1;
    while ((1 << k) < p) {
        ++k;
    }
    const i32 m = (1 << k);
    Array<i32> perm(alloc);
    perm.reserve(as<usize>(p));
    for (i32 j = 0; j < m; ++j) {
        i32 rev = 0;
        for (i32 b = 0; b < k; ++b) {
            if (((j >> b) & 1) != 0) {
                rev |= (1 << (k - 1 - b));
            }
        }
        if (rev < p) {
            perm.push_back(rev);
        }
    }
    return perm;
}

f64 radical_inverse(i64 idx, i32 p, const Array<i32>* perm) noexcept {
    f64 r = 0.0;
    f64 f = 1.0 / as<f64>(p);
    while (idx > 0) {
        i32 digit = as<i32>(idx % p);
        if (perm) {
            digit = (*perm)[as<usize>(digit)];
        }
        r += f * as<f64>(digit);
        idx /= p;
        f /= as<f64>(p);
    }
    return r;
}

inline f64 sq_dist(const f64* a, const f64* b, usize d) noexcept {
    f64 s = 0.0;
    for (usize k = 0; k < d; ++k) {
        const f64 diff = a[k] - b[k];
        s += diff * diff;
    }
    return s;
}

} // namespace

// ============================================================================
// Tied Ranks
// ============================================================================

StatsStatus tiedrank(ArrayRef<f64> x, MutArrayRef<f64> out_ranks, f64& out_tieadj,
                     Allocator* alloc) {
    const usize n = x.size();
    if (out_ranks.size() < n) {
        return StatsStatus::kInvalidArgument;
    }
    out_tieadj = 0.0;
    if (n == 0) {
        return StatsStatus::kOk;
    }
    ArenaTemp scratch = get_scratch(alloc);
    Array<Pair<f64, usize>> nz(scratch.allocator());
    nz.reserve(n);
    for (usize i = 0; i < n; ++i) {
        if (is_nan(x[i])) {
            out_ranks[i] = qnan();
        } else {
            nz.emplace_back(x[i], i);
        }
    }
    stable_sort(nz.begin(), nz.end(),
                [](const Pair<f64, usize>& a, const Pair<f64, usize>& b) noexcept {
                    return a.first < b.first;
                });

    f64 tieadj = 0.0;
    const usize m = nz.size();
    usize i = 0;
    while (i < m) {
        usize j = i + 1;
        while (j < m && nz[j].first == nz[i].first) {
            ++j;
        }
        const usize group_len = j - i;
        const f64 avg_rank = (as<f64>(i + 1) + as<f64>(j)) * 0.5;
        for (usize k = i; k < j; ++k) {
            out_ranks[nz[k].second] = avg_rank;
        }
        if (group_len > 1) {
            const f64 t = as<f64>(group_len);
            tieadj += (t * t * t - t) * 0.5;
        }
        i = j;
    }
    out_tieadj = tieadj;
    return StatsStatus::kOk;
}

// ============================================================================
// Correlation P-Values (Pearson, Kendall, Spearman Exact & AS89)
// ============================================================================

f64 corr_pearson_pval(CorrTail tail, f64 r, usize n) noexcept {
    if (n < 3 || is_nan(r)) {
        return qnan();
    }
    if (abs(r) >= 1.0) {
        const bool pos = (r > 0.0);
        switch (tail) {
        case CorrTail::kBoth:
            return 0.0;
        case CorrTail::kRight:
            return pos ? 0.0 : 1.0;
        case CorrTail::kLeft:
            return pos ? 1.0 : 0.0;
        }
    }
    const f64 df = as<f64>(n) - 2.0;
    const f64 t = r * sqrt(df / (1.0 - r * r));
    switch (tail) {
    case CorrTail::kBoth:
        return min(1.0, 2.0 * t_cdf(-abs(t), df));
    case CorrTail::kRight:
        return min(1.0, t_cdf(-t, df));
    case CorrTail::kLeft:
        return min(1.0, t_cdf(t, df));
    }
    return qnan();
}

StatsStatus corr_kendall_exact_lu(f64 tau, usize n, f64& out_lower, f64& out_upper,
                                  Allocator* alloc) {
    if (n < 2 || is_nan(tau)) {
        out_lower = qnan();
        out_upper = qnan();
        return StatsStatus::kInvalidArgument;
    }
    const usize d_max = n * (n - 1) / 2;
    i64 d_obs = round_i64(as<f64>(d_max) * (1.0 - tau) * 0.5);
    if (d_obs < 0) {
        d_obs = 0;
    }
    if (d_obs > as<i64>(d_max)) {
        d_obs = as<i64>(d_max);
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> prob(d_max + 1, 0.0, scratch.allocator());
    Array<f64> next(d_max + 1, 0.0, scratch.allocator());
    prob[0] = 1.0;
    usize cur_len = 1;
    for (usize k = 1; k < n; ++k) {
        const usize new_len = cur_len + k;
        fill(next.begin(), next.begin() + new_len, 0.0);
        const f64 w = 1.0 / as<f64>(k + 1);
        for (usize d = 0; d < cur_len; ++d) {
            const f64 pv = prob[d] * w;
            for (usize s = 0; s <= k; ++s) {
                next[d + s] += pv;
            }
        }
        swap(prob, next);
        cur_len = new_len;
    }
    f64 lower = 0.0;
    f64 upper = 0.0;
    for (i64 d = 0; d <= d_obs; ++d) {
        lower += prob[as<usize>(d)];
    }
    for (i64 d = d_obs; d < as<i64>(cur_len); ++d) {
        upper += prob[as<usize>(d)];
    }
    out_lower = lower;
    out_upper = upper;
    return StatsStatus::kOk;
}

f64 corr_kendall_normal_pval(CorrTail tail, f64 tau, usize n) noexcept {
    if (n < 2 || is_nan(tau)) {
        return qnan();
    }
    if (abs(tau) >= 1.0) {
        return rank_tail_from_lu(tail, tau > 0.0 ? 1.0 : 0.0, tau > 0.0 ? 0.0 : 1.0);
    }
    const f64 nd = as<f64>(n);
    const f64 z = 3.0 * tau * sqrt(nd * (nd - 1.0)) / sqrt(2.0 * (2.0 * nd + 5.0));
    const f64 phi = 0.5 * erfc(-z * kInvSqrt2);
    switch (tail) {
    case CorrTail::kLeft:
        return min(1.0, phi);
    case CorrTail::kRight:
        return min(1.0, 1.0 - phi);
    case CorrTail::kBoth:
        return min(1.0, 2.0 * min(phi, 1.0 - phi));
    }
    return qnan();
}

f64 corr_spearman_no_ties_pval(CorrTail tail, f64 rho, usize n, Allocator* alloc) {
    if (n < 2 || is_nan(rho)) {
        return qnan();
    }
    const f64 nd = as<f64>(n);
    if (n < 10) {
        const f64 scale = nd * (nd * nd - 1.0) / 6.0;
        i64 d_obs = round_i64((1.0 - rho) * scale);
        if (d_obs < 0) {
            d_obs = 0;
        }
        ArenaTemp scratch = get_scratch(alloc);
        Array<i32> perm(n, 0, scratch.allocator());
        for (usize i = 0; i < n; ++i) {
            perm[i] = as<i32>(i);
        }
        usize total = 0;
        usize le_cnt = 0;
        usize ge_cnt = 0;
        do {
            i64 d = 0;
            for (usize i = 0; i < n; ++i) {
                const i64 diff = as<i64>(perm[i]) - as<i64>(i);
                d += diff * diff;
            }
            if (d <= d_obs) {
                ++le_cnt;
            }
            if (d >= d_obs) {
                ++ge_cnt;
            }
            ++total;
        } while (next_permutation(perm.begin(), perm.end()));
        return rank_tail_from_lu(tail, as<f64>(le_cnt) / as<f64>(total),
                                 as<f64>(ge_cnt) / as<f64>(total));
    }
    const f64 n3c = (nd + 1.0) * nd * (nd - 1.0) / 3.0;
    const f64 scale = 0.5 * n3c;
    const f64 d_stat = scale - as<f64>(round_i64(rho * scale));
    switch (tail) {
    case CorrTail::kBoth:
        return min(1.0, 2.0 * as89_upper(max(d_stat, n3c - d_stat), n, n3c));
    case CorrTail::kLeft:
        return as89_upper(d_stat, n, n3c);
    case CorrTail::kRight:
        return as89_upper(n3c - d_stat, n, n3c);
    }
    return qnan();
}

StatsStatus corr_spearman_exact_midrank_lu(ArrayRef<f64> xr, ArrayRef<f64> yr, f64 d_obs,
                                         f64& out_lower, f64& out_upper, Allocator* alloc) {
    const usize n = xr.size();
    if (n == 0 || yr.size() < n) {
        out_lower = qnan();
        out_upper = qnan();
        return StatsStatus::kInvalidArgument;
    }
    ArenaTemp scratch = get_scratch(alloc);
    Array<i32> perm(n, 0, scratch.allocator());
    for (usize i = 0; i < n; ++i) {
        perm[i] = as<i32>(i);
    }
    usize total = 0;
    usize le_cnt = 0;
    usize ge_cnt = 0;
    do {
        f64 d = 0.0;
        for (usize i = 0; i < n; ++i) {
            const f64 diff = xr[i] - yr[as<usize>(perm[i])];
            d += diff * diff;
        }
        if (d <= d_obs) {
            ++le_cnt;
        }
        if (d >= d_obs) {
            ++ge_cnt;
        }
        ++total;
    } while (next_permutation(perm.begin(), perm.end()));
    out_lower = as<f64>(le_cnt) / as<f64>(total);
    out_upper = as<f64>(ge_cnt) / as<f64>(total);
    return StatsStatus::kOk;
}

// ============================================================================
// Outlier Detection: Grubbs & GESD
// ============================================================================

StatsStatus grubbs_outliers(ArrayRef<f64> x, f64 alpha, MutArrayRef<u8> out_mask,
                            OutlierBounds& out_bounds, Allocator* alloc) {
    const usize n = x.size();
    if (out_mask.size() < n) {
        return StatsStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n; ++i) {
        out_mask[i] = 0;
    }
    out_bounds = {qnan(), -pinf(), pinf()};

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> v(scratch.allocator());
    v.reserve(n);
    for (usize i = 0; i < n; ++i) {
        if (!is_nan(x[i])) {
            v.push_back(x[i]);
        }
    }
    if (v.empty()) {
        return StatsStatus::kOk;
    }

    f64 astd = 0.0;
    f64 thr = 0.0;
    f64 cen = 0.0;
    while (v.size() >= 3) {
        const usize m = v.size();
        const f64 md = as<f64>(m);
        f64 s = 0.0;
        for (usize i = 0; i < m; ++i) {
            s += v[i];
        }
        cen = s / md;
        f64 ss = 0.0;
        for (usize i = 0; i < m; ++i) {
            const f64 d = v[i] - cen;
            ss += d * d;
        }
        astd = sqrt(ss / (md - 1.0));
        if (!(astd > 0.0)) {
            break;
        }
        usize jmax = 0;
        f64 amax = -1.0;
        for (usize j = 0; j < m; ++j) {
            const f64 ad = abs(v[j] - cen);
            if (ad > amax) {
                amax = ad;
                jmax = j;
            }
        }
        const f64 dof = md - 2.0;
        const f64 t = t_inv(alpha / (2.0 * md), dof);
        thr = ((md - 1.0) / sqrt(md)) * abs(t) / sqrt(dof + t * t);
        if (amax / astd > thr) {
            v.remove_at(jmax);
        } else {
            break;
        }
    }
    out_bounds.center = cen;
    out_bounds.lo = cen - astd * thr;
    out_bounds.hi = cen + astd * thr;
    for (usize i = 0; i < n; ++i) {
        out_mask[i] = is_nan(x[i]) ? 0u : ((x[i] > out_bounds.hi || x[i] < out_bounds.lo) ? 1u : 0u);
    }
    return StatsStatus::kOk;
}

StatsStatus gesd_outliers(ArrayRef<f64> x, f64 alpha, i64 max_outliers, MutArrayRef<u8> out_mask,
                          OutlierBounds& out_bounds, Allocator* alloc) {
    const usize n = x.size();
    if (out_mask.size() < n) {
        return StatsStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n; ++i) {
        out_mask[i] = 0;
    }
    out_bounds = {qnan(), -pinf(), pinf()};

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> v(scratch.allocator());
    v.reserve(n);
    for (usize i = 0; i < n; ++i) {
        if (!is_nan(x[i])) {
            v.push_back(x[i]);
        }
    }
    const usize nv = v.size();
    if (nv < 2) {
        return StatsStatus::kOk;
    }
    const usize maxo =
        (max_outliers >= 0) ? as<usize>(max_outliers) : as<usize>(ceil(0.1 * as<f64>(nv)));
    const usize rounds = min(nv, maxo);
    if (rounds == 0) {
        return StatsStatus::kOk;
    }

    Array<f64> amean(rounds, 0.0, scratch.allocator());
    Array<f64> astd(rounds, 0.0, scratch.allocator());
    Array<f64> lam(rounds, 0.0, scratch.allocator());
    Array<f64> rstat(rounds, 0.0, scratch.allocator());
    for (usize it = 0; it < rounds; ++it) {
        const usize m = v.size();
        const f64 md = as<f64>(m);
        const f64 nm1 = as<f64>(nv - it);
        f64 s = 0.0;
        for (usize j = 0; j < m; ++j) {
            s += v[j];
        }
        amean[it] = s / md;
        f64 ss = 0.0;
        for (usize j = 0; j < m; ++j) {
            const f64 d = v[j] - amean[it];
            ss += d * d;
        }
        astd[it] = (m >= 2) ? sqrt(ss / (md - 1.0)) : 0.0;
        usize jmax = 0;
        f64 amax = -1.0;
        for (usize j = 0; j < m; ++j) {
            const f64 ad = abs(v[j] - amean[it]);
            if (ad > amax) {
                amax = ad;
                jmax = j;
            }
        }
        rstat[it] = (astd[it] > 0.0) ? (amax / astd[it]) : pinf();
        v.remove_at(jmax);
        const f64 dof = as<f64>(nv) - as<f64>(it) - 1.0;
        const f64 pp = 1.0 - alpha / (2.0 * nm1);
        const f64 t = t_inv(pp, dof);
        lam[it] = (nm1 - 1.0) * t / sqrt((dof + t * t) * nm1);
    }
    i64 last = -1;
    for (usize it = 0; it < rounds; ++it) {
        if (rstat[it] > lam[it]) {
            last = as<i64>(it);
        }
    }
    const usize tindex = (last < 0) ? 0 : min<usize>(as<usize>(last) + 1, rounds - 1);
    out_bounds.center = amean[tindex];
    out_bounds.lo = amean[tindex] - astd[tindex] * lam[tindex];
    out_bounds.hi = amean[tindex] + astd[tindex] * lam[tindex];
    for (usize i = 0; i < n; ++i) {
        out_mask[i] = is_nan(x[i]) ? 0u : ((x[i] > out_bounds.hi || x[i] < out_bounds.lo) ? 1u : 0u);
    }
    return StatsStatus::kOk;
}

// ============================================================================
// 1-D Smoothing (Gaussian, Savitzky-Golay, LOWESS/LOESS) & Hampel Filter
// ============================================================================

void smooth_gaussian_strided(const f64* src, usize n, isize step, i64 kb, i64 kf, f64 sigma,
                             bool omit_nan, f64* dst) noexcept {
    const f64 inv2s2 = (sigma > 0.0) ? (1.0 / (2.0 * sigma * sigma)) : 0.0;
    const i64 n_i = as<i64>(n);
    for (i64 i = 0; i < n_i; ++i) {
        const i64 lo = max<i64>(0, i - kb);
        const i64 hi = min<i64>(n_i - 1, i + kf);
        f64 sw = 0.0;
        f64 ssum = 0.0;
        for (i64 j = lo; j <= hi; ++j) {
            const f64 v = src[j * step];
            if (omit_nan && is_nan(v)) {
                continue;
            }
            const f64 dd = as<f64>(j - i);
            const f64 wt = exp(-dd * dd * inv2s2);
            ssum += wt * v;
            sw += wt;
        }
        dst[i * step] = (sw > 0.0) ? (ssum / sw) : qnan();
    }
}

StatsStatus smooth_sgolay_matrix(i32 frame_len, i32 deg, Array<f64>& out_b, Allocator* alloc) {
    if (frame_len < 1 || (frame_len % 2) == 0 || deg < 0 || frame_len < deg + 1) {
        return StatsStatus::kInvalidArgument;
    }
    const i32 p1 = deg + 1;
    const i32 m = (frame_len - 1) / 2;
    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> ata(as<usize>(p1) * as<usize>(p1), 0.0, scratch.allocator());
    for (i32 a = 0; a < p1; ++a) {
        for (i32 b = 0; b < p1; ++b) {
            f64 s = 0.0;
            for (i32 i = 0; i < frame_len; ++i) {
                s += pow(as<f64>(i - m), a + b);
            }
            ata[as<usize>(a) * as<usize>(p1) + as<usize>(b)] = s;
        }
    }
    const i32 w = 2 * p1;
    Array<f64> aug(as<usize>(p1) * as<usize>(w), 0.0, scratch.allocator());
    for (i32 i = 0; i < p1; ++i) {
        for (i32 j = 0; j < p1; ++j) {
            aug[as<usize>(i) * as<usize>(w) + as<usize>(j)] =
                ata[as<usize>(i) * as<usize>(p1) + as<usize>(j)];
        }
        aug[as<usize>(i) * as<usize>(w) + as<usize>(p1 + i)] = 1.0;
    }
    for (i32 col = 0; col < p1; ++col) {
        i32 piv = col;
        f64 best = abs(aug[as<usize>(col) * as<usize>(w) + as<usize>(col)]);
        for (i32 r = col + 1; r < p1; ++r) {
            const f64 v = abs(aug[as<usize>(r) * as<usize>(w) + as<usize>(col)]);
            if (v > best) {
                best = v;
                piv = r;
            }
        }
        if (best == 0.0) {
            return StatsStatus::kSingularMatrix;
        }
        if (piv != col) {
            for (i32 j = 0; j < w; ++j) {
                swap(aug[as<usize>(col) * as<usize>(w) + as<usize>(j)],
                     aug[as<usize>(piv) * as<usize>(w) + as<usize>(j)]);
            }
        }
        const f64 diag = aug[as<usize>(col) * as<usize>(w) + as<usize>(col)];
        for (i32 j = 0; j < w; ++j) {
            aug[as<usize>(col) * as<usize>(w) + as<usize>(j)] /= diag;
        }
        for (i32 r = 0; r < p1; ++r) {
            if (r == col) {
                continue;
            }
            const f64 f = aug[as<usize>(r) * as<usize>(w) + as<usize>(col)];
            if (f == 0.0) {
                continue;
            }
            for (i32 j = 0; j < w; ++j) {
                aug[as<usize>(r) * as<usize>(w) + as<usize>(j)] -=
                    f * aug[as<usize>(col) * as<usize>(w) + as<usize>(j)];
            }
        }
    }

    out_b = Array<f64>(as<usize>(frame_len) * as<usize>(frame_len), 0.0, alloc);
    for (i32 i = 0; i < frame_len; ++i) {
        for (i32 j = 0; j < frame_len; ++j) {
            f64 s = 0.0;
            for (i32 a = 0; a < p1; ++a) {
                const f64 ia = pow(as<f64>(i - m), a);
                for (i32 b = 0; b < p1; ++b) {
                    s += ia * aug[as<usize>(a) * as<usize>(w) + as<usize>(p1 + b)] *
                         pow(as<f64>(j - m), b);
                }
            }
            out_b[as<usize>(i) * as<usize>(frame_len) + as<usize>(j)] = s;
        }
    }
    return StatsStatus::kOk;
}

void smooth_sgolay_strided(const f64* src, usize n, isize step, ArrayRef<f64> b_row_major,
                           i32 frame_len, f64* dst) noexcept {
    const i32 n_i = as<i32>(n);
    const i32 m = (frame_len - 1) / 2;
    for (i32 nn = 0; nn < n_i; ++nn) {
        i32 row = m;
        i32 base = nn - m;
        if (nn < m) {
            row = nn;
            base = 0;
        } else if (nn >= n_i - m) {
            row = nn - (n_i - frame_len);
            base = n_i - frame_len;
        }
        f64 s = 0.0;
        for (i32 j = 0; j < frame_len; ++j) {
            s += b_row_major[as<usize>(row) * as<usize>(frame_len) + as<usize>(j)] *
                 src[as<isize>(base + j) * step];
        }
        dst[as<isize>(nn) * step] = s;
    }
}

void smooth_local_regress_strided(const f64* src, usize n, isize step, i32 frame_len, i32 deg,
                                  f64* dst) noexcept {
    const i32 n_i = as<i32>(n);
    const i32 m = (frame_len - 1) / 2;
    const i32 p = deg + 1;
    for (i32 i = 0; i < n_i; ++i) {
        i32 lo = i - m;
        if (lo < 0) {
            lo = 0;
        }
        if (lo > n_i - frame_len) {
            lo = n_i - frame_len;
        }
        f64 dmax = 0.0;
        for (i32 j = lo; j < lo + frame_len; ++j) {
            const f64 d = abs(as<f64>(j - i));
            if (d > dmax) {
                dmax = d;
            }
        }
        if (dmax == 0.0) {
            dmax = 1.0;
        }

        f64 ata[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        f64 atb[3] = {0, 0, 0};
        for (i32 j = lo; j < lo + frame_len; ++j) {
            f64 u = abs(as<f64>(j - i)) / dmax;
            if (u > 1.0) {
                u = 1.0;
            }
            const f64 t1 = 1.0 - u * u * u;
            const f64 w = t1 * t1 * t1;
            if (w == 0.0) {
                continue;
            }
            f64 basis[3] = {1.0, 0.0, 0.0};
            for (i32 k = 1; k < p; ++k) {
                basis[k] = basis[k - 1] * as<f64>(j - i);
            }
            const f64 yj = src[as<isize>(j) * step];
            for (i32 a = 0; a < p; ++a) {
                atb[a] += w * basis[a] * yj;
                for (i32 b = 0; b < p; ++b) {
                    ata[a * p + b] += w * basis[a] * basis[b];
                }
            }
        }

        f64 mat[9];
        f64 rhs[3];
        for (i32 t = 0; t < p * p; ++t) {
            mat[t] = ata[t];
        }
        for (i32 t = 0; t < p; ++t) {
            rhs[t] = atb[t];
        }
        bool ok = true;
        for (i32 col = 0; col < p && ok; ++col) {
            i32 piv = col;
            f64 best = abs(mat[col * p + col]);
            for (i32 r = col + 1; r < p; ++r) {
                const f64 v = abs(mat[r * p + col]);
                if (v > best) {
                    best = v;
                    piv = r;
                }
            }
            if (best < 1e-12) {
                ok = false;
                break;
            }
            if (piv != col) {
                for (i32 j = 0; j < p; ++j) {
                    swap(mat[col * p + j], mat[piv * p + j]);
                }
                swap(rhs[col], rhs[piv]);
            }
            const f64 diag = mat[col * p + col];
            for (i32 j = 0; j < p; ++j) {
                mat[col * p + j] /= diag;
            }
            rhs[col] /= diag;
            for (i32 r = 0; r < p; ++r) {
                if (r == col) {
                    continue;
                }
                const f64 f = mat[r * p + col];
                for (i32 j = 0; j < p; ++j) {
                    mat[r * p + j] -= f * mat[col * p + j];
                }
                rhs[r] -= f * rhs[col];
            }
        }
        dst[as<isize>(i) * step] =
            ok ? rhs[0] : (ata[0] > 0.0 ? (atb[0] / ata[0]) : src[as<isize>(i) * step]);
    }
}

StatsStatus hampel_1d(ArrayRef<f64> src, i32 k, f64 nsigmas, MutArrayRef<f64> dst,
                      MutArrayRef<u8> mask, MutArrayRef<f64> median_out, MutArrayRef<f64> sigma_out,
                      Allocator* alloc) {
    const usize n = src.size();
    if (k < 0 || !(nsigmas > 0.0) || dst.size() < n) {
        return StatsStatus::kInvalidArgument;
    }
    if ((!mask.empty() && mask.size() < n) || (!median_out.empty() && median_out.size() < n) ||
        (!sigma_out.empty() && sigma_out.size() < n)) {
        return StatsStatus::kInvalidArgument;
    }
    if (n == 0) {
        return StatsStatus::kOk;
    }
    ArenaTemp scratch = get_scratch(alloc);
    const usize max_len = as<usize>(2 * k + 1);
    Array<f64> buf(max_len, 0.0, scratch.allocator());
    Array<f64> copy1(max_len, 0.0, scratch.allocator());
    Array<f64> devs(max_len, 0.0, scratch.allocator());

    const i64 n_i = as<i64>(n);
    for (i64 i = 0; i < n_i; ++i) {
        const i64 lo = max<i64>(0, i - k);
        const i64 hi = min<i64>(n_i - 1, i + k);
        const usize len = as<usize>(hi - lo + 1);
        for (usize j = 0; j < len; ++j) {
            const f64 v = src[as<usize>(lo) + j];
            buf[j] = v;
            copy1[j] = v;
        }
        const f64 med = win_median_inplace(copy1.data(), len);
        for (usize j = 0; j < len; ++j) {
            devs[j] = abs(buf[j] - med);
        }
        const f64 mad = win_median_inplace(devs.data(), len);
        const f64 sig = kMadToStd * mad;
        const usize ui = as<usize>(i);
        const bool is_out = abs(src[ui] - med) > nsigmas * sig;
        dst[ui] = is_out ? med : src[ui];
        if (!mask.empty()) {
            mask[ui] = is_out ? 1u : 0u;
        }
        if (!median_out.empty()) {
            median_out[ui] = med;
        }
        if (!sigma_out.empty()) {
            sigma_out[ui] = sig;
        }
    }
    return StatsStatus::kOk;
}

// ============================================================================
// Halton Quasi-Random Sequence
// ============================================================================

StatsStatus halton_net(usize n, usize d, i64 skip, i64 leap, bool rr2_scramble,
                       MutArrayRef<f64> out_col_major, Allocator* alloc) {
    if (d == 0 || out_col_major.size() < n * d) {
        return StatsStatus::kInvalidArgument;
    }
    if (n == 0) {
        return StatsStatus::kOk;
    }
    ArenaTemp scratch = get_scratch(alloc);
    const Array<i32> primes = first_primes(d, scratch.allocator());
    Array<Array<i32>> perms(scratch.allocator());
    if (rr2_scramble) {
        perms.reserve(d);
        for (usize k = 0; k < d; ++k) {
            perms.push_back(rr2_permutation(primes[k], scratch.allocator()));
        }
    }
    const i64 step = leap + 1;
    for (usize row = 0; row < n; ++row) {
        const i64 idx = skip + as<i64>(row) * step;
        for (usize k = 0; k < d; ++k) {
            out_col_major[row + k * n] =
                radical_inverse(idx, primes[k], rr2_scramble ? &perms[k] : nullptr);
        }
    }
    return StatsStatus::kOk;
}

// ============================================================================
// Lloyd K-Means Iteration
// ============================================================================

StatsStatus kmeans_lloyd(ArrayRef<f64> x_row_major, usize n, usize d, i32 k,
                         MutArrayRef<f64> centroids_row_major, i32 max_iter,
                         MutArrayRef<i32> out_idx, MutArrayRef<f64> out_sumd, f64& out_total,
                         Allocator* alloc) {
    if (k < 1 || as<usize>(k) > n || d == 0 || x_row_major.size() < n * d ||
        centroids_row_major.size() < as<usize>(k) * d || out_idx.size() < n ||
        out_sumd.size() < as<usize>(k)) {
        return StatsStatus::kInvalidArgument;
    }
    if (max_iter <= 0) {
        max_iter = 100;
    }
    const usize ku = as<usize>(k);
    for (usize i = 0; i < n; ++i) {
        out_idx[i] = 0;
    }
    ArenaTemp scratch = get_scratch(alloc);
    Array<i32> counts(ku, 0, scratch.allocator());
    Array<f64> cnext(ku * d, 0.0, scratch.allocator());

    for (i32 iter = 0; iter < max_iter; ++iter) {
        bool changed = false;
        fill(counts.begin(), counts.end(), 0);
        fill(cnext.begin(), cnext.end(), 0.0);
        for (usize i = 0; i < n; ++i) {
            const f64* xi = &x_row_major[i * d];
            i32 best = 0;
            f64 bd = pinf();
            for (i32 c = 0; c < k; ++c) {
                const f64 dist = sq_dist(xi, &centroids_row_major[as<usize>(c) * d], d);
                if (dist < bd) {
                    bd = dist;
                    best = c;
                }
            }
            if (out_idx[i] != best) {
                out_idx[i] = best;
                changed = true;
            }
            const usize bu = as<usize>(best);
            counts[bu] += 1;
            for (usize j = 0; j < d; ++j) {
                cnext[bu * d + j] += xi[j];
            }
        }
        for (usize c = 0; c < ku; ++c) {
            if (counts[c] > 0) {
                const f64 inv = 1.0 / as<f64>(counts[c]);
                for (usize j = 0; j < d; ++j) {
                    centroids_row_major[c * d + j] = cnext[c * d + j] * inv;
                }
            }
        }
        if (!changed) {
            break;
        }
    }

    for (usize c = 0; c < ku; ++c) {
        out_sumd[c] = 0.0;
    }
    out_total = 0.0;
    for (usize i = 0; i < n; ++i) {
        const usize c = as<usize>(out_idx[i]);
        const f64 dist = sq_dist(&x_row_major[i * d], &centroids_row_major[c * d], d);
        out_sumd[c] += dist;
        out_total += dist;
    }
    return StatsStatus::kOk;
}

} // namespace keel::stats
