// keelcpp/stats/src/robust_canon.cpp
// Pure zero-STL implementations of:
//   robustfit (IRLS + DuMouchel-O'Brien leverage adjustment),
//   robustcov (FAST-MCD-lite + Pison-Van Aelst-Willems consistency scaling),
//   partialcorri (OLS residualization partial correlation),
//   canoncorr (QR + SVD canonical correlation analysis),
//   studentized_range_cdf / studentized_range_inv / multcompare.

#include "keel/stats.h"

#include "keel/algo.h"
#include "keel/arena.h"
#include "keel/array.h"
#include "keel/linalg.h"
#include "keel/math.h"

namespace keel::stats {

namespace {

constexpr f64 kSqrtEps = 1.4901161193847656e-08;

// In-place Gaussian elimination with partial pivoting on p x p row-major M * x = b.
bool gauss_solve(f64* m, f64* b, usize p) noexcept {
    for (usize k = 0; k < p; ++k) {
        usize piv = k;
        f64 pmax = abs(m[k * p + k]);
        for (usize rr = k + 1; rr < p; ++rr) {
            const f64 v = abs(m[rr * p + k]);
            if (v > pmax) {
                pmax = v;
                piv = rr;
            }
        }
        if (pmax == 0.0) {
            return false;
        }
        if (piv != k) {
            for (usize j = 0; j < p; ++j) {
                swap(m[k * p + j], m[piv * p + j]);
            }
            swap(b[k], b[piv]);
        }
        const f64 pv = m[k * p + k];
        for (usize rr = k + 1; rr < p; ++rr) {
            const f64 f = m[rr * p + k] / pv;
            for (usize j = k; j < p; ++j) {
                m[rr * p + j] -= f * m[k * p + j];
            }
            b[rr] -= f * b[k];
        }
    }
    for (usize k = p; k-- > 0;) {
        f64 s = b[k];
        for (usize j = k + 1; j < p; ++j) {
            s -= m[k * p + j] * b[j];
        }
        b[k] = s / m[k * p + k];
    }
    return true;
}

// Weighted OLS: solve (X' W X) beta = X' W y. X is n x p col-major.
bool weighted_ls(const f64* x, usize n, usize p, const f64* y, const f64* w, f64* beta_out,
                 Allocator* alloc) {
    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> m(p * p, 0.0, scratch.allocator());
    for (usize i = 0; i < p; ++i) {
        for (usize j = 0; j < p; ++j) {
            f64 s = 0.0;
            for (usize k = 0; k < n; ++k) {
                s += x[i * n + k] * w[k] * x[j * n + k];
            }
            m[i * p + j] = s;
        }
        f64 s = 0.0;
        for (usize k = 0; k < n; ++k) {
            s += x[i * n + k] * w[k] * y[k];
        }
        beta_out[i] = s;
    }
    if (!gauss_solve(m.data(), beta_out, p)) {
        const f64 nanv = Limits<f64>::quiet_nan();
        for (usize i = 0; i < p; ++i) {
            beta_out[i] = nanv;
        }
        return false;
    }
    return true;
}

f64 robust_weight_val(RobustWeight wt, f64 u) noexcept {
    const f64 a = abs(u);
    switch (wt) {
    case RobustWeight::kAndrews: {
        const f64 r = max(kSqrtEps, a);
        return (r < kPi) ? (sin(r) / r) : 0.0;
    }
    case RobustWeight::kBisquare: {
        if (a >= 1.0) {
            return 0.0;
        }
        const f64 t = 1.0 - u * u;
        return t * t;
    }
    case RobustWeight::kCauchy:
        return 1.0 / (1.0 + u * u);
    case RobustWeight::kFair:
        return 1.0 / (1.0 + a);
    case RobustWeight::kHuber:
        return 1.0 / max(1.0, a);
    case RobustWeight::kLogistic: {
        const f64 r = max(kSqrtEps, a);
        return tanh(r) / r;
    }
    case RobustWeight::kOls:
        return 1.0;
    case RobustWeight::kTalwar:
        return (a < 1.0) ? 1.0 : 0.0;
    case RobustWeight::kWelsch:
        return exp(-(u * u));
    }
    return 1.0;
}

f64 madsigma(ArrayRef<f64> r_in, usize p, Allocator* alloc) {
    const usize n = r_in.size();
    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> r(n, 0.0, scratch.allocator());
    for (usize i = 0; i < n; ++i) {
        r[i] = abs(r_in[i]);
    }
    sort(r.begin(), r.end());
    const usize start = (p >= 1) ? (p - 1) : 0;
    const usize m = (start < n) ? (n - start) : 0;
    if (m == 0) {
        return 0.0;
    }
    const f64 med = (m & 1u) ? r[start + m / 2] : 0.5 * (r[start + m / 2 - 1] + r[start + m / 2]);
    return med / 0.6745;
}

f64 sample_std_vec(ArrayRef<f64> v) noexcept {
    const usize n = v.size();
    if (n < 2) {
        return 0.0;
    }
    f64 s = 0.0;
    for (usize i = 0; i < n; ++i) {
        s += v[i];
    }
    const f64 m = s / as<f64>(n);
    f64 ss = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 d = v[i] - m;
        ss += d * d;
    }
    return sqrt(ss / as<f64>(n - 1));
}

bool invert_row_major(ArrayRef<f64> c, usize d, Array<f64>& out, Allocator* alloc) {
    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> a(d * 2 * d, 0.0, scratch.allocator());
    for (usize i = 0; i < d; ++i) {
        for (usize j = 0; j < d; ++j) {
            a[i * 2 * d + j] = c[i * d + j];
        }
        a[i * 2 * d + (d + i)] = 1.0;
    }
    for (usize k = 0; k < d; ++k) {
        usize piv = k;
        f64 pmax = abs(a[k * 2 * d + k]);
        for (usize rr = k + 1; rr < d; ++rr) {
            const f64 v = abs(a[rr * 2 * d + k]);
            if (v > pmax) {
                pmax = v;
                piv = rr;
            }
        }
        if (pmax == 0.0) {
            return false;
        }
        if (piv != k) {
            for (usize j = 0; j < 2 * d; ++j) {
                swap(a[k * 2 * d + j], a[piv * 2 * d + j]);
            }
        }
        const f64 pv = a[k * 2 * d + k];
        for (usize j = 0; j < 2 * d; ++j) {
            a[k * 2 * d + j] /= pv;
        }
        for (usize rr = 0; rr < d; ++rr) {
            if (rr == k) {
                continue;
            }
            const f64 f = a[rr * 2 * d + k];
            if (f == 0.0) {
                continue;
            }
            for (usize j = 0; j < 2 * d; ++j) {
                a[rr * 2 * d + j] -= f * a[k * 2 * d + j];
            }
        }
    }
    out.assign(d * d, 0.0);
    for (usize i = 0; i < d; ++i) {
        for (usize j = 0; j < d; ++j) {
            out[i * d + j] = a[i * 2 * d + (d + j)];
        }
    }
    return true;
}

void compute_leverage(const f64* x, usize n, usize p, Array<f64>& h, Allocator* alloc) {
    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> xtx(p * p, 0.0, scratch.allocator());
    for (usize j = 0; j < p; ++j) {
        for (usize k = 0; k < p; ++k) {
            f64 s = 0.0;
            for (usize i = 0; i < n; ++i) {
                s += x[j * n + i] * x[k * n + i];
            }
            xtx[j * p + k] = s;
        }
    }
    Array<f64> inv(p * p, 0.0, scratch.allocator());
    for (usize i = 0; i < p; ++i) {
        inv[i * p + i] = 1.0;
    }
    for (usize col = 0; col < p; ++col) {
        usize piv = col;
        f64 best = abs(xtx[col * p + col]);
        for (usize r = col + 1; r < p; ++r) {
            const f64 v = abs(xtx[r * p + col]);
            if (v > best) {
                best = v;
                piv = r;
            }
        }
        if (piv != col) {
            for (usize c = 0; c < p; ++c) {
                swap(xtx[col * p + c], xtx[piv * p + c]);
                swap(inv[col * p + c], inv[piv * p + c]);
            }
        }
        const f64 d = xtx[col * p + col];
        const f64 invd = (d != 0.0) ? (1.0 / d) : 0.0;
        for (usize c = 0; c < p; ++c) {
            xtx[col * p + c] *= invd;
            inv[col * p + c] *= invd;
        }
        for (usize r = 0; r < p; ++r) {
            if (r == col) {
                continue;
            }
            const f64 f = xtx[r * p + col];
            if (f == 0.0) {
                continue;
            }
            for (usize c = 0; c < p; ++c) {
                xtx[r * p + c] -= f * xtx[col * p + c];
                inv[r * p + c] -= f * inv[col * p + c];
            }
        }
    }
    h.assign(n, 0.0);
    for (usize i = 0; i < n; ++i) {
        f64 s = 0.0;
        for (usize j = 0; j < p; ++j) {
            f64 t = 0.0;
            for (usize k = 0; k < p; ++k) {
                t += inv[j * p + k] * x[k * n + i];
            }
            s += x[j * n + i] * t;
        }
        h[i] = min(0.9999, s);
    }
}

void mean_cov_subset(const f64* x, usize n_orig, usize d, ArrayRef<usize> keep, Array<f64>& mu,
                     Array<f64>& cov, Allocator* alloc) {
    const usize k = keep.size();
    mu.assign(d, 0.0);
    for (usize idx = 0; idx < k; ++idx) {
        const usize row = keep[idx];
        for (usize j = 0; j < d; ++j) {
            mu[j] += x[j * n_orig + row];
        }
    }
    for (usize j = 0; j < d; ++j) {
        mu[j] /= as<f64>(k);
    }
    cov.assign(d * d, 0.0);
    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> diff(d, 0.0, scratch.allocator());
    for (usize idx = 0; idx < k; ++idx) {
        const usize row = keep[idx];
        for (usize j = 0; j < d; ++j) {
            diff[j] = x[j * n_orig + row] - mu[j];
        }
        for (usize a = 0; a < d; ++a) {
            for (usize b = 0; b < d; ++b) {
                cov[a * d + b] += diff[a] * diff[b];
            }
        }
    }
    const f64 denom = as<f64>(k > 1 ? k - 1 : 1);
    for (usize i = 0; i < d * d; ++i) {
        cov[i] /= denom;
    }
}

void residualise_col(const f64* c_mat, usize n, usize p, const f64* mm, const f64* wcol, f64* out,
                     Allocator* alloc) {
    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> b(p, 0.0, scratch.allocator());
    for (usize i = 0; i < p; ++i) {
        for (usize k = 0; k < n; ++k) {
            b[i] += c_mat[i * n + k] * wcol[k];
        }
    }
    Array<f64> mcopy(ArrayRef<f64>(mm, p * p), scratch.allocator());
    if (!gauss_solve(mcopy.data(), b.data(), p)) {
        fill(b.begin(), b.end(), 0.0);
    }
    for (usize k = 0; k < n; ++k) {
        f64 pred = 0.0;
        for (usize i = 0; i < p; ++i) {
            pred += c_mat[i * n + k] * b[i];
        }
        out[k] = wcol[k] - pred;
    }
}

f64 pearson_centered(const f64* a, const f64* b, usize n) noexcept {
    f64 sab = 0.0;
    f64 saa = 0.0;
    f64 sbb = 0.0;
    for (usize k = 0; k < n; ++k) {
        sab += a[k] * b[k];
        saa += a[k] * a[k];
        sbb += b[k] * b[k];
    }
    if (saa <= 0.0 || sbb <= 0.0) {
        return Limits<f64>::quiet_nan();
    }
    return sab / sqrt(saa * sbb);
}

constexpr f64 kGl16X[8] = {
    0.0950125098376374401853, 0.2816035507792589132305, 0.4580167776572273863424,
    0.6178762444026437484467, 0.7554044083550030338951, 0.8656312023878317438805,
    0.9445750230732325760780, 0.9894009349916499325961};
constexpr f64 kGl16W[8] = {
    0.1894506104550684962854, 0.1826034150449235888668, 0.1691565193950025381893,
    0.1495959888165767320815, 0.1246289712555338720525, 0.0951585116824927848099,
    0.0622535239386478928628, 0.0271524594117540948518};

inline f64 norm_phi(f64 z) noexcept { return kInvSqrt2Pi * exp(-0.5 * z * z); }
inline f64 norm_cdf0(f64 z) noexcept { return 0.5 * erfc(-z * kInvSqrt2); }

f64 normal_range_cdf(f64 w, usize k) noexcept {
    if (w <= 0.0) {
        return 0.0;
    }
    const f64 kd = as<f64>(k);
    const f64 km1 = as<f64>(k - 1);
    auto integrand = [&](f64 z) noexcept -> f64 {
        const f64 diff = norm_cdf0(z + w) - norm_cdf0(z);
        if (diff <= 0.0) {
            return 0.0;
        }
        return norm_phi(z) * pow(diff, km1);
    };
    auto quad16 = [&](f64 a, f64 b) noexcept -> f64 {
        const f64 mid = 0.5 * (a + b);
        const f64 half = 0.5 * (b - a);
        f64 s = 0.0;
        for (i32 i = 0; i < 8; ++i) {
            const f64 dx = half * kGl16X[i];
            s += kGl16W[i] * (integrand(mid - dx) + integrand(mid + dx));
        }
        return s * half;
    };
    return min(1.0, max(0.0, kd * (quad16(-8.5, -4.5) + quad16(-4.5, -1.5) + quad16(-1.5, 2.0) +
                                   quad16(2.0, 5.5))));
}

} // namespace

// ============================================================================
// Robust Regression & Covariance
// ============================================================================

f64 robust_default_tune(RobustWeight wt) noexcept {
    switch (wt) {
    case RobustWeight::kAndrews:
        return 1.339;
    case RobustWeight::kBisquare:
        return 4.685;
    case RobustWeight::kCauchy:
        return 2.385;
    case RobustWeight::kFair:
        return 1.400;
    case RobustWeight::kHuber:
        return 1.345;
    case RobustWeight::kLogistic:
        return 1.205;
    case RobustWeight::kOls:
        return 1.0;
    case RobustWeight::kTalwar:
        return 2.795;
    case RobustWeight::kWelsch:
        return 2.985;
    }
    return 1.0;
}

StatsStatus robustfit(ArrayRef<f64> x_col_major, usize n, usize p, ArrayRef<f64> y,
                      RobustWeight weight, f64 tune, MutArrayRef<f64> out_beta, f64& out_s,
                      Allocator* alloc) {
    if (p == 0 || n <= p || (n > Limits<usize>::kMax / p) || x_col_major.size() < n * p ||
        y.size() < n || out_beta.size() < p) {
        return StatsStatus::kInvalidArgument;
    }
    if (is_nan(tune)) {
        tune = robust_default_tune(weight);
    }
    if (!(tune > 0.0) || !is_finite(tune)) {
        return StatsStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n; ++i) {
        if (!is_finite(y[i])) {
            return StatsStatus::kInvalidArgument;
        }
    }
    for (usize i = 0; i < n * p; ++i) {
        if (!is_finite(x_col_major[i])) {
            return StatsStatus::kInvalidArgument;
        }
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> h(scratch.allocator());
    compute_leverage(x_col_major.data(), n, p, h, scratch.allocator());

    Array<f64> adj(n, 0.0, scratch.allocator());
    for (usize i = 0; i < n; ++i) {
        adj[i] = 1.0 / sqrt(1.0 - h[i]);
    }

    f64 tiny_s = 1e-6 * sample_std_vec(y);
    if (!(tiny_s > 0.0)) {
        tiny_s = 1.0;
    }

    Array<f64> w(n, 1.0, scratch.allocator());
    Array<f64> beta(p, 0.0, scratch.allocator());
    Array<f64> beta_prev(p, 0.0, scratch.allocator());
    if (!weighted_ls(x_col_major.data(), n, p, y.data(), w.data(), beta.data(),
                     scratch.allocator())) {
        return StatsStatus::kSingularMatrix;
    }

    Array<f64> r(n, 0.0, scratch.allocator());
    Array<f64> radj(n, 0.0, scratch.allocator());
    f64 s = 1.0;
    for (i32 it = 0; it < 50; ++it) {
        for (usize i = 0; i < n; ++i) {
            f64 pred = 0.0;
            for (usize j = 0; j < p; ++j) {
                pred += x_col_major[j * n + i] * beta[j];
            }
            r[i] = y[i] - pred;
            radj[i] = r[i] * adj[i];
        }
        s = madsigma(radj.as_ref(), p, scratch.allocator());
        const f64 denom = max(s, tiny_s) * tune;
        for (usize i = 0; i < n; ++i) {
            w[i] = robust_weight_val(weight, radj[i] / denom);
        }
        copy_n(beta_prev.data(), beta.data(), p);
        if (!weighted_ls(x_col_major.data(), n, p, y.data(), w.data(), beta.data(),
                         scratch.allocator())) {
            copy_n(beta.data(), beta_prev.data(), p);
            break;
        }

        bool changed = false;
        for (usize j = 0; j < p; ++j) {
            if (abs(beta[j] - beta_prev[j]) > kSqrtEps * max(abs(beta[j]), abs(beta_prev[j]))) {
                changed = true;
            }
        }
        if (!changed) {
            break;
        }
    }

    copy_n(out_beta.data(), beta.data(), p);
    out_s = s;
    return StatsStatus::kOk;
}

StatsStatus robustcov(ArrayRef<f64> x_col_major, usize n, usize d,
                      MutArrayRef<f64> out_sigma_col_major, MutArrayRef<f64> out_mu,
                      Allocator* alloc) {
    if (d == 0 || n <= d + 1 || (n > Limits<usize>::kMax / d) || (d > Limits<usize>::kMax / d) ||
        x_col_major.size() < n * d || out_sigma_col_major.size() < d * d || out_mu.size() < d) {
        return StatsStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n * d; ++i) {
        if (!is_finite(x_col_major[i])) {
            return StatsStatus::kInvalidArgument;
        }
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<usize> keep(n, 0, scratch.allocator());
    for (usize i = 0; i < n; ++i) {
        keep[i] = i;
    }

    Array<f64> mu(scratch.allocator());
    Array<f64> cov(scratch.allocator());
    Array<f64> inv_c(scratch.allocator());
    mean_cov_subset(x_col_major.data(), n, d, keep.as_ref(), mu, cov, scratch.allocator());

    const usize h = min<usize>(n, as<usize>(ceil(0.75 * as<f64>(n))));
    Array<usize> prev_keep(scratch.allocator());
    Array<f64> diff(d, 0.0, scratch.allocator());
    Array<Pair<f64, usize>> pairs(n, Pair<f64, usize>{0.0, 0}, scratch.allocator());
    Array<usize> new_keep(h, 0, scratch.allocator());

    for (i32 it = 0; it < 30; ++it) {
        if (!invert_row_major(cov.as_ref(), d, inv_c, scratch.allocator())) {
            break;
        }
        for (usize i = 0; i < n; ++i) {
            for (usize j = 0; j < d; ++j) {
                diff[j] = x_col_major[j * n + i] - mu[j];
            }
            f64 md = 0.0;
            for (usize a = 0; a < d; ++a) {
                for (usize b = 0; b < d; ++b) {
                    md += diff[a] * inv_c[a * d + b] * diff[b];
                }
            }
            pairs[i] = {md, i};
        }
        if (h < n) {
            nth_element(pairs.begin(), pairs.begin() + h, pairs.end());
        }
        for (usize i = 0; i < h; ++i) {
            new_keep[i] = pairs[i].second;
        }
        sort(new_keep.begin(), new_keep.end());
        if (new_keep.size() == prev_keep.size() &&
            equal(new_keep.begin(), new_keep.end(), prev_keep.begin())) {
            break;
        }
        prev_keep.resize(h);
        copy_n(prev_keep.data(), new_keep.data(), h);
        mean_cov_subset(x_col_major.data(), n, d, prev_keep.as_ref(), mu, cov, scratch.allocator());
    }

    const f64 h_ratio = as<f64>(h) / as<f64>(n);
    const f64 q = chi2_inv(h_ratio, as<f64>(d));
    const f64 f_dp2 = chi2_cdf(q, as<f64>(d + 2));
    const f64 scale = (f_dp2 > 0.0) ? (h_ratio / f_dp2) : 1.0;
    for (usize i = 0; i < d * d; ++i) {
        cov[i] *= scale;
    }

    // Transpose row-major cov to column-major output.
    for (usize i = 0; i < d; ++i) {
        for (usize j = 0; j < d; ++j) {
            out_sigma_col_major[j * d + i] = cov[i * d + j];
        }
    }
    copy_n(out_mu.data(), mu.data(), d);
    return StatsStatus::kOk;
}

// ============================================================================
// Partial Correlation (partialcorri) & Canonical Correlation (canoncorr)
// ============================================================================

StatsStatus partialcorri(ArrayRef<f64> y_col_major, usize n, usize py, ArrayRef<f64> x_col_major,
                         usize px, ArrayRef<f64> z_col_major, usize pz,
                         MutArrayRef<f64> out_r_col_major, Allocator* alloc) {
    if (y_col_major.size() < n * py || x_col_major.size() < n * px ||
        (pz > 0 && z_col_major.size() < n * pz) || out_r_col_major.size() < py * px) {
        return StatsStatus::kInvalidArgument;
    }
    if (px == 0 || py == 0) {
        return StatsStatus::kOk;
    }
    ArenaTemp scratch = get_scratch(alloc);
    const usize p_ctrl = 1u + (px > 0 ? px - 1u : 0u) + pz;
    Array<f64> c_mat(n * p_ctrl, 0.0, scratch.allocator());
    Array<f64> mm(p_ctrl * p_ctrl, 0.0, scratch.allocator());
    Array<f64> y_res(n, 0.0, scratch.allocator());
    Array<f64> x_res(n, 0.0, scratch.allocator());

    for (usize j = 0; j < px; ++j) {
        fill(c_mat.begin(), c_mat.end(), 0.0);
        for (usize k = 0; k < n; ++k) {
            c_mat[k] = 1.0;
        }
        usize cidx = 1;
        for (usize jj = 0; jj < px; ++jj) {
            if (jj == j) {
                continue;
            }
            for (usize k = 0; k < n; ++k) {
                c_mat[cidx * n + k] = x_col_major[jj * n + k];
            }
            ++cidx;
        }
        for (usize jz = 0; jz < pz; ++jz) {
            for (usize k = 0; k < n; ++k) {
                c_mat[cidx * n + k] = z_col_major[jz * n + k];
            }
            ++cidx;
        }

        for (usize a = 0; a < p_ctrl; ++a) {
            for (usize b = 0; b < p_ctrl; ++b) {
                f64 s = 0.0;
                for (usize k = 0; k < n; ++k) {
                    s += c_mat[a * n + k] * c_mat[b * n + k];
                }
                mm[a * p_ctrl + b] = s;
            }
        }

        residualise_col(c_mat.data(), n, p_ctrl, mm.data(), x_col_major.data() + j * n,
                        x_res.data(), scratch.allocator());
        for (usize i = 0; i < py; ++i) {
            residualise_col(c_mat.data(), n, p_ctrl, mm.data(), y_col_major.data() + i * n,
                            y_res.data(), scratch.allocator());
            out_r_col_major[j * py + i] = pearson_centered(y_res.data(), x_res.data(), n);
        }
    }
    return StatsStatus::kOk;
}

StatsStatus canoncorr(ArrayRef<f64> x_col_major, usize n, usize p, ArrayRef<f64> y_col_major,
                      usize q, MutArrayRef<f64> out_a_col_major, MutArrayRef<f64> out_b_col_major,
                      MutArrayRef<f64> out_r, Allocator* alloc) {
    if (p == 0 || q == 0 || n < p + q || x_col_major.size() < n * p ||
        y_col_major.size() < n * q) {
        return StatsStatus::kInvalidArgument;
    }
    const usize k = min(p, q);
    if (out_a_col_major.size() < p * k || out_b_col_major.size() < q * k || out_r.size() < k) {
        return StatsStatus::kInvalidArgument;
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> xc(n * p, 0.0, scratch.allocator());
    Array<f64> yc(n * q, 0.0, scratch.allocator());
    for (usize j = 0; j < p; ++j) {
        f64 m = 0.0;
        for (usize i = 0; i < n; ++i) {
            m += x_col_major[j * n + i];
        }
        m /= as<f64>(n);
        for (usize i = 0; i < n; ++i) {
            xc[j * n + i] = x_col_major[j * n + i] - m;
        }
    }
    for (usize j = 0; j < q; ++j) {
        f64 m = 0.0;
        for (usize i = 0; i < n; ++i) {
            m += y_col_major[j * n + i];
        }
        m /= as<f64>(n);
        for (usize i = 0; i < n; ++i) {
            yc[j * n + i] = y_col_major[j * n + i] - m;
        }
    }

    Array<f64> qx(n * n, 0.0, scratch.allocator());
    Array<f64> rx(n * p, 0.0, scratch.allocator());
    Array<f64> qy(n * n, 0.0, scratch.allocator());
    Array<f64> ry(n * q, 0.0, scratch.allocator());
    linalg::qr_householder(xc.data(), n, p, qx.data(), rx.data(), scratch.allocator());
    linalg::qr_householder(yc.data(), n, q, qy.data(), ry.data(), scratch.allocator());

    // M = QX(:, 0:p)' * QY(:, 0:q), size p x q col-major.
    Array<f64> m_mat(p * q, 0.0, scratch.allocator());
    for (usize a = 0; a < p; ++a) {
        for (usize b = 0; b < q; ++b) {
            f64 s = 0.0;
            for (usize i = 0; i < n; ++i) {
                s += qx[a * n + i] * qy[b * n + i];
            }
            m_mat[b * p + a] = s;
        }
    }

    Array<f64> u(p * p, 0.0, scratch.allocator());
    Array<f64> sval(k, 0.0, scratch.allocator());
    Array<f64> v(q * q, 0.0, scratch.allocator());
    linalg::svd_full(m_mat.data(), p, q, u.data(), sval.data(), v.data(), scratch.allocator());

    auto solve_upper = [](const f64* r_mat, usize r_rows, const f64* rhs, usize dim, usize cols,
                          f64* out_mat) noexcept {
        for (usize col = 0; col < cols; ++col) {
            for (usize i = dim; i-- > 0;) {
                f64 s = rhs[col * dim + i];
                for (usize j = i + 1; j < dim; ++j) {
                    s -= r_mat[j * r_rows + i] * out_mat[col * dim + j];
                }
                const f64 diag = r_mat[i * r_rows + i];
                out_mat[col * dim + i] = (abs(diag) > 0.0) ? (s / diag) : 0.0;
            }
        }
    };

    solve_upper(rx.data(), n, u.data(), p, k, out_a_col_major.data());
    solve_upper(ry.data(), n, v.data(), q, k, out_b_col_major.data());
    // Scale canonical coefficients by sqrt(n - 1) so canonical variates U = Xc*A, V = Yc*B
    // have unit sample variance (cov(U) = I, cov(V) = I), matching reference canoncorr.
    const f64 scale = sqrt(as<f64>(n - 1));
    for (usize i = 0; i < p * k; ++i) {
        out_a_col_major[i] *= scale;
    }
    for (usize i = 0; i < q * k; ++i) {
        out_b_col_major[i] *= scale;
    }
    for (usize i = 0; i < k; ++i) {
        out_r[i] = min(1.0, max(0.0, sval[i]));
    }
    return StatsStatus::kOk;
}

// ============================================================================
// Studentized Range Distribution & Multiple Comparisons
// ============================================================================

f64 studentized_range_cdf(f64 q, usize k, f64 df) noexcept {
    if (is_nan(q) || is_nan(df) || !(df > 0.0) || k < 2) {
        return Limits<f64>::quiet_nan();
    }
    if (q <= 0.0) {
        return 0.0;
    }
    if (is_inf(q)) {
        return 1.0;
    }
    if (is_inf(df) || df > 1e5) {
        return normal_range_cdf(q, k);
    }

    const f64 half_df = 0.5 * df;
    const f64 log_const = 0.6931471805599453094172 + half_df * log(half_df) - lgamma(half_df);
    auto integrand = [&](f64 u) noexcept -> f64 {
        if (u <= 0.0) {
            return 0.0;
        }
        const f64 lp = log_const + (df - 1.0) * log(u) - half_df * u * u;
        if (lp < -700.0) {
            return 0.0;
        }
        return exp(lp) * normal_range_cdf(q * u, k);
    };
    auto quad16 = [&](f64 a, f64 b) noexcept -> f64 {
        if (b <= a) {
            return 0.0;
        }
        const f64 mid = 0.5 * (a + b);
        const f64 half = 0.5 * (b - a);
        f64 s = 0.0;
        for (i32 i = 0; i < 8; ++i) {
            const f64 dx = half * kGl16X[i];
            s += kGl16W[i] * (integrand(mid - dx) + integrand(mid + dx));
        }
        return s * half;
    };
    const f64 spread = 10.0 / sqrt(max(0.5, df));
    const f64 u_min = max(1e-7, 1.0 - spread);
    const f64 u_max = 1.0 + spread;
    // Composite 4 panels on each side of the mode u = 1 for high accuracy at low df.
    f64 total = 0.0;
    const f64 step_lo = (1.0 - u_min) / 4.0;
    const f64 step_hi = (u_max - 1.0) / 4.0;
    for (i32 p = 0; p < 4; ++p) {
        total += quad16(u_min + as<f64>(p) * step_lo, u_min + as<f64>(p + 1) * step_lo);
        total += quad16(1.0 + as<f64>(p) * step_hi, 1.0 + as<f64>(p + 1) * step_hi);
    }
    return min(1.0, max(0.0, total));
}

f64 studentized_range_inv(f64 p, usize k, f64 df) noexcept {
    if (is_nan(p) || p < 0.0 || p > 1.0 || is_nan(df) || !(df > 0.0) || k < 2) {
        return Limits<f64>::quiet_nan();
    }
    if (p == 0.0) {
        return 0.0;
    }
    if (p == 1.0) {
        return Limits<f64>::infinity();
    }
    f64 lo = 0.0;
    f64 hi = 4.0;
    while (studentized_range_cdf(hi, k, df) < p && hi < 1e6) {
        lo = hi;
        hi *= 2.0;
    }
    for (i32 iter = 0; iter < 60; ++iter) {
        const f64 mid = 0.5 * (lo + hi);
        if (studentized_range_cdf(mid, k, df) < p) {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo <= 1e-12 * max(1.0, mid)) {
            break;
        }
    }
    return 0.5 * (lo + hi);
}

StatsStatus multcompare(ArrayRef<f64> means, ArrayRef<f64> counts, f64 s, f64 df, f64 alpha,
                        McCorrection ctype, MutArrayRef<f64> out_col_major) noexcept {
    const usize k = means.size();
    if (k < 2 || counts.size() < k || !(s >= 0.0) || !is_finite(s) || !(df > 0.0) ||
        !(alpha > 0.0 && alpha < 1.0)) {
        return StatsStatus::kInvalidArgument;
    }
    for (usize i = 0; i < k; ++i) {
        if (!is_finite(means[i]) || !(counts[i] > 0.0) || !is_finite(counts[i])) {
            return StatsStatus::kInvalidArgument;
        }
    }
    const usize n_pairs = k * (k - 1) / 2;
    if (out_col_major.size() < n_pairs * 6) {
        return StatsStatus::kInvalidArgument;
    }

    f64 t_crit = 0.0;
    if (ctype == McCorrection::kTukeyKramer) {
        t_crit = studentized_range_inv(1.0 - alpha, k, df) * kInvSqrt2;
    } else {
        const f64 alpha_eff =
            (ctype == McCorrection::kBonferroni) ? (alpha / as<f64>(n_pairs)) : alpha;
        t_crit = t_inv(1.0 - 0.5 * alpha_eff, df);
    }

    usize row = 0;
    for (usize i = 0; i < k; ++i) {
        for (usize j = i + 1; j < k; ++j) {
            const f64 diff = means[i] - means[j];
            const f64 se = s * sqrt(1.0 / counts[i] + 1.0 / counts[j]);
            const f64 half_ci = t_crit * se;
            const f64 t_stat = (se > 0.0) ? (diff / se) : 0.0;
            f64 p_val = 1.0;
            if (ctype == McCorrection::kTukeyKramer) {
                const f64 q_stat = abs(t_stat) * sqrt(2.0);
                p_val = min(1.0, max(0.0, 1.0 - studentized_range_cdf(q_stat, k, df)));
            } else {
                const f64 f_val = t_cdf(abs(t_stat), df);
                p_val = min(1.0, max(0.0, 2.0 * (1.0 - f_val)));
                if (ctype == McCorrection::kBonferroni) {
                    p_val = min(1.0, p_val * as<f64>(n_pairs));
                }
            }
            out_col_major[0 * n_pairs + row] = as<f64>(i + 1);
            out_col_major[1 * n_pairs + row] = as<f64>(j + 1);
            out_col_major[2 * n_pairs + row] = diff - half_ci;
            out_col_major[3 * n_pairs + row] = diff;
            out_col_major[4 * n_pairs + row] = diff + half_ci;
            out_col_major[5 * n_pairs + row] = p_val;
            ++row;
        }
    }
    return StatsStatus::kOk;
}

} // namespace keel::stats
