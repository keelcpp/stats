// keelcpp/stats/src/hypothesis.cpp
// Pure zero-STL parametric and non-parametric hypothesis tests.

#include "keel/stats.h"

#include "keel/algo.h"
#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"
#include "keel/random.h"
#include "keel/specfun.h"

namespace keel::stats {

namespace {

void sample_mean_var(ArrayRef<f64> x, f64& mean_out, f64& var_out) noexcept {
    const usize n = x.size();
    if (n == 0) {
        mean_out = 0.0;
        var_out = 0.0;
        return;
    }
    f64 s = 0.0;
    for (usize i = 0; i < n; ++i) {
        s += x[i];
    }
    mean_out = s / as<f64>(n);
    if (n < 2) {
        var_out = 0.0;
        return;
    }
    f64 sq = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 d = x[i] - mean_out;
        sq += d * d;
    }
    var_out = sq / as<f64>(n - 1);
}

f64 tail_from_cdf(f64 cdf, TestTail tail) noexcept {
    switch (tail) {
        case TestTail::kBoth: return 2.0 * min(cdf, 1.0 - cdf);
        case TestTail::kRight: return 1.0 - cdf;
        case TestTail::kLeft: return cdf;
    }
    return 1.0;
}

void mat_mul_sq(const f64* a, const f64* b, f64* c, i32 m) noexcept {
    const usize mu = as<usize>(m);
    for (usize i = 0; i < mu; ++i) {
        for (usize j = 0; j < mu; ++j) {
            f64 s = 0.0;
            for (usize k = 0; k < mu; ++k) {
                s += a[i * mu + k] * b[k * mu + j];
            }
            c[i * mu + j] = s;
        }
    }
}

void mat_power(const f64* a, i32 ea, Array<f64>& v, i32& ev, i32 m, i32 n, Allocator* alloc) {
    const usize sz = as<usize>(m) * as<usize>(m);
    if (n == 1) {
        v.resize(sz);
        for (usize i = 0; i < sz; ++i) {
            v[i] = a[i];
        }
        ev = ea;
        return;
    }
    mat_power(a, ea, v, ev, m, n / 2, alloc);
    Array<f64> b(sz, 0.0, alloc);
    mat_mul_sq(v.data(), v.data(), b.data(), m);
    const i32 eb = 2 * ev;
    if ((n & 1) == 0) {
        v = move(b);
        ev = eb;
    } else {
        mat_mul_sq(a, b.data(), v.data(), m);
        ev = ea + eb;
    }
    const usize mid = as<usize>(m / 2) * as<usize>(m) + as<usize>(m / 2);
    if (v[mid] > 1e140) {
        for (usize i = 0; i < sz; ++i) {
            v[i] *= 1e-140;
        }
        ev += 140;
    }
}

f64 marsaglia_k(i32 n, f64 d, Allocator* alloc) {
    if (d <= 0.0) {
        return 0.0;
    }
    if (d >= 1.0) {
        return 1.0;
    }
    const f64 nd = as<f64>(n) * d;
    const i32 k = as<i32>(nd) + 1;
    const i32 m = 2 * k - 1;
    const f64 h = as<f64>(k) - nd;
    const usize mu = as<usize>(m);

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> h_mat(mu * mu, 0.0, scratch.allocator());
    Array<f64> q_mat(mu * mu, 0.0, scratch.allocator());
    for (i32 i = 0; i < m; ++i) {
        for (i32 j = 0; j < m; ++j) {
            h_mat[as<usize>(i) * mu + as<usize>(j)] = (i - j + 1 < 0) ? 0.0 : 1.0;
        }
    }
    for (i32 i = 0; i < m; ++i) {
        h_mat[as<usize>(i) * mu + 0] -= pow(h, as<f64>(i + 1));
        h_mat[as<usize>(m - 1) * mu + as<usize>(i)] -= pow(h, as<f64>(m - i));
    }
    if (2.0 * h - 1.0 > 0.0) {
        h_mat[as<usize>(m - 1) * mu + 0] += pow(2.0 * h - 1.0, as<f64>(m));
    }
    for (i32 i = 0; i < m; ++i) {
        for (i32 j = 0; j < m; ++j) {
            if (i - j + 1 > 0) {
                for (i32 g = 1; g <= i - j + 1; ++g) {
                    h_mat[as<usize>(i) * mu + as<usize>(j)] /= as<f64>(g);
                }
            }
        }
    }
    i32 eq = 0;
    mat_power(h_mat.data(), 0, q_mat, eq, m, n, scratch.allocator());
    f64 s = q_mat[as<usize>(k - 1) * mu + as<usize>(k - 1)];
    for (i32 i = 1; i <= n; ++i) {
        s = s * as<f64>(i) / as<f64>(n);
        if (s < 1e-140) {
            s *= 1e140;
            eq -= 140;
        }
    }
    return s * pow(10.0, as<f64>(eq));
}

f64 birnbaum_tingey(i32 n, f64 d) noexcept {
    if (d <= 0.0) {
        return 1.0;
    }
    if (d >= 1.0) {
        return 0.0;
    }
    const f64 dn = as<f64>(n);
    const i32 jmax = as<i32>(floor(dn * (1.0 - d)));
    f64 sum = 0.0;
    for (i32 j = 0; j <= jmax; ++j) {
        const f64 dj = as<f64>(j);
        const f64 a = 1.0 - d - dj / dn;
        const f64 b = d + dj / dn;
        if (a <= 0.0 && n - j > 0) {
            continue;
        }
        const f64 log_c =
            specfun::gammaln(dn + 1.0) - specfun::gammaln(dj + 1.0) - specfun::gammaln(dn - dj + 1.0);
        f64 logterm = log_c + (dj - 1.0) * log(b);
        if (n - j > 0) {
            logterm += (dn - dj) * log(a);
        }
        sum += exp(logterm);
    }
    return clamp(d * sum, 0.0, 1.0);
}

f64 ks_one_sample_two_sided_p(i32 n, f64 d, Allocator* alloc) {
    const f64 s = as<f64>(n) * d * d;
    if (s > 7.24 || (s > 3.76 && n > 99)) {
        return min(1.0, 2.0 * exp(-(2.000071 + 0.331 / sqrt(as<f64>(n)) + 1.409 / as<f64>(n)) * s));
    }
    return 1.0 - marsaglia_k(n, d, alloc);
}

f64 ks_two_sample_p(f64 neff, f64 d, bool two_sided) noexcept {
    const f64 lambda = (sqrt(neff) + 0.12 + 0.11 / sqrt(neff)) * d;
    if (!two_sided) {
        return clamp(exp(-2.0 * lambda * lambda), 0.0, 1.0);
    }
    f64 total = 0.0;
    for (i32 k = 1; k <= 101; ++k) {
        const f64 dk = as<f64>(k);
        const f64 term = exp(-2.0 * lambda * lambda * dk * dk);
        total += ((k & 1) != 0) ? term : -term;
        if (term < 1e-20 && k > 4) {
            break;
        }
    }
    return clamp(2.0 * total, 0.0, 1.0);
}

template <typename PFn>
f64 ks_critical_value(PFn&& pfn, f64 alpha) {
    f64 lo = 0.0;
    f64 hi = 1.0;
    for (i32 it = 0; it < 100; ++it) {
        const f64 mid = 0.5 * (lo + hi);
        if (pfn(mid) > alpha) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return 0.5 * (lo + hi);
}

f64 interp_cdf(ArrayRef<f64> xg, ArrayRef<f64> fg, f64 x) noexcept {
    if (xg.empty()) {
        return 0.0;
    }
    if (x <= xg[0]) {
        return fg[0];
    }
    if (x >= xg[xg.size() - 1]) {
        return fg[xg.size() - 1];
    }
    usize lo = 0;
    usize hi = xg.size();
    while (lo < hi) {
        const usize mid = lo + (hi - lo) / 2;
        if (xg[mid] <= x) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    const usize k = lo - 1;
    const f64 t = (x - xg[k]) / (xg[k + 1] - xg[k]);
    return fg[k] + t * (fg[k + 1] - fg[k]);
}

f64 jb_stat(const f64* x, usize n) noexcept {
    f64 mean_v = 0.0;
    for (usize i = 0; i < n; ++i) {
        mean_v += x[i];
    }
    mean_v /= as<f64>(n);
    f64 m2 = 0.0;
    f64 m3 = 0.0;
    f64 m4 = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 d = x[i] - mean_v;
        const f64 d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    const f64 dn = as<f64>(n);
    m2 /= dn;
    m3 /= dn;
    m4 /= dn;
    const f64 s = (m2 > 0.0) ? (m3 / pow(m2, 1.5)) : 0.0;
    const f64 k = (m2 > 0.0) ? (m4 / (m2 * m2)) : 0.0;
    return dn / 6.0 * (s * s + 0.25 * (k - 3.0) * (k - 3.0));
}

f64 log_binom(f64 n, f64 k) noexcept {
    if (k < 0.0 || k > n) {
        return -Limits<f64>::infinity();
    }
    return specfun::gammaln(n + 1.0) - specfun::gammaln(k + 1.0) - specfun::gammaln(n - k + 1.0);
}

f64 exact_runs_pmf(i32 r, i32 n1, i32 n0) noexcept {
    if (r < 2) {
        return 0.0;
    }
    const f64 log_c = log_binom(as<f64>(n1 + n0), as<f64>(n1));
    if ((r & 1) == 0) {
        const i32 k = r / 2;
        if (k < 1 || k > n1 || k > n0) {
            return 0.0;
        }
        const f64 lp = log(2.0) + log_binom(as<f64>(n1 - 1), as<f64>(k - 1)) +
                       log_binom(as<f64>(n0 - 1), as<f64>(k - 1)) - log_c;
        return exp(lp);
    }
    const i32 k = (r - 1) / 2;
    const f64 neg_inf = -Limits<f64>::infinity();
    f64 t1 = neg_inf;
    f64 t2 = neg_inf;
    if (k <= n1 - 1 && k - 1 >= 0 && k - 1 <= n0 - 1) {
        t1 = log_binom(as<f64>(n1 - 1), as<f64>(k)) + log_binom(as<f64>(n0 - 1), as<f64>(k - 1));
    }
    if (k - 1 >= 0 && k - 1 <= n1 - 1 && k <= n0 - 1) {
        t2 = log_binom(as<f64>(n1 - 1), as<f64>(k - 1)) + log_binom(as<f64>(n0 - 1), as<f64>(k));
    }
    if (is_inf(t1) && is_inf(t2)) {
        return 0.0;
    }
    const f64 lmax = max(t1, t2);
    const f64 lsum =
        lmax + log((is_inf(t1) ? 0.0 : exp(t1 - lmax)) + (is_inf(t2) ? 0.0 : exp(t2 - lmax)));
    return exp(lsum - log_c);
}

f64 median_inplace(f64* a, usize n) noexcept {
    sort(a, n);
    if ((n & 1) != 0) {
        return a[n / 2];
    }
    return 0.5 * (a[n / 2 - 1] + a[n / 2]);
}

i64 lcm_i64(i64 a, i64 b) noexcept {
    if (a == 0 || b == 0) {
        return 0;
    }
    i64 g = a;
    i64 h = b;
    while (h != 0) {
        const i64 t = g % h;
        g = h;
        h = t;
    }
    return a / g * b;
}

} // namespace

StatsStatus ttest_1sample(ArrayRef<f64> x, f64 m, f64 alpha, TestTail tail,
                          TTestResult& out) noexcept {
    const usize n = x.size();
    if (n < 2) {
        return StatsStatus::kInvalidArgument;
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        alpha = 0.05;
    }
    f64 mu_hat = 0.0;
    f64 var_hat = 0.0;
    sample_mean_var(x, mu_hat, var_hat);
    const f64 sd = sqrt(var_hat);
    const f64 se = sd / sqrt(as<f64>(n));
    const f64 t = (mu_hat - m) / se;
    const f64 df = as<f64>(n - 1);
    const f64 p = tail_from_cdf(t_cdf(t, df), tail);
    const i32 h = (p < alpha) ? 1 : 0;

    f64 clo = 0.0;
    f64 chi = 0.0;
    const f64 inf_v = Limits<f64>::infinity();
    switch (tail) {
        case TestTail::kBoth: {
            const f64 tcrit = t_inv(1.0 - 0.5 * alpha, df);
            clo = mu_hat - tcrit * se;
            chi = mu_hat + tcrit * se;
            break;
        }
        case TestTail::kRight: {
            const f64 tc = t_inv(1.0 - alpha, df);
            clo = mu_hat - tc * se;
            chi = inf_v;
            break;
        }
        case TestTail::kLeft: {
            const f64 tc = t_inv(1.0 - alpha, df);
            clo = -inf_v;
            chi = mu_hat + tc * se;
            break;
        }
    }
    out = TTestResult{h, p, clo, chi, t, df, sd};
    return StatsStatus::kOk;
}

StatsStatus ttest_2sample(ArrayRef<f64> x, ArrayRef<f64> y, f64 alpha, TestTail tail,
                          bool equal_var, TTest2Result& out) noexcept {
    const usize nx = x.size();
    const usize ny = y.size();
    if (nx < 2 || ny < 2) {
        return StatsStatus::kInvalidArgument;
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        alpha = 0.05;
    }
    f64 mx = 0.0, vx = 0.0, my = 0.0, vy = 0.0;
    sample_mean_var(x, mx, vx);
    sample_mean_var(y, my, vy);

    const f64 dnx = as<f64>(nx);
    const f64 dny = as<f64>(ny);
    f64 df = 0.0;
    f64 se = 0.0;
    f64 sd_pooled = 0.0;
    if (equal_var) {
        const f64 sp2 = ((dnx - 1.0) * vx + (dny - 1.0) * vy) / (dnx + dny - 2.0);
        se = sqrt(sp2 * (1.0 / dnx + 1.0 / dny));
        df = dnx + dny - 2.0;
        sd_pooled = sqrt(sp2);
    } else {
        const f64 tx = vx / dnx;
        const f64 ty = vy / dny;
        se = sqrt(tx + ty);
        const f64 num = (tx + ty) * (tx + ty);
        const f64 den = tx * tx / (dnx - 1.0) + ty * ty / (dny - 1.0);
        df = (den > 0.0) ? (num / den) : (dnx + dny - 2.0);
    }
    const f64 t = (mx - my) / se;
    const f64 p = tail_from_cdf(t_cdf(t, df), tail);
    const i32 h = (p < alpha) ? 1 : 0;
    const f64 tcrit = t_inv(1.0 - 0.5 * alpha, df);
    out = TTest2Result{
        h, p, (mx - my) - tcrit * se, (mx - my) + tcrit * se, t, df, sqrt(vx), sqrt(vy), sd_pooled};
    return StatsStatus::kOk;
}

StatsStatus ztest_1sample(ArrayRef<f64> x, f64 m, f64 sigma, f64 alpha, TestTail tail,
                          ZTestResult& out) noexcept {
    if (x.empty() || !(sigma > 0.0)) {
        return StatsStatus::kInvalidArgument;
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        alpha = 0.05;
    }
    f64 mu_hat = 0.0;
    f64 var_hat = 0.0;
    sample_mean_var(x, mu_hat, var_hat);
    const f64 se = sigma / sqrt(as<f64>(x.size()));
    const f64 z = (mu_hat - m) / se;
    const f64 p = tail_from_cdf(norm_cdf(z, 0.0, 1.0), tail);
    const i32 h = (p < alpha) ? 1 : 0;
    const f64 zcrit = norm_inv(1.0 - 0.5 * alpha, 0.0, 1.0);
    out = ZTestResult{h, p, mu_hat - zcrit * se, mu_hat + zcrit * se, z};
    return StatsStatus::kOk;
}

StatsStatus vartest_1sample(ArrayRef<f64> x, f64 v, f64 alpha, TestTail tail,
                            VarTestResult& out) noexcept {
    if (x.size() < 2 || !(v > 0.0)) {
        return StatsStatus::kInvalidArgument;
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        alpha = 0.05;
    }
    f64 mu_hat = 0.0;
    f64 var_hat = 0.0;
    sample_mean_var(x, mu_hat, var_hat);
    const f64 df = as<f64>(x.size() - 1);
    const f64 t_stat = df * var_hat / v;
    const f64 p = tail_from_cdf(chi2_cdf(t_stat, df), tail);
    const i32 h = (p < alpha) ? 1 : 0;
    const f64 chi_lo = chi2_inv(0.5 * alpha, df);
    const f64 chi_hi = chi2_inv(1.0 - 0.5 * alpha, df);
    const f64 ci_lo = (chi_hi > 0.0) ? (df * var_hat / chi_hi) : 0.0;
    const f64 ci_hi = (chi_lo > 0.0) ? (df * var_hat / chi_lo) : Limits<f64>::infinity();
    out = VarTestResult{h, p, ci_lo, ci_hi, t_stat, df};
    return StatsStatus::kOk;
}

StatsStatus vartest_2sample(ArrayRef<f64> x, ArrayRef<f64> y, f64 alpha, TestTail tail,
                            VarTest2Result& out) noexcept {
    if (x.size() < 2 || y.size() < 2) {
        return StatsStatus::kInvalidArgument;
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        alpha = 0.05;
    }
    f64 mx = 0.0, vx = 0.0, my = 0.0, vy = 0.0;
    sample_mean_var(x, mx, vx);
    sample_mean_var(y, my, vy);
    if (vy == 0.0) {
        return StatsStatus::kInvalidArgument;
    }
    const f64 f_stat = vx / vy;
    const f64 v1 = as<f64>(x.size() - 1);
    const f64 v2 = as<f64>(y.size() - 1);
    const f64 p = tail_from_cdf(f_cdf(f_stat, v1, v2), tail);
    const i32 h = (p < alpha) ? 1 : 0;
    const f64 f_lo = f_inv(0.5 * alpha, v1, v2);
    const f64 f_hi = f_inv(1.0 - 0.5 * alpha, v1, v2);
    const f64 ci_lo = (f_hi > 0.0) ? (f_stat / f_hi) : 0.0;
    const f64 ci_hi = (f_lo > 0.0) ? (f_stat / f_lo) : Limits<f64>::infinity();
    out = VarTest2Result{h, p, ci_lo, ci_hi, f_stat, v1, v2};
    return StatsStatus::kOk;
}

StatsStatus kstest_1sample(ArrayRef<f64> x, ArrayRef<f64> cdf_x, ArrayRef<f64> cdf_f, f64 alpha,
                           TestTail tail, KsTestResult& out, Allocator* alloc) {
    const usize n = x.size();
    if (n == 0 || cdf_x.size() != cdf_f.size()) {
        return StatsStatus::kInvalidArgument;
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        alpha = 0.05;
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> xs(x, scratch.allocator());
    sort(xs.as_mut_ref());

    Array<f64> xg(scratch.allocator());
    Array<f64> fg(scratch.allocator());
    const bool use_table = !cdf_x.empty();
    if (use_table) {
        const usize rows = cdf_x.size();
        Array<usize> ord(rows, usize{0}, scratch.allocator());
        for (usize i = 0; i < rows; ++i) {
            ord[i] = i;
        }
        sort(ord.as_mut_ref(), [&](usize a, usize b) noexcept { return cdf_x[a] < cdf_x[b]; });
        xg.resize(rows);
        fg.resize(rows);
        for (usize i = 0; i < rows; ++i) {
            xg[i] = cdf_x[ord[i]];
            fg[i] = cdf_f[ord[i]];
        }
    }

    f64 d_plus = 0.0;
    f64 d_minus = 0.0;
    const f64 dn = as<f64>(n);
    for (usize i = 0; i < n; ++i) {
        const f64 f_ref =
            use_table ? interp_cdf(xg.as_ref(), fg.as_ref(), xs[i]) : norm_cdf(xs[i], 0.0, 1.0);
        const f64 up = as<f64>(i + 1) / dn - f_ref;
        const f64 down = f_ref - as<f64>(i) / dn;
        if (up > d_plus) {
            d_plus = up;
        }
        if (down > d_minus) {
            d_minus = down;
        }
    }
    f64 d_stat = max(d_plus, d_minus);
    if (tail == TestTail::kRight) {
        d_stat = d_plus;
    } else if (tail == TestTail::kLeft) {
        d_stat = d_minus;
    }

    const i32 ni = as<i32>(n);
    f64 p = 1.0;
    f64 cv = 0.0;
    if (tail == TestTail::kBoth) {
        p = ks_one_sample_two_sided_p(ni, d_stat, scratch.allocator());
        cv = ks_critical_value(
            [ni, &scratch](f64 dd) { return ks_one_sample_two_sided_p(ni, dd, scratch.allocator()); },
            alpha);
    } else {
        p = birnbaum_tingey(ni, d_stat);
        cv = ks_critical_value([ni](f64 dd) noexcept { return birnbaum_tingey(ni, dd); }, alpha);
    }
    out = KsTestResult{(p < alpha) ? 1 : 0, p, d_stat, cv};
    return StatsStatus::kOk;
}

StatsStatus kstest_2sample(ArrayRef<f64> x, ArrayRef<f64> y, f64 alpha, TestTail tail,
                           KsTestResult& out, Allocator* alloc) {
    const usize nx = x.size();
    const usize ny = y.size();
    if (nx == 0 || ny == 0) {
        return StatsStatus::kInvalidArgument;
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        alpha = 0.05;
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> xs(x, scratch.allocator());
    Array<f64> ys(y, scratch.allocator());
    sort(xs.as_mut_ref());
    sort(ys.as_mut_ref());

    f64 d_plus = 0.0;
    f64 d_minus = 0.0;
    usize i = 0;
    usize j = 0;
    const f64 dnx = as<f64>(nx);
    const f64 dny = as<f64>(ny);
    while (i < nx && j < ny) {
        const f64 a = xs[i];
        const f64 b = ys[j];
        if (a <= b) {
            ++i;
        }
        if (b <= a) {
            ++j;
        }
        const f64 fx = as<f64>(i) / dnx;
        const f64 fy = as<f64>(j) / dny;
        if (fx - fy > d_plus) {
            d_plus = fx - fy;
        }
        if (fy - fx > d_minus) {
            d_minus = fy - fx;
        }
    }
    f64 d_stat = max(d_plus, d_minus);
    if (tail == TestTail::kRight) {
        d_stat = d_plus;
    } else if (tail == TestTail::kLeft) {
        d_stat = d_minus;
    }

    const f64 n_eff = dnx * dny / (dnx + dny);
    const bool two_sided = (tail == TestTail::kBoth);
    const f64 p = ks_two_sample_p(n_eff, d_stat, two_sided);
    const f64 cv = ks_critical_value(
        [n_eff, two_sided](f64 dd) noexcept { return ks_two_sample_p(n_eff, dd, two_sided); },
        alpha);
    out = KsTestResult{(p < alpha) ? 1 : 0, p, d_stat, cv};
    return StatsStatus::kOk;
}

StatsStatus jbtest(ArrayRef<f64> x, f64 alpha, f64 mctol, JbTestResult& out, Allocator* alloc) {
    const usize n = x.size();
    if (n < 4) {
        return StatsStatus::kInvalidArgument;
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        alpha = 0.05;
    }
    const f64 jb = jb_stat(x.data(), n);
    f64 p = 1.0;
    f64 cv = 0.0;
    const bool use_mc = is_finite(mctol) || n < 2000;
    const f64 mctol_eff = is_finite(mctol) ? mctol : 1e-3;

    if (use_mc) {
        constexpr usize kBatch = 1000;
        constexpr usize kMaxReps = 1000000;
        constexpr f64 kCapP = 0.5;
        random::Xoshiro256PlusPlus gen(u64{12345});
        ArenaTemp scratch = get_scratch(alloc);
        Array<f64> buf(n, 0.0, scratch.allocator());
        Array<f64> jb_h0(scratch.allocator());
        jb_h0.reserve(kBatch * 4);

        usize total = 0;
        usize exceeds = 0;
        while (total < kMaxReps) {
            const usize this_batch = min(kBatch, kMaxReps - total);
            for (usize i = 0; i < this_batch; ++i) {
                for (usize k = 0; k < n; ++k) {
                    buf[k] = random::normal_f64(gen);
                }
                const f64 j = jb_stat(buf.data(), n);
                jb_h0.push_back(j);
                if (j >= jb) {
                    ++exceeds;
                }
            }
            total += this_batch;
            const f64 p_hat = as<f64>(exceeds) / as<f64>(total);
            const f64 se = sqrt(max(p_hat * (1.0 - p_hat), 1.0 / as<f64>(total)) / as<f64>(total));
            if (se < mctol_eff && total >= 3 * kBatch) {
                break;
            }
        }
        sort(jb_h0.as_mut_ref());
        const f64 rank = (1.0 - alpha) * as<f64>(jb_h0.size() - 1);
        const usize lo = as<usize>(floor(rank));
        const usize hi = min(lo + 1, jb_h0.size() - 1);
        const f64 frac = rank - as<f64>(lo);
        cv = jb_h0[lo] * (1.0 - frac) + jb_h0[hi] * frac;
        p = min(as<f64>(exceeds) / as<f64>(total), kCapP);
    } else {
        p = 1.0 - chi2_cdf(jb, 2.0);
        cv = chi2_inv(1.0 - alpha, 2.0);
    }
    out = JbTestResult{(p < alpha) ? 1 : 0, p, jb, cv};
    return StatsStatus::kOk;
}

StatsStatus signtest(ArrayRef<f64> x, ArrayRef<f64> y, f64 m0, f64 alpha, TestTail tail,
                     SignTestResult& out) noexcept {
    const bool paired = !y.empty();
    if (paired && y.size() != x.size()) {
        return StatsStatus::kInvalidArgument;
    }
    i64 n_pos = 0;
    i64 n_eff = 0;
    for (usize i = 0; i < x.size(); ++i) {
        const f64 xi = x[i];
        const f64 yi = paired ? y[i] : m0;
        if (is_nan(xi) || is_nan(yi)) {
            continue;
        }
        const f64 d = xi - yi;
        if (d == 0.0) {
            continue;
        }
        ++n_eff;
        if (d > 0.0) {
            ++n_pos;
        }
    }

    f64 p = 1.0;
    if (n_eff > 0) {
        const f64 cdf_le = bino_cdf(as<f64>(n_pos), as<f64>(n_eff), 0.5);
        const f64 cdf_lt = (n_pos > 0) ? bino_cdf(as<f64>(n_pos - 1), as<f64>(n_eff), 0.5) : 0.0;
        const f64 p_left = cdf_le;
        const f64 p_right = 1.0 - cdf_lt;
        switch (tail) {
            case TestTail::kBoth: p = min(1.0, 2.0 * min(p_left, p_right)); break;
            case TestTail::kRight: p = p_right; break;
            case TestTail::kLeft: p = p_left; break;
        }
    }
    out = SignTestResult{p, (p < alpha) ? 1 : 0, n_pos};
    return StatsStatus::kOk;
}

StatsStatus fishertest(f64 a, f64 b, f64 c, f64 d, f64 alpha, TestTail tail, FisherTestResult& out,
                       Allocator* alloc) {
    const f64 row1 = a + b;
    const f64 col1 = a + c;
    const f64 ntot = a + b + c + d;
    const i64 kmin = as<i64>(max(0.0, row1 + col1 - ntot));
    const i64 kmax = as<i64>(min(row1, col1));
    const i64 kobs = as<i64>(a);
    if (kmax < kmin || kobs < kmin || kobs > kmax) {
        return StatsStatus::kInvalidArgument;
    }

    const f64 log_c_den = log_binom(ntot, col1);
    const usize k_cnt = as<usize>(kmax - kmin + 1);
    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> pmf(k_cnt, 0.0, scratch.allocator());
    for (i64 k = kmin; k <= kmax; ++k) {
        const f64 dk = as<f64>(k);
        const f64 lp = log_binom(row1, dk) + log_binom(ntot - row1, col1 - dk) - log_c_den;
        pmf[as<usize>(k - kmin)] = exp(lp);
    }
    const f64 p_obs = pmf[as<usize>(kobs - kmin)];

    f64 p = 0.0;
    switch (tail) {
        case TestTail::kBoth: {
            const f64 tol = p_obs * (1.0 + 1e-9);
            for (usize i = 0; i < k_cnt; ++i) {
                if (pmf[i] <= tol) {
                    p += pmf[i];
                }
            }
            break;
        }
        case TestTail::kRight:
            for (i64 k = kobs; k <= kmax; ++k) {
                p += pmf[as<usize>(k - kmin)];
            }
            break;
        case TestTail::kLeft:
            for (i64 k = kmin; k <= kobs; ++k) {
                p += pmf[as<usize>(k - kmin)];
            }
            break;
    }
    if (p > 1.0) {
        p = 1.0;
    }

    const f64 inf_v = Limits<f64>::infinity();
    const f64 odds = (b * c == 0.0) ? inf_v : ((a * d) / (b * c));
    f64 ci_lo = 0.0;
    f64 ci_hi = inf_v;
    if (a > 0.0 && b > 0.0 && c > 0.0 && d > 0.0) {
        const f64 log_or = log(odds);
        const f64 se = sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d);
        const f64 zcrit = norm_inv(1.0 - 0.5 * alpha, 0.0, 1.0);
        ci_lo = exp(log_or - zcrit * se);
        ci_hi = exp(log_or + zcrit * se);
    }
    out = FisherTestResult{(p < alpha) ? 1 : 0, p, odds, ci_lo, ci_hi};
    return StatsStatus::kOk;
}

StatsStatus chi2gof_freq(ArrayRef<f64> observed, ArrayRef<f64> expected, i32 nparams, f64 alpha,
                         Chi2GofResult& out) noexcept {
    const usize k = observed.size();
    if (expected.size() != k) {
        return StatsStatus::kInvalidArgument;
    }
    const f64 qnan = Limits<f64>::quiet_nan();
    if (k < 2) {
        out = Chi2GofResult{qnan, 0, qnan, 0.0};
        return StatsStatus::kOk;
    }
    f64 chi2 = 0.0;
    for (usize i = 0; i < k; ++i) {
        if (expected[i] <= 0.0) {
            continue;
        }
        const f64 d = observed[i] - expected[i];
        chi2 += d * d / expected[i];
    }
    const f64 df = as<f64>(k) - 1.0 - as<f64>(nparams);
    if (df <= 0.0) {
        out = Chi2GofResult{qnan, 0, chi2, df};
        return StatsStatus::kOk;
    }
    const f64 p = max(0.0, 1.0 - chi2_cdf(chi2, df));
    out = Chi2GofResult{p, (p < alpha) ? 1 : 0, chi2, df};
    return StatsStatus::kOk;
}

StatsStatus vartestn(ArrayRef<f64> x, ArrayRef<f64> group, VarTestNMethod method,
                     VarTestNResult& out, Allocator* alloc) {
    const usize nx = x.size();
    if (nx == 0 || group.size() != nx) {
        return StatsStatus::kInvalidArgument;
    }
    const f64 qnan = Limits<f64>::quiet_nan();

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> labels(scratch.allocator());
    Array<usize> counts(scratch.allocator());
    Array<i32> bucket_for(nx, -1, scratch.allocator());
    for (usize i = 0; i < nx; ++i) {
        if (is_nan(x[i]) || is_nan(group[i])) {
            continue;
        }
        i32 b = -1;
        for (usize j = 0; j < labels.size(); ++j) {
            if (labels[j] == group[i]) {
                b = as<i32>(j);
                break;
            }
        }
        if (b < 0) {
            b = as<i32>(labels.size());
            labels.push_back(group[i]);
            counts.push_back(0);
        }
        bucket_for[i] = b;
        ++counts[as<usize>(b)];
    }

    const usize k_all = labels.size();
    Array<usize> raw_off(k_all + 1, usize{0}, scratch.allocator());
    for (usize i = 0; i < k_all; ++i) {
        raw_off[i + 1] = raw_off[i] + counts[i];
    }
    Array<f64> raw_data(raw_off[k_all], 0.0, scratch.allocator());
    Array<usize> cursor(k_all, usize{0}, scratch.allocator());
    for (usize i = 0; i < nx; ++i) {
        const i32 b = bucket_for[i];
        if (b >= 0) {
            const usize bu = as<usize>(b);
            raw_data[raw_off[bu] + cursor[bu]++] = x[i];
        }
    }

    Array<usize> ns(scratch.allocator());
    Array<usize> offsets(scratch.allocator());
    Array<f64> data(scratch.allocator());
    offsets.push_back(0);
    for (usize g = 0; g < k_all; ++g) {
        if (counts[g] < 2) {
            continue;
        }
        ns.push_back(counts[g]);
        for (usize j = raw_off[g]; j < raw_off[g + 1]; ++j) {
            data.push_back(raw_data[j]);
        }
        offsets.push_back(data.size());
    }

    const usize k = ns.size();
    if (k < 2) {
        out = VarTestNResult{qnan, 0.0, as<f64>(k > 0 ? k - 1 : 0), qnan};
        return StatsStatus::kOk;
    }

    usize total_n = 0;
    for (usize i = 0; i < k; ++i) {
        total_n += ns[i];
    }

    Array<f64> means(k, 0.0, scratch.allocator());
    Array<f64> meds(k, 0.0, scratch.allocator());
    Array<f64> vars(k, 0.0, scratch.allocator());
    for (usize g = 0; g < k; ++g) {
        const usize lo = offsets[g];
        const usize hi = offsets[g + 1];
        f64 s = 0.0;
        for (usize j = lo; j < hi; ++j) {
            s += data[j];
        }
        means[g] = s / as<f64>(ns[g]);
        f64 s2 = 0.0;
        for (usize j = lo; j < hi; ++j) {
            const f64 d = data[j] - means[g];
            s2 += d * d;
        }
        vars[g] = s2 / as<f64>(ns[g] - 1);
        Array<f64> tmp(ArrayRef<f64>(data.data() + lo, hi - lo), scratch.allocator());
        meds[g] = median_inplace(tmp.data(), tmp.size());
    }

    if (method == VarTestNMethod::kBartlett) {
        f64 sp2_num = 0.0;
        for (usize i = 0; i < k; ++i) {
            sp2_num += as<f64>(ns[i] - 1) * vars[i];
        }
        const f64 sp2 = sp2_num / as<f64>(total_n - k);
        f64 q = as<f64>(total_n - k) * log(sp2);
        for (usize i = 0; i < k; ++i) {
            q -= as<f64>(ns[i] - 1) * log(vars[i]);
        }
        f64 inv_sum = 0.0;
        for (usize i = 0; i < k; ++i) {
            inv_sum += 1.0 / as<f64>(ns[i] - 1);
        }
        inv_sum -= 1.0 / as<f64>(total_n - k);
        const f64 c_corr = 1.0 + inv_sum / (3.0 * as<f64>(k - 1));
        const f64 chisq = q / c_corr;
        const f64 df = as<f64>(k - 1);
        const f64 p = max(0.0, 1.0 - chi2_cdf(chisq, df));
        out = VarTestNResult{p, chisq, df, qnan};
        return StatsStatus::kOk;
    }

    Array<f64> z_vals(data.size(), 0.0, scratch.allocator());
    for (usize g = 0; g < k; ++g) {
        const usize lo = offsets[g];
        const usize hi = offsets[g + 1];
        for (usize j = lo; j < hi; ++j) {
            const f64 v = data[j];
            switch (method) {
                case VarTestNMethod::kLeveneQuadratic: {
                    const f64 d = v - means[g];
                    z_vals[j] = d * d;
                    break;
                }
                case VarTestNMethod::kLeveneAbsolute: z_vals[j] = abs(v - means[g]); break;
                case VarTestNMethod::kBrownForsythe: z_vals[j] = abs(v - meds[g]); break;
                case VarTestNMethod::kOBrien: {
                    const f64 ng = as<f64>(ns[g]);
                    const f64 dev = (v - means[g]) * (v - means[g]);
                    z_vals[j] = ((ng - 1.5) * ng * dev - 0.5 * vars[g] * (ng - 1.0)) /
                                ((ng - 1.0) * (ng - 2.0));
                    break;
                }
                default: z_vals[j] = 0.0; break;
            }
        }
    }

    Array<f64> zbar(k, 0.0, scratch.allocator());
    f64 grand = 0.0;
    for (usize g = 0; g < k; ++g) {
        f64 s = 0.0;
        for (usize j = offsets[g]; j < offsets[g + 1]; ++j) {
            s += z_vals[j];
        }
        zbar[g] = s / as<f64>(ns[g]);
        grand += s;
    }
    grand /= as<f64>(total_n);

    f64 ssb = 0.0;
    f64 ssw = 0.0;
    for (usize g = 0; g < k; ++g) {
        const f64 db = zbar[g] - grand;
        ssb += as<f64>(ns[g]) * db * db;
        for (usize j = offsets[g]; j < offsets[g + 1]; ++j) {
            const f64 dw = z_vals[j] - zbar[g];
            ssw += dw * dw;
        }
    }
    const f64 df1 = as<f64>(k - 1);
    const f64 df2 = as<f64>(total_n - k);
    const f64 f_stat = (ssw > 0.0) ? ((ssb / df1) / (ssw / df2)) : 0.0;
    const f64 p = max(0.0, 1.0 - f_cdf(f_stat, df1, df2));
    out = VarTestNResult{p, f_stat, df1, df2};
    return StatsStatus::kOk;
}

StatsStatus runstest(ArrayRef<f64> x, f64 threshold, f64 alpha, TestTail tail, bool exact,
                     RunsTestResult& out, Allocator* alloc) {
    ArenaTemp scratch = get_scratch(alloc);
    f64 v = threshold;
    if (is_nan(v)) {
        Array<f64> sorted(scratch.allocator());
        sorted.reserve(x.size());
        for (usize i = 0; i < x.size(); ++i) {
            if (!is_nan(x[i])) {
                sorted.push_back(x[i]);
            }
        }
        v = sorted.empty() ? 0.0 : median_inplace(sorted.data(), sorted.size());
    }

    Array<i32> bin(scratch.allocator());
    bin.reserve(x.size());
    for (usize i = 0; i < x.size(); ++i) {
        if (is_nan(x[i])) {
            continue;
        }
        if (x[i] > v) {
            bin.push_back(1);
        } else if (x[i] < v) {
            bin.push_back(0);
        }
    }

    i32 n1 = 0;
    i32 n0 = 0;
    i32 runs = 0;
    i32 prev = -1;
    for (usize i = 0; i < bin.size(); ++i) {
        if (bin[i] != 0) {
            ++n1;
        } else {
            ++n0;
        }
        if (bin[i] != prev) {
            ++runs;
            prev = bin[i];
        }
    }

    const f64 qnan = Limits<f64>::quiet_nan();
    if (n1 == 0 || n0 == 0) {
        out = RunsTestResult{qnan, 0, runs, n1, n0, qnan};
        return StatsStatus::kOk;
    }

    f64 p = 1.0;
    f64 zval = qnan;
    if (exact) {
        const i32 total = n1 + n0;
        f64 cdf_le = 0.0;
        f64 cdf_ge = 0.0;
        for (i32 r = 2; r <= total; ++r) {
            const f64 pr = exact_runs_pmf(r, n1, n0);
            if (r <= runs) {
                cdf_le += pr;
            }
            if (r >= runs) {
                cdf_ge += pr;
            }
        }
        switch (tail) {
            case TestTail::kBoth: p = min(1.0, 2.0 * min(cdf_le, cdf_ge)); break;
            case TestTail::kRight: p = cdf_ge; break;
            case TestTail::kLeft: p = cdf_le; break;
        }
    } else {
        const f64 dn1 = as<f64>(n1);
        const f64 dn0 = as<f64>(n0);
        const f64 nd = dn1 + dn0;
        const f64 mu = 2.0 * dn1 * dn0 / nd + 1.0;
        const f64 vr = 2.0 * dn1 * dn0 * (2.0 * dn1 * dn0 - nd) / (nd * nd * (nd - 1.0));
        const f64 sd = sqrt(max(vr, 0.0));
        f64 cc = 0.0;
        if (as<f64>(runs) > mu) {
            cc = 0.5;
        } else if (as<f64>(runs) < mu) {
            cc = -0.5;
        }
        zval = (sd > 0.0) ? ((as<f64>(runs) - mu - cc) / sd) : 0.0;
        p = tail_from_cdf(norm_cdf(zval, 0.0, 1.0), tail);
    }

    out = RunsTestResult{p, (p < alpha) ? 1 : 0, runs, n1, n0, zval};
    return StatsStatus::kOk;
}

StatsStatus ranksum(ArrayRef<f64> x, ArrayRef<f64> y, f64 alpha, TestTail tail, i32 method_mode,
                    RankTestResult& out, Allocator* alloc) {
    const f64 qnan = Limits<f64>::quiet_nan();
    if (x.size() + y.size() == 0) {
        out = RankTestResult{1.0, 0, 0.0, qnan};
        return StatsStatus::kOk;
    }

    struct Item {
        f64 v;
        bool is_x;
    };
    ArenaTemp scratch = get_scratch(alloc);
    Array<Item> items(scratch.allocator());
    items.reserve(x.size() + y.size());
    usize nx_eff = 0;
    usize ny_eff = 0;
    for (usize i = 0; i < x.size(); ++i) {
        if (!is_nan(x[i])) {
            items.push_back(Item{x[i], true});
            ++nx_eff;
        }
    }
    for (usize i = 0; i < y.size(); ++i) {
        if (!is_nan(y[i])) {
            items.push_back(Item{y[i], false});
            ++ny_eff;
        }
    }
    const usize neff = items.size();
    if (neff == 0 || nx_eff == 0 || ny_eff == 0) {
        out = RankTestResult{1.0, 0, 0.0, qnan};
        return StatsStatus::kOk;
    }

    Array<usize> ord(neff, usize{0}, scratch.allocator());
    for (usize i = 0; i < neff; ++i) {
        ord[i] = i;
    }
    sort(ord.as_mut_ref(), [&](usize a, usize b) noexcept { return items[a].v < items[b].v; });

    Array<f64> ranks(neff, 0.0, scratch.allocator());
    Array<usize> tie_sizes(scratch.allocator());
    for (usize i = 0; i < neff;) {
        usize j = i + 1;
        while (j < neff && items[ord[j]].v == items[ord[i]].v) {
            ++j;
        }
        const f64 avg = as<f64>(i + j + 1) * 0.5;
        for (usize k = i; k < j; ++k) {
            ranks[ord[k]] = avg;
        }
        if (j - i > 1) {
            tie_sizes.push_back(j - i);
        }
        i = j;
    }

    f64 wx = 0.0;
    for (usize k = 0; k < neff; ++k) {
        if (items[k].is_x) {
            wx += ranks[k];
        }
    }

    const bool exact = (method_mode == 1) ? true : (method_mode == 2 ? false : (nx_eff < 10 && ny_eff < 10));
    f64 p = 1.0;
    f64 zval = qnan;

    if (exact) {
        Array<i64> rk2(neff, i64{0}, scratch.allocator());
        i64 total2 = 0;
        for (usize k = 0; k < neff; ++k) {
            rk2[k] = as<i64>(round(ranks[k] * 2.0));
            total2 += rk2[k];
        }
        const usize s_cap = as<usize>(total2 + 1);
        Array<f64> dp((nx_eff + 1) * s_cap, 0.0, scratch.allocator());
        dp[0] = 1.0;
        for (usize k = 0; k < neff; ++k) {
            const i64 r = rk2[k];
            for (i64 kk = as<i64>(nx_eff) - 1; kk >= 0; --kk) {
                for (i64 s = as<i64>(s_cap) - 1; s >= r; --s) {
                    dp[as<usize>(kk + 1) * s_cap + as<usize>(s)] +=
                        dp[as<usize>(kk) * s_cap + as<usize>(s - r)];
                }
            }
        }
        const i64 wx2 = as<i64>(round(wx * 2.0));
        f64 total = 0.0;
        for (usize s = 0; s < s_cap; ++s) {
            total += dp[nx_eff * s_cap + s];
        }
        f64 cdf_le = 0.0;
        f64 cdf_ge = 0.0;
        for (i64 s = 0; s <= wx2 && s < as<i64>(s_cap); ++s) {
            cdf_le += dp[nx_eff * s_cap + as<usize>(s)];
        }
        for (i64 s = wx2; s < as<i64>(s_cap); ++s) {
            cdf_ge += dp[nx_eff * s_cap + as<usize>(s)];
        }
        cdf_le /= total;
        cdf_ge /= total;
        switch (tail) {
            case TestTail::kBoth: p = min(1.0, 2.0 * min(cdf_le, cdf_ge)); break;
            case TestTail::kRight: p = cdf_ge; break;
            case TestTail::kLeft: p = cdf_le; break;
        }
    } else {
        const f64 nd = as<f64>(neff);
        const f64 dnx = as<f64>(nx_eff);
        const f64 dny = as<f64>(ny_eff);
        const f64 mu = dnx * (nd + 1.0) * 0.5;
        f64 vr = dnx * dny * (nd + 1.0) / 12.0;
        if (nd > 1.0) {
            f64 tie_sum = 0.0;
            for (usize i = 0; i < tie_sizes.size(); ++i) {
                const f64 td = as<f64>(tie_sizes[i]);
                tie_sum += td * td * td - td;
            }
            vr -= dnx * dny / (12.0 * nd * (nd - 1.0)) * tie_sum;
        }
        const f64 sd = sqrt(max(vr, 0.0));
        f64 cc = 0.0;
        if (wx > mu) {
            cc = 0.5;
        } else if (wx < mu) {
            cc = -0.5;
        }
        zval = (sd > 0.0) ? ((wx - mu - cc) / sd) : 0.0;
        p = tail_from_cdf(norm_cdf(zval, 0.0, 1.0), tail);
    }

    out = RankTestResult{p, (p < alpha) ? 1 : 0, wx, zval};
    return StatsStatus::kOk;
}

StatsStatus signrank(ArrayRef<f64> x, ArrayRef<f64> y, f64 m0, f64 alpha, TestTail tail,
                     i32 method_mode, RankTestResult& out, Allocator* alloc) {
    const bool paired = !y.empty();
    if (paired && y.size() != x.size()) {
        return StatsStatus::kInvalidArgument;
    }

    struct DSign {
        f64 absd;
        i32 sign;
    };
    ArenaTemp scratch = get_scratch(alloc);
    Array<DSign> ds(scratch.allocator());
    ds.reserve(x.size());
    for (usize i = 0; i < x.size(); ++i) {
        const f64 xi = x[i];
        const f64 yi = paired ? y[i] : m0;
        if (is_nan(xi) || is_nan(yi)) {
            continue;
        }
        const f64 d = xi - yi;
        if (d == 0.0) {
            continue;
        }
        ds.push_back(DSign{abs(d), (d > 0.0) ? 1 : -1});
    }

    const f64 qnan = Limits<f64>::quiet_nan();
    const usize n = ds.size();
    if (n == 0) {
        out = RankTestResult{1.0, 0, 0.0, qnan};
        return StatsStatus::kOk;
    }

    Array<usize> ord(n, usize{0}, scratch.allocator());
    for (usize i = 0; i < n; ++i) {
        ord[i] = i;
    }
    sort(ord.as_mut_ref(), [&](usize a, usize b) noexcept { return ds[a].absd < ds[b].absd; });

    Array<f64> ranks(n, 0.0, scratch.allocator());
    Array<usize> tie_sizes(scratch.allocator());
    for (usize i = 0; i < n;) {
        usize j = i + 1;
        while (j < n && ds[ord[j]].absd == ds[ord[i]].absd) {
            ++j;
        }
        const f64 avg = as<f64>(i + j + 1) * 0.5;
        for (usize k = i; k < j; ++k) {
            ranks[ord[k]] = avg;
        }
        if (j - i > 1) {
            tie_sizes.push_back(j - i);
        }
        i = j;
    }

    f64 w_plus = 0.0;
    for (usize k = 0; k < n; ++k) {
        if (ds[k].sign > 0) {
            w_plus += ranks[k];
        }
    }

    const bool exact = (method_mode == 1) ? true : (method_mode == 2 ? false : (n <= 15));
    f64 p = 1.0;
    f64 zval = qnan;

    if (exact) {
        Array<i64> rk2(n, i64{0}, scratch.allocator());
        i64 total2 = 0;
        for (usize k = 0; k < n; ++k) {
            rk2[k] = as<i64>(round(ranks[k] * 2.0));
            total2 += rk2[k];
        }
        Array<f64> poly(as<usize>(total2 + 1), 0.0, scratch.allocator());
        poly[0] = 1.0;
        i64 cur = 0;
        for (usize k = 0; k < n; ++k) {
            const i64 r = rk2[k];
            for (i64 s = cur + r; s >= r; --s) {
                poly[as<usize>(s)] += poly[as<usize>(s - r)];
            }
            cur += r;
        }
        const i64 w2 = as<i64>(round(w_plus * 2.0));
        const f64 total = pow(2.0, as<f64>(n));
        f64 cdf_le = 0.0;
        for (i64 s = 0; s <= w2; ++s) {
            cdf_le += poly[as<usize>(s)];
        }
        cdf_le /= total;
        f64 cdf_ge = 0.0;
        for (i64 s = w2; s <= total2; ++s) {
            cdf_ge += poly[as<usize>(s)];
        }
        cdf_ge /= total;
        switch (tail) {
            case TestTail::kBoth: p = min(1.0, 2.0 * min(cdf_le, cdf_ge)); break;
            case TestTail::kRight: p = cdf_ge; break;
            case TestTail::kLeft: p = cdf_le; break;
        }
    } else {
        const f64 dn = as<f64>(n);
        const f64 mu = dn * (dn + 1.0) * 0.25;
        f64 vr = dn * (dn + 1.0) * (2.0 * dn + 1.0) / 24.0;
        for (usize i = 0; i < tie_sizes.size(); ++i) {
            const f64 td = as<f64>(tie_sizes[i]);
            vr -= (td * td * td - td) / 48.0;
        }
        const f64 sd = sqrt(max(vr, 0.0));
        zval = (sd > 0.0) ? ((w_plus - mu) / sd) : 0.0;
        p = tail_from_cdf(norm_cdf(zval, 0.0, 1.0), tail);
    }

    out = RankTestResult{p, (p < alpha) ? 1 : 0, w_plus, zval};
    return StatsStatus::kOk;
}

StatsStatus ansaribradley(ArrayRef<f64> x, ArrayRef<f64> y, f64 alpha, TestTail tail,
                          RankTestResult& out, Allocator* alloc) {
    const f64 qnan = Limits<f64>::quiet_nan();
    if (x.empty() || y.empty()) {
        out = RankTestResult{1.0, 0, 0.0, qnan};
        return StatsStatus::kOk;
    }

    struct Item {
        f64 v;
        bool is_x;
    };
    ArenaTemp scratch = get_scratch(alloc);
    Array<Item> items(scratch.allocator());
    items.reserve(x.size() + y.size());
    usize m = 0;
    usize n = 0;
    for (usize i = 0; i < x.size(); ++i) {
        if (!is_nan(x[i])) {
            items.push_back(Item{x[i], true});
            ++m;
        }
    }
    for (usize i = 0; i < y.size(); ++i) {
        if (!is_nan(y[i])) {
            items.push_back(Item{y[i], false});
            ++n;
        }
    }
    const usize total_n = items.size();
    if (m == 0 || n == 0) {
        out = RankTestResult{1.0, 0, 0.0, qnan};
        return StatsStatus::kOk;
    }

    Array<usize> ord(total_n, usize{0}, scratch.allocator());
    for (usize i = 0; i < total_n; ++i) {
        ord[i] = i;
    }
    sort(ord.as_mut_ref(), [&](usize a, usize b) noexcept { return items[a].v < items[b].v; });

    Array<f64> rraw(total_n, 0.0, scratch.allocator());
    for (usize i = 0; i < total_n; ++i) {
        const usize pos = i + 1;
        const usize alt = total_n + 1 - pos;
        rraw[i] = as<f64>(min(pos, alt));
    }

    Array<f64> rsorted(total_n, 0.0, scratch.allocator());
    i64 scale = 1;
    const bool exact_path = (min(m, n) <= 10);
    for (usize i = 0; i < total_n;) {
        usize j = i + 1;
        while (j < total_n && items[ord[j]].v == items[ord[i]].v) {
            ++j;
        }
        f64 s = 0.0;
        for (usize k = i; k < j; ++k) {
            s += rraw[k];
        }
        const f64 avg = s / as<f64>(j - i);
        for (usize k = i; k < j; ++k) {
            rsorted[k] = avg;
        }
        if (exact_path) {
            scale = lcm_i64(scale, as<i64>(j - i));
            if (scale > 10000) {
                scale = 10000;
            }
        }
        i = j;
    }

    Array<f64> ranks(total_n, 0.0, scratch.allocator());
    for (usize k = 0; k < total_n; ++k) {
        ranks[ord[k]] = rsorted[k];
    }
    f64 w_stat = 0.0;
    for (usize k = 0; k < total_n; ++k) {
        if (items[k].is_x) {
            w_stat += ranks[k];
        }
    }

    f64 sum_r = 0.0;
    for (usize k = 0; k < total_n; ++k) {
        sum_r += rsorted[k];
    }
    const f64 nd = as<f64>(total_n);
    const f64 mean_r = sum_r / nd;
    f64 ss = 0.0;
    for (usize k = 0; k < total_n; ++k) {
        const f64 d = rsorted[k] - mean_r;
        ss += d * d;
    }
    const f64 ew = as<f64>(m) * mean_r;
    const f64 vw = as<f64>(m) * as<f64>(n) / (nd - 1.0) * (ss / nd);
    const f64 sd = sqrt(max(vw, 0.0));
    const f64 w_star = (sd > 0.0) ? ((w_stat - ew) / sd) : 0.0;

    f64 p_val = 1.0;
    if (exact_path) {
        Array<i64> rk2(total_n, i64{0}, scratch.allocator());
        i64 total2 = 0;
        for (usize k = 0; k < total_n; ++k) {
            rk2[k] = as<i64>(round(rsorted[k] * as<f64>(scale)));
            total2 += rk2[k];
        }
        const usize s_cap = as<usize>(total2 + 1);
        Array<f64> dp((m + 1) * s_cap, 0.0, scratch.allocator());
        dp[0] = 1.0;
        for (usize k = 0; k < total_n; ++k) {
            const i64 r = rk2[k];
            for (i64 kk = as<i64>(m) - 1; kk >= 0; --kk) {
                for (i64 s = as<i64>(s_cap) - 1; s >= r; --s) {
                    dp[as<usize>(kk + 1) * s_cap + as<usize>(s)] +=
                        dp[as<usize>(kk) * s_cap + as<usize>(s - r)];
                }
            }
        }
        const i64 w_scaled = as<i64>(round(w_stat * as<f64>(scale)));
        f64 total = 0.0;
        for (usize s = 0; s < s_cap; ++s) {
            total += dp[m * s_cap + s];
        }
        f64 cdf_le = 0.0;
        f64 cdf_ge = 0.0;
        for (i64 s = 0; s <= w_scaled && s < as<i64>(s_cap); ++s) {
            cdf_le += dp[m * s_cap + as<usize>(s)];
        }
        for (i64 s = w_scaled; s < as<i64>(s_cap); ++s) {
            cdf_ge += dp[m * s_cap + as<usize>(s)];
        }
        cdf_le /= total;
        cdf_ge /= total;
        switch (tail) {
            case TestTail::kBoth: p_val = min(1.0, 2.0 * min(cdf_le, cdf_ge)); break;
            case TestTail::kRight: p_val = cdf_le; break;
            case TestTail::kLeft: p_val = cdf_ge; break;
        }
    } else {
        const f64 phi = norm_cdf(w_star, 0.0, 1.0);
        switch (tail) {
            case TestTail::kBoth: p_val = 2.0 * min(phi, 1.0 - phi); break;
            case TestTail::kRight: p_val = phi; break;
            case TestTail::kLeft: p_val = 1.0 - phi; break;
        }
    }
    p_val = clamp(p_val, 0.0, 1.0);
    out = RankTestResult{p_val, (p_val < alpha) ? 1 : 0, w_stat, w_star};
    return StatsStatus::kOk;
}

StatsStatus lillietest(ArrayRef<f64> x, f64 alpha, LillieTestResult& out, Allocator* alloc) {
    const usize n = x.size();
    if (n < 4 || alpha <= 0.0 || alpha >= 1.0) {
        return StatsStatus::kInvalidArgument;
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> xs(x, scratch.allocator());
    sort(xs.as_mut_ref());

    f64 mean_v = 0.0;
    f64 var_v = 0.0;
    sample_mean_var(xs.as_ref(), mean_v, var_v);
    const f64 sd = sqrt(var_v);
    if (sd == 0.0) {
        return StatsStatus::kSingularMatrix;
    }

    f64 d_stat = 0.0;
    const f64 dn = as<f64>(n);
    constexpr f64 kInvSqrt2 = 0.7071067811865476;
    for (usize i = 0; i < n; ++i) {
        const f64 z = (xs[i] - mean_v) / sd;
        const f64 f_val = 0.5 * (1.0 + specfun::erf(z * kInvSqrt2));
        const f64 f_lo = as<f64>(i) / dn;
        const f64 f_hi = as<f64>(i + 1) / dn;
        d_stat = max(d_stat, max(abs(f_val - f_lo), abs(f_val - f_hi)));
    }

    const f64 sqn = sqrt(dn);
    const f64 scale = sqn - 0.01 + 0.85 / sqn;
    const f64 d_star = d_stat * scale;

    auto stephens_p = [](f64 ds) noexcept -> f64 {
        if (ds < 0.474) {
            return 0.50;
        }
        struct Pt {
            f64 ds;
            f64 p;
        };
        static constexpr Pt kTbl[8] = {
            {0.474, 0.50}, {0.741, 0.20}, {0.775, 0.15}, {0.819, 0.10},
            {0.895, 0.05}, {1.035, 0.01}, {1.50, 1e-4},  {2.00, 1e-7},
        };
        for (usize k = 1; k < 8; ++k) {
            if (ds <= kTbl[k].ds) {
                const f64 frac = (ds - kTbl[k - 1].ds) / (kTbl[k].ds - kTbl[k - 1].ds);
                const f64 lp = log(kTbl[k - 1].p) + frac * (log(kTbl[k].p) - log(kTbl[k - 1].p));
                return exp(lp);
            }
        }
        return 1e-10;
    };

    auto stephens_cv = [](f64 a) noexcept -> f64 {
        struct Pt {
            f64 a;
            f64 ds;
        };
        static constexpr Pt kTbl[5] = {
            {0.20, 0.741}, {0.15, 0.775}, {0.10, 0.819}, {0.05, 0.895}, {0.01, 1.035},
        };
        for (usize k = 1; k < 5; ++k) {
            if (a >= kTbl[k].a) {
                const f64 frac = (a - kTbl[k - 1].a) / (kTbl[k].a - kTbl[k - 1].a);
                return kTbl[k - 1].ds + frac * (kTbl[k].ds - kTbl[k - 1].ds);
            }
        }
        return kTbl[4].ds;
    };

    const f64 p_val = clamp(stephens_p(d_star), 1e-10, 0.5);
    const f64 d_crit = stephens_cv(alpha) / scale;
    out = LillieTestResult{(d_stat > d_crit) ? 1 : 0, p_val, d_stat, d_crit};
    return StatsStatus::kOk;
}

} // namespace keel::stats
