// keelcpp/stats/src/distributions_extra.cpp
// Pure zero-STL scalar PDF/PMF, CDF, ICDF, and Moments for:
//   GP, EV (Gumbel-min), Rician, Noncentral Chi-squared, Nakagami,
//   Geometric, Discrete Uniform, Negative Binomial, Hypergeometric.

#include "keel/stats.h"

#include "keel/math.h"
#include "keel/specfun.h"

namespace keel::stats {

namespace {

constexpr f64 kEulerConst = 0.5772156649015328606065;
constexpr f64 kPiSqOver6 = 1.6449340668482264364724;

inline f64 qnan() noexcept { return Limits<f64>::quiet_nan(); }
inline f64 pinf() noexcept { return Limits<f64>::infinity(); }

inline f64 log_choose(f64 a, f64 b) noexcept {
    if (b < 0.0 || b > a) {
        return -pinf();
    }
    return lgamma(a + 1.0) - lgamma(b + 1.0) - lgamma(a - b + 1.0);
}

inline bool hyge_params_ok(f64 m, f64 big_k, f64 n) noexcept {
    return m >= 0.0 && big_k >= 0.0 && n >= 0.0 && floor(m) == m && floor(big_k) == big_k &&
           floor(n) == n && big_k <= m && n <= m;
}

} // namespace

// ============================================================================
// Generalized Pareto GP(k, sigma, theta)
// ============================================================================

f64 gp_pdf(f64 x, f64 k, f64 sigma, f64 theta) noexcept {
    if (!(sigma > 0.0) || is_nan(x) || is_nan(k) || is_nan(theta)) {
        return qnan();
    }
    const f64 inv_s = 1.0 / sigma;
    const f64 z = (x - theta) * inv_s;
    if (z < 0.0) {
        return 0.0;
    }
    if (k == 0.0) {
        return inv_s * exp(-z);
    }
    const f64 t = 1.0 + k * z;
    if (t <= 0.0) {
        return 0.0;
    }
    return inv_s * pow(t, -1.0 / k - 1.0);
}

f64 gp_cdf(f64 x, f64 k, f64 sigma, f64 theta) noexcept {
    if (!(sigma > 0.0) || is_nan(x) || is_nan(k) || is_nan(theta)) {
        return qnan();
    }
    const f64 z = (x - theta) / sigma;
    if (z <= 0.0) {
        return 0.0;
    }
    if (k == 0.0) {
        return -expm1(-z);
    }
    const f64 t = 1.0 + k * z;
    if (t <= 0.0) {
        return (k > 0.0) ? 0.0 : 1.0;
    }
    return 1.0 - pow(t, -1.0 / k);
}

f64 gp_inv(f64 p, f64 k, f64 sigma, f64 theta) noexcept {
    if (!(sigma > 0.0) || !(p >= 0.0 && p <= 1.0) || is_nan(k) || is_nan(theta)) {
        return qnan();
    }
    if (p == 0.0) {
        return theta;
    }
    if (p == 1.0) {
        return (k < 0.0) ? (theta - sigma / k) : pinf();
    }
    if (k == 0.0) {
        return theta - sigma * log1p(-p);
    }
    return theta + sigma * (pow(1.0 - p, -k) - 1.0) / k;
}

Moments gp_stat(f64 k, f64 sigma, f64 theta) noexcept {
    if (!(sigma > 0.0) || is_nan(k) || is_nan(theta)) {
        return {qnan(), qnan()};
    }
    if (!(k < 1.0)) {
        return {pinf(), qnan()};
    }
    const f64 m = theta + sigma / (1.0 - k);
    if (!(k < 0.5)) {
        return {m, pinf()};
    }
    const f64 v = sigma * sigma / ((1.0 - k) * (1.0 - k) * (1.0 - 2.0 * k));
    return {m, v};
}

// ============================================================================
// Type-I Extreme Value (Gumbel minimum) EV(mu, sigma)
// ============================================================================

f64 ev_pdf(f64 x, f64 mu, f64 sigma) noexcept {
    if (!(sigma > 0.0) || is_nan(x) || is_nan(mu)) {
        return qnan();
    }
    const f64 inv_s = 1.0 / sigma;
    const f64 t = (x - mu) * inv_s;
    return inv_s * exp(t - exp(t));
}

f64 ev_cdf(f64 x, f64 mu, f64 sigma) noexcept {
    if (!(sigma > 0.0) || is_nan(x) || is_nan(mu)) {
        return qnan();
    }
    const f64 t = (x - mu) / sigma;
    return -expm1(-exp(t));
}

f64 ev_inv(f64 p, f64 mu, f64 sigma) noexcept {
    if (!(sigma > 0.0) || !(p >= 0.0 && p <= 1.0) || is_nan(mu)) {
        return qnan();
    }
    if (p == 0.0) {
        return -pinf();
    }
    if (p == 1.0) {
        return pinf();
    }
    return mu + sigma * log(-log1p(-p));
}

Moments ev_stat(f64 mu, f64 sigma) noexcept {
    if (!(sigma > 0.0) || is_nan(mu)) {
        return {qnan(), qnan()};
    }
    return {mu - sigma * kEulerConst, sigma * sigma * kPiSqOver6};
}

// ============================================================================
// Rician(s, sigma)
// ============================================================================

f64 rice_pdf(f64 x, f64 s, f64 sigma) noexcept {
    if (!(sigma > 0.0) || !(s >= 0.0) || is_nan(x)) {
        return qnan();
    }
    if (x <= 0.0) {
        return 0.0;
    }
    const f64 sg2 = sigma * sigma;
    const f64 arg = x * s / sg2;
    const f64 i0 = specfun::besseli(0.0, arg);
    return (x / sg2) * exp(-(x * x + s * s) / (2.0 * sg2)) * i0;
}

f64 rice_cdf(f64 x, f64 s, f64 sigma) noexcept {
    if (!(sigma > 0.0) || !(s >= 0.0) || is_nan(x)) {
        return qnan();
    }
    if (x <= 0.0) {
        return 0.0;
    }
    return 1.0 - specfun::marcumq(s / sigma, x / sigma, 1);
}

f64 rice_inv(f64 p, f64 s, f64 sigma) noexcept {
    if (!(sigma > 0.0) || !(s >= 0.0) || !(p >= 0.0 && p <= 1.0)) {
        return qnan();
    }
    if (p == 0.0) {
        return 0.0;
    }
    if (p == 1.0) {
        return pinf();
    }
    const f64 a = s / sigma;
    f64 lo = 0.0;
    f64 hi = s + 8.0 * sigma;
    for (i32 iter = 0; iter < 50; ++iter) {
        if (1.0 - specfun::marcumq(a, hi / sigma, 1) >= p) {
            break;
        }
        hi *= 2.0;
    }
    for (i32 iter = 0; iter < 80; ++iter) {
        const f64 mid = 0.5 * (lo + hi);
        if (mid == lo || mid == hi) {
            return mid;
        }
        const f64 fval = 1.0 - specfun::marcumq(a, mid / sigma, 1);
        if (fval < p) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return 0.5 * (lo + hi);
}

Moments rice_stat(f64 s, f64 sigma) noexcept {
    if (!(sigma > 0.0) || !(s >= 0.0)) {
        return {qnan(), qnan()};
    }
    if (s == 0.0) {
        return {sigma * sqrt(kPi / 2.0), sigma * sigma * (2.0 - kPi / 2.0)};
    }
    const f64 s2 = s * s;
    const f64 sg2 = sigma * sigma;
    const f64 z = -s2 / (2.0 * sg2);
    const f64 half = s2 / (4.0 * sg2);
    const f64 i0 = specfun::besseli(0.0, half);
    const f64 i1 = specfun::besseli(1.0, half);
    const f64 lag = exp(z / 2.0) * ((1.0 - z) * i0 - z * i1);
    const f64 m = sigma * sqrt(kPi / 2.0) * lag;
    const f64 v = 2.0 * sg2 + s2 - m * m;
    return {m, v};
}

// ============================================================================
// Noncentral Chi-squared(k, lambda)
// ============================================================================

f64 ncx2_pdf(f64 x, f64 k, f64 lambda) noexcept {
    if (!(k > 0.0) || !(lambda >= 0.0) || is_nan(x)) {
        return qnan();
    }
    if (lambda == 0.0) {
        if (x <= 0.0) {
            return (k == 2.0 && x == 0.0) ? 0.5 : 0.0;
        }
        const f64 half_k = 0.5 * k;
        return exp(-half_k * log(2.0) - lgamma(half_k) + (half_k - 1.0) * log(x) - 0.5 * x);
    }
    if (x <= 0.0) {
        return 0.0;
    }
    const f64 nu = 0.5 * (k - 2.0);
    const f64 iv = specfun::besseli(nu, sqrt(lambda * x));
    return 0.5 * exp(-0.5 * (x + lambda)) * pow(x / lambda, 0.25 * (k - 2.0)) * iv;
}

f64 ncx2_cdf(f64 x, f64 k, f64 lambda) noexcept {
    if (!(k > 0.0) || !(lambda >= 0.0) || is_nan(x)) {
        return qnan();
    }
    if (x <= 0.0) {
        return 0.0;
    }
    if (lambda == 0.0) {
        return specfun::gammainc(0.5 * x, 0.5 * k);
    }
    const f64 half_l = 0.5 * lambda;
    const f64 half_x = 0.5 * x;
    f64 pj = exp(-half_l);
    f64 sum = 0.0;
    for (i32 j = 0; j < 2000; ++j) {
        const f64 cdf_term = specfun::gammainc(half_x, 0.5 * k + as<f64>(j));
        const f64 contrib = pj * cdf_term;
        sum += contrib;
        if (j > 5 && contrib < 1e-16 * (sum + 1e-300)) {
            break;
        }
        pj *= half_l / as<f64>(j + 1);
    }
    return min(1.0, sum);
}

f64 ncx2_inv(f64 p, f64 k, f64 lambda) noexcept {
    if (!(k > 0.0) || !(lambda >= 0.0) || !(p >= 0.0 && p <= 1.0)) {
        return qnan();
    }
    if (p == 0.0) {
        return 0.0;
    }
    if (p == 1.0) {
        return pinf();
    }
    const f64 m = k + lambda;
    const f64 v = 2.0 * (k + 2.0 * lambda);
    f64 lo = 0.0;
    f64 hi = m + 10.0 * sqrt(v) + 1.0;
    for (i32 iter = 0; iter < 50; ++iter) {
        if (ncx2_cdf(hi, k, lambda) >= p) {
            break;
        }
        hi *= 2.0;
    }
    for (i32 iter = 0; iter < 80; ++iter) {
        const f64 mid = 0.5 * (lo + hi);
        if (mid == lo || mid == hi) {
            return mid;
        }
        if (ncx2_cdf(mid, k, lambda) < p) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return 0.5 * (lo + hi);
}

Moments ncx2_stat(f64 k, f64 lambda) noexcept {
    if (!(k > 0.0) || !(lambda >= 0.0)) {
        return {qnan(), qnan()};
    }
    return {k + lambda, 2.0 * (k + 2.0 * lambda)};
}

// ============================================================================
// Nakagami(mu, omega)
// ============================================================================

f64 naka_pdf(f64 x, f64 mu, f64 omega) noexcept {
    if (!(mu > 0.0) || !(omega > 0.0) || is_nan(x)) {
        return qnan();
    }
    if (x < 0.0) {
        return 0.0;
    }
    const f64 c = 2.0 * pow(mu, mu) / (tgamma(mu) * pow(omega, mu));
    if (x == 0.0) {
        if (mu < 0.5) {
            return pinf();
        }
        return (mu == 0.5) ? c : 0.0;
    }
    return c * pow(x, 2.0 * mu - 1.0) * exp(-(mu / omega) * x * x);
}

f64 naka_cdf(f64 x, f64 mu, f64 omega) noexcept {
    if (!(mu > 0.0) || !(omega > 0.0) || is_nan(x)) {
        return qnan();
    }
    if (x <= 0.0) {
        return 0.0;
    }
    return specfun::gammainc((mu / omega) * x * x, mu);
}

f64 naka_inv(f64 p, f64 mu, f64 omega) noexcept {
    if (!(mu > 0.0) || !(omega > 0.0) || !(p >= 0.0 && p <= 1.0)) {
        return qnan();
    }
    if (p == 0.0) {
        return 0.0;
    }
    if (p == 1.0) {
        return pinf();
    }
    const f64 q = specfun::gammaincinv(p, mu);
    return sqrt((omega / mu) * q);
}

Moments naka_stat(f64 mu, f64 omega) noexcept {
    if (!(mu > 0.0) || !(omega > 0.0)) {
        return {qnan(), qnan()};
    }
    const f64 r = tgamma(mu + 0.5) / tgamma(mu);
    const f64 m = sqrt(omega / mu) * r;
    const f64 v = omega * (1.0 - (r * r) / mu);
    return {m, v};
}

// ============================================================================
// Geometric(p)
// ============================================================================

f64 geo_pdf(f64 k, f64 p) noexcept {
    if (!(p > 0.0 && p <= 1.0) || is_nan(k)) {
        return qnan();
    }
    if (k < 0.0 || floor(k) != k) {
        return 0.0;
    }
    if (p == 1.0) {
        return (k == 0.0) ? 1.0 : 0.0;
    }
    return pow(1.0 - p, k) * p;
}

f64 geo_cdf(f64 k, f64 p) noexcept {
    if (!(p > 0.0 && p <= 1.0) || is_nan(k)) {
        return qnan();
    }
    if (k < 0.0) {
        return 0.0;
    }
    if (p == 1.0) {
        return 1.0;
    }
    return -expm1((floor(k) + 1.0) * log1p(-p));
}

f64 geo_inv(f64 q, f64 p) noexcept {
    if (!(p > 0.0 && p <= 1.0) || !(q >= 0.0 && q <= 1.0)) {
        return qnan();
    }
    if (q == 0.0) {
        return 0.0;
    }
    if (q == 1.0) {
        return pinf();
    }
    if (p == 1.0) {
        return 0.0;
    }
    const f64 lp = log1p(-p);
    const f64 v = log1p(-q) / lp - 1.0;
    f64 k = ceil(v);
    if (k < 0.0) {
        k = 0.0;
    }
    if (k > 0.0) {
        const f64 cdf_prev = -expm1(k * lp);
        const f64 tol = max(1e-13, q * 1e-13);
        if (cdf_prev >= q - tol) {
            k -= 1.0;
        }
    }
    return k;
}

Moments geo_stat(f64 p) noexcept {
    if (!(p > 0.0 && p <= 1.0)) {
        return {qnan(), qnan()};
    }
    const f64 q = 1.0 - p;
    return {q / p, q / (p * p)};
}

// ============================================================================
// Discrete Uniform(n) on {1, ..., n}
// ============================================================================

f64 unid_pdf(f64 k, f64 n) noexcept {
    if (!(n >= 1.0) || floor(n) != n || is_nan(k)) {
        return qnan();
    }
    if (k < 1.0 || k > n || floor(k) != k) {
        return 0.0;
    }
    return 1.0 / n;
}

f64 unid_cdf(f64 k, f64 n) noexcept {
    if (!(n >= 1.0) || floor(n) != n || is_nan(k)) {
        return qnan();
    }
    if (k < 1.0) {
        return 0.0;
    }
    if (k >= n) {
        return 1.0;
    }
    return floor(k) / n;
}

f64 unid_inv(f64 p, f64 n) noexcept {
    if (!(n >= 1.0) || floor(n) != n || is_nan(p) || p <= 0.0 || p > 1.0) {
        return qnan();
    }
    const f64 r = ceil(p * n);
    if (r < 1.0) {
        return 1.0;
    }
    return (r > n) ? n : r;
}

Moments unid_stat(f64 n) noexcept {
    if (!(n >= 1.0) || floor(n) != n) {
        return {qnan(), qnan()};
    }
    return {0.5 * (n + 1.0), (n * n - 1.0) / 12.0};
}

// ============================================================================
// Negative Binomial(r, p)
// ============================================================================

f64 nbin_pdf(f64 k, f64 r, f64 p) noexcept {
    if (!(r > 0.0 && p > 0.0 && p <= 1.0) || is_nan(k)) {
        return qnan();
    }
    if (k < 0.0 || floor(k) != k) {
        return 0.0;
    }
    if (p == 1.0) {
        return (k == 0.0) ? 1.0 : 0.0;
    }
    const f64 lc = lgamma(k + r) - lgamma(k + 1.0) - lgamma(r);
    return exp(lc + r * log(p) + k * log1p(-p));
}

f64 nbin_cdf(f64 k, f64 r, f64 p) noexcept {
    if (!(r > 0.0 && p > 0.0 && p <= 1.0) || is_nan(k)) {
        return qnan();
    }
    if (k < 0.0) {
        return 0.0;
    }
    if (p == 1.0) {
        return 1.0;
    }
    return specfun::betainc(p, r, floor(k) + 1.0);
}

f64 nbin_inv(f64 q, f64 r, f64 p) noexcept {
    if (!(r > 0.0 && p > 0.0 && p <= 1.0) || !(q >= 0.0 && q <= 1.0)) {
        return qnan();
    }
    if (q == 0.0) {
        return 0.0;
    }
    if (q == 1.0) {
        return pinf();
    }
    if (p == 1.0) {
        return 0.0;
    }
    const f64 q1m = 1.0 - p;
    f64 pmf = pow(p, r);
    f64 cdf = pmf;
    const f64 tol = max(1e-13, q * 1e-13);
    if (cdf >= q - tol) {
        return 0.0;
    }
    for (f64 k = 0.0; k < 1e12; k += 1.0) {
        pmf *= (k + r) / (k + 1.0) * q1m;
        cdf += pmf;
        if (cdf >= q - tol) {
            return k + 1.0;
        }
    }
    return pinf();
}

Moments nbin_stat(f64 r, f64 p) noexcept {
    if (!(r > 0.0 && p > 0.0 && p <= 1.0)) {
        return {qnan(), qnan()};
    }
    const f64 m = r * (1.0 - p) / p;
    return {m, m / p};
}

// ============================================================================
// Hypergeometric(m, big_k, n)
// ============================================================================

f64 hyge_pdf(f64 k, f64 m, f64 big_k, f64 n) noexcept {
    if (!hyge_params_ok(m, big_k, n) || is_nan(k)) {
        return qnan();
    }
    const f64 k_min = max(0.0, n - (m - big_k));
    const f64 k_max = min(n, big_k);
    if (k < k_min || k > k_max || floor(k) != k) {
        return 0.0;
    }
    return exp(log_choose(big_k, k) + log_choose(m - big_k, n - k) - log_choose(m, n));
}

f64 hyge_cdf(f64 k, f64 m, f64 big_k, f64 n) noexcept {
    if (!hyge_params_ok(m, big_k, n) || is_nan(k)) {
        return qnan();
    }
    if (k < 0.0) {
        return 0.0;
    }
    const f64 k_max = min(n, big_k);
    if (k >= k_max) {
        return 1.0;
    }
    const f64 k_min = max(0.0, n - (m - big_k));
    if (k < k_min) {
        return 0.0;
    }
    f64 f = hyge_pdf(k_min, m, big_k, n);
    f64 s = f;
    const f64 kf = floor(k);
    for (f64 j = k_min; j < kf; j += 1.0) {
        const f64 num = (big_k - j) * (n - j);
        const f64 den = (j + 1.0) * (m - big_k - n + j + 1.0);
        f = (den == 0.0) ? 0.0 : (f * num / den);
        s += f;
    }
    return min(1.0, max(0.0, s));
}

f64 hyge_inv(f64 q, f64 m, f64 big_k, f64 n) noexcept {
    if (!hyge_params_ok(m, big_k, n) || !(q >= 0.0 && q <= 1.0)) {
        return qnan();
    }
    const f64 k_min = max(0.0, n - (m - big_k));
    const f64 k_max = min(n, big_k);
    if (q == 0.0) {
        return k_min;
    }
    if (q >= 1.0) {
        return k_max;
    }
    const f64 tol = max(1e-13, q * 1e-13);
    f64 f = hyge_pdf(k_min, m, big_k, n);
    f64 s = f;
    if (s >= q - tol) {
        return k_min;
    }
    for (f64 j = k_min; j < k_max; j += 1.0) {
        const f64 num = (big_k - j) * (n - j);
        const f64 den = (j + 1.0) * (m - big_k - n + j + 1.0);
        f = (den == 0.0) ? 0.0 : (f * num / den);
        s += f;
        if (s >= q - tol) {
            return j + 1.0;
        }
    }
    return k_max;
}

Moments hyge_stat(f64 m, f64 big_k, f64 n) noexcept {
    if (!hyge_params_ok(m, big_k, n) || m < 1.0) {
        return {qnan(), qnan()};
    }
    const f64 mean_v = n * big_k / m;
    if (m < 2.0) {
        return {mean_v, qnan()};
    }
    const f64 var_v = n * big_k * (m - big_k) * (m - n) / (m * m * (m - 1.0));
    return {mean_v, var_v};
}

} // namespace keel::stats
