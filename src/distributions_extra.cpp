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
    return specfun::gammaln(a + 1.0) - specfun::gammaln(b + 1.0) - specfun::gammaln(a - b + 1.0);
}

inline bool hyge_params_ok(f64 m, f64 big_k, f64 n) noexcept {
    return is_finite(m) && is_finite(big_k) && is_finite(n) && m >= 0.0 && big_k >= 0.0 &&
           n >= 0.0 && floor(m) == m && floor(big_k) == big_k && floor(n) == n && big_k <= m &&
           n <= m;
}

// Exponentially scaled modified Bessel function exp(-x) * I_nu(x) for x >= 0.
// Uses direct besseli for moderate x and Hankel's asymptotic series for large x
// to avoid Inf * 0 = NaN overflow in Rician and Noncentral Chi-squared densities.
f64 besseli_scaled(f64 nu, f64 x) noexcept {
    if (x < 0.0 || is_nan(x) || is_nan(nu)) {
        return qnan();
    }
    if (x == 0.0) {
        return (nu == 0.0) ? 1.0 : 0.0;
    }
    if (is_inf(x)) {
        return 0.0;
    }
    if (x < 650.0) {
        const f64 iv = specfun::besseli(nu, x);
        if (is_finite(iv)) {
            return exp(-x) * iv;
        }
    }
    // Hankel asymptotic expansion: exp(-x)*I_nu(x) ~ (1/sqrt(2*pi*x)) * sum (-1)^k a_k(nu) / x^k
    const f64 mu4 = 4.0 * nu * nu;
    f64 sum = 1.0;
    f64 term = 1.0;
    for (i32 k = 1; k <= 20; ++k) {
        const f64 odd = as<f64>(2 * k - 1);
        const f64 next_term = -term * (mu4 - odd * odd) / (8.0 * as<f64>(k) * x);
        if (abs(next_term) > abs(term) && k > 2) {
            break;
        }
        term = next_term;
        sum += term;
        if (abs(term) <= 1e-16 * abs(sum)) {
            break;
        }
    }
    return sum / sqrt(2.0 * kPi * x);
}

} // namespace

// ============================================================================
// Generalized Pareto GP(k, sigma, theta)
// ============================================================================

f64 gp_pdf(f64 x, f64 k, f64 sigma, f64 theta) noexcept {
    if (!(sigma > 0.0) || is_nan(x) || is_nan(k) || is_nan(theta)) {
        return qnan();
    }
    if (is_inf(x)) {
        return 0.0;
    }
    const f64 inv_s = 1.0 / sigma;
    const f64 z = (x - theta) * inv_s;
    if (z < 0.0) {
        return 0.0;
    }
    if (k == 0.0) {
        return inv_s * exp(-z);
    }
    const f64 kz = k * z;
    if (kz <= -1.0) {
        return 0.0;
    }
    return inv_s * exp((-1.0 / k - 1.0) * log1p(kz));
}

f64 gp_cdf(f64 x, f64 k, f64 sigma, f64 theta) noexcept {
    if (!(sigma > 0.0) || is_nan(x) || is_nan(k) || is_nan(theta)) {
        return qnan();
    }
    if (is_inf(x)) {
        return (x > 0.0) ? 1.0 : 0.0;
    }
    const f64 z = (x - theta) / sigma;
    if (z <= 0.0) {
        return 0.0;
    }
    if (k == 0.0) {
        return -expm1(-z);
    }
    const f64 kz = k * z;
    if (kz <= -1.0) {
        return (k > 0.0) ? 0.0 : 1.0;
    }
    return -expm1(-log1p(kz) / k);
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
    const f64 ln1mp = log1p(-p);
    if (k == 0.0) {
        return theta - sigma * ln1mp;
    }
    return theta + sigma * expm1(-k * ln1mp) / k;
}

Moments gp_stat(f64 k, f64 sigma, f64 theta) noexcept {
    if (!(sigma > 0.0) || is_nan(k) || is_nan(theta)) {
        return {qnan(), qnan()};
    }
    if (!(k < 1.0)) {
        return {pinf(), pinf()};
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
    if (is_inf(x)) {
        return 0.0;
    }
    const f64 inv_s = 1.0 / sigma;
    const f64 t = (x - mu) * inv_s;
    if (t > 709.0) {
        return 0.0;
    }
    return inv_s * exp(t - exp(t));
}

f64 ev_cdf(f64 x, f64 mu, f64 sigma) noexcept {
    if (!(sigma > 0.0) || is_nan(x) || is_nan(mu)) {
        return qnan();
    }
    if (is_inf(x)) {
        return (x > 0.0) ? 1.0 : 0.0;
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
    if (x <= 0.0 || is_inf(x)) {
        return 0.0;
    }
    const f64 xs = x / sigma;
    const f64 ss = s / sigma;
    const f64 diff = xs - ss;
    const f64 i0_scaled = besseli_scaled(0.0, xs * ss);
    return (xs / sigma) * exp(-0.5 * diff * diff) * i0_scaled;
}

f64 rice_cdf(f64 x, f64 s, f64 sigma) noexcept {
    if (!(sigma > 0.0) || !(s >= 0.0) || is_nan(x)) {
        return qnan();
    }
    if (x <= 0.0) {
        return 0.0;
    }
    if (is_inf(x)) {
        return 1.0;
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
    const f64 ratio = s / sigma;
    const f64 sg2 = sigma * sigma;
    if (ratio > 1e4) {
        // Asymptotic expansion avoids catastrophic cancellation in 2*sg2 + s^2 - m^2.
        const f64 inv_r2 = 1.0 / (ratio * ratio);
        const f64 m = s * (1.0 + 0.5 * inv_r2 + 0.125 * inv_r2 * inv_r2);
        const f64 v = sg2 * (1.0 - 0.5 * inv_r2);
        return {m, v};
    }
    const f64 half = 0.25 * ratio * ratio;
    const f64 z = -2.0 * half;
    const f64 i0e = besseli_scaled(0.0, half);
    const f64 i1e = besseli_scaled(1.0, half);
    const f64 lag = (1.0 - z) * i0e - z * i1e;
    const f64 m = sigma * sqrt(kPi / 2.0) * lag;
    const f64 v = max(0.0, 2.0 * sg2 + s * s - m * m);
    return {m, v};
}

// ============================================================================
// Noncentral Chi-squared(k, lambda)
// ============================================================================

f64 ncx2_pdf(f64 x, f64 k, f64 lambda) noexcept {
    if (!(k > 0.0) || !(lambda >= 0.0) || is_nan(x)) {
        return qnan();
    }
    if (x < 0.0 || is_inf(x)) {
        return 0.0;
    }
    if (lambda == 0.0) {
        return chi2_pdf(x, k);
    }
    if (x == 0.0) {
        if (k < 2.0) {
            return pinf();
        }
        if (k == 2.0) {
            return 0.5 * exp(-0.5 * lambda);
        }
        return 0.0;
    }
    const f64 nu = 0.5 * (k - 2.0);
    const f64 sq_x = sqrt(x);
    const f64 sq_l = sqrt(lambda);
    const f64 arg = sq_l * sq_x;
    const f64 iv_scaled = besseli_scaled(nu, arg);
    const f64 diff = sq_x - sq_l;
    const f64 log_pref = -0.5 * diff * diff + 0.25 * (k - 2.0) * (log(x) - log(lambda));
    return 0.5 * exp(log_pref) * iv_scaled;
}

f64 ncx2_cdf(f64 x, f64 k, f64 lambda) noexcept {
    if (!(k > 0.0) || !(lambda >= 0.0) || is_nan(x)) {
        return qnan();
    }
    if (x <= 0.0) {
        return 0.0;
    }
    if (is_inf(x)) {
        return 1.0;
    }
    if (lambda == 0.0) {
        return specfun::gammainc(0.5 * x, 0.5 * k);
    }
    const f64 half_l = 0.5 * lambda;
    const f64 half_x = 0.5 * x;

    // Start from the Poisson mode j0 = floor(lambda / 2) and sum both upward and downward
    // so exp(-lambda / 2) never underflows for large lambda.
    const i64 j0 = as<i64>(floor(half_l));
    const f64 dj0 = as<f64>(j0);
    const f64 p0 =
        (j0 == 0) ? exp(-half_l) : exp(dj0 * log(half_l) - half_l - specfun::gammaln(dj0 + 1.0));

    f64 sum = p0 * specfun::gammainc(half_x, 0.5 * k + dj0);

    // Upward summation from j0 + 1.
    f64 pj = p0;
    for (i64 step = 1; step < 50000; ++step) {
        const i64 j = j0 + step;
        pj *= half_l / as<f64>(j);
        if (pj == 0.0) {
            break;
        }
        const f64 contrib = pj * specfun::gammainc(half_x, 0.5 * k + as<f64>(j));
        sum += contrib;
        if (step > 5 && contrib < 1e-16 * (sum + 1e-300)) {
            break;
        }
    }

    // Downward summation from j0 - 1 to 0.
    pj = p0;
    for (i64 j = j0; j > 0; --j) {
        pj *= as<f64>(j) / half_l;
        if (pj == 0.0) {
            break;
        }
        const f64 contrib = pj * specfun::gammainc(half_x, 0.5 * k + as<f64>(j - 1));
        sum += contrib;
        if (j0 - j > 5 && contrib < 1e-16 * (sum + 1e-300)) {
            break;
        }
    }

    return min(1.0, max(0.0, sum));
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
    if (lambda == 0.0) {
        return chi2_inv(p, k);
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
    if (x < 0.0 || is_inf(x)) {
        return 0.0;
    }
    const f64 log_c = log(2.0) + mu * log(mu / omega) - specfun::gammaln(mu);
    if (x == 0.0) {
        if (mu < 0.5) {
            return pinf();
        }
        return (mu == 0.5) ? exp(log_c) : 0.0;
    }
    return exp(log_c + (2.0 * mu - 1.0) * log(x) - (mu / omega) * x * x);
}

f64 naka_cdf(f64 x, f64 mu, f64 omega) noexcept {
    if (!(mu > 0.0) || !(omega > 0.0) || is_nan(x)) {
        return qnan();
    }
    if (x <= 0.0) {
        return 0.0;
    }
    if (is_inf(x)) {
        return 1.0;
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
    if (mu >= 1e6) {
        // Asymptotic expansion: (Gamma(mu+1/2)/Gamma(mu))^2 / mu = 1 - 1/(4*mu) + 1/(32*mu^2) + ...
        const f64 inv_m = 1.0 / mu;
        const f64 r_over_sqrtm = 1.0 - 0.125 * inv_m + (1.0 / 128.0) * inv_m * inv_m;
        const f64 m = sqrt(omega) * r_over_sqrtm;
        const f64 v = omega * (0.25 * inv_m - (1.0 / 32.0) * inv_m * inv_m);
        return {m, v};
    }
    const f64 log_r = specfun::gammaln(mu + 0.5) - specfun::gammaln(mu);
    const f64 r = exp(log_r);
    const f64 m = sqrt(omega / mu) * r;
    const f64 v = omega * max(0.0, 1.0 - exp(2.0 * log_r - log(mu)));
    return {m, v};
}

// ============================================================================
// Geometric(p)
// ============================================================================

f64 geo_pdf(f64 k, f64 p) noexcept {
    if (!(p > 0.0 && p <= 1.0) || is_nan(k)) {
        return qnan();
    }
    if (k < 0.0 || !is_finite(k) || floor(k) != k) {
        return 0.0;
    }
    if (p == 1.0) {
        return (k == 0.0) ? 1.0 : 0.0;
    }
    return exp(k * log1p(-p)) * p;
}

f64 geo_cdf(f64 k, f64 p) noexcept {
    if (!(p > 0.0 && p <= 1.0) || is_nan(k)) {
        return qnan();
    }
    if (k < 0.0) {
        return 0.0;
    }
    if (p == 1.0 || is_inf(k)) {
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
    if (!(n >= 1.0) || !is_finite(n) || floor(n) != n) {
        return qnan();
    }
    if (k < 1.0 || k > n || !is_finite(k) || floor(k) != k) {
        return 0.0;
    }
    return 1.0 / n;
}

f64 unid_cdf(f64 k, f64 n) noexcept {
    if (!(n >= 1.0) || !is_finite(n) || floor(n) != n || is_nan(k)) {
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
    if (!(n >= 1.0) || !is_finite(n) || floor(n) != n || is_nan(p) || p <= 0.0 || p > 1.0) {
        return qnan();
    }
    const f64 r = ceil(p * n);
    if (r < 1.0) {
        return 1.0;
    }
    return (r > n) ? n : r;
}

Moments unid_stat(f64 n) noexcept {
    if (!(n >= 1.0) || !is_finite(n) || floor(n) != n) {
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
    if (k < 0.0 || !is_finite(k) || floor(k) != k) {
        return 0.0;
    }
    if (p == 1.0) {
        return (k == 0.0) ? 1.0 : 0.0;
    }
    const f64 lc = specfun::gammaln(k + r) - specfun::gammaln(k + 1.0) - specfun::gammaln(r);
    return exp(lc + r * log(p) + k * log1p(-p));
}

f64 nbin_cdf(f64 k, f64 r, f64 p) noexcept {
    if (!(r > 0.0 && p > 0.0 && p <= 1.0) || is_nan(k)) {
        return qnan();
    }
    if (k < 0.0) {
        return 0.0;
    }
    if (p == 1.0 || is_inf(k)) {
        return 1.0;
    }
    return specfun::betainc(p, r, floor(k) + 1.0);
}

f64 nbin_inv(f64 q, f64 r, f64 p) noexcept {
    if (!(r > 0.0 && is_finite(r) && p > 0.0 && p <= 1.0) || !(q >= 0.0 && q <= 1.0)) {
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
    const f64 tol = max(1e-13, q * 1e-13);
    const f64 target = q - tol;
    if (nbin_cdf(0.0, r, p) >= target) {
        return 0.0;
    }

    // Bracket [lo, hi] via mean + std estimate, then exact integer binary search using nbin_cdf.
    const f64 mean_v = r * (1.0 - p) / p;
    const f64 sd_v = sqrt(mean_v / p);
    const f64 z = norm_inv(q, 0.0, 1.0);
    f64 guess = floor(mean_v + sd_v * z);
    if (guess < 1.0 || !is_finite(guess)) {
        guess = 1.0;
    }

    f64 lo = 0.0;
    f64 hi = guess;
    if (nbin_cdf(hi, r, p) >= target) {
        f64 step = max(1.0, floor(sd_v));
        lo = max(0.0, hi - step);
        while (lo > 0.0 && nbin_cdf(lo, r, p) >= target) {
            hi = lo;
            step *= 2.0;
            lo = max(0.0, hi - step);
        }
    } else {
        lo = hi;
        f64 step = max(1.0, floor(sd_v));
        hi = lo + step;
        while (nbin_cdf(hi, r, p) < target && hi < 1e15) {
            lo = hi;
            step *= 2.0;
            hi = lo + step;
        }
        if (nbin_cdf(hi, r, p) < target) {
            return hi;
        }
    }

    for (i32 it = 0; it < 64 && hi - lo > 1.0; ++it) {
        const f64 mid = floor(lo + 0.5 * (hi - lo));
        if (mid <= lo || mid >= hi) {
            break;
        }
        if (nbin_cdf(mid, r, p) >= target) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return hi;
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
    if (k < k_min || k > k_max || !is_finite(k) || floor(k) != k) {
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
    const f64 kf = floor(k);
    const f64 mode = floor((n + 1.0) * (big_k + 1.0) / (m + 2.0));
    // Sum from kf downward (when kf <= mode) or complement from kf+1 upward (when kf > mode).
    // Because terms decrease away from the mode, this never suffers k_min underflow!
    if (kf <= mode) {
        f64 f = hyge_pdf(kf, m, big_k, n);
        f64 s = f;
        for (f64 j = kf; j > k_min; j -= 1.0) {
            const f64 num = j * (m - big_k - n + j);
            const f64 den = (big_k - j + 1.0) * (n - j + 1.0);
            f = (den == 0.0) ? 0.0 : (f * num / den);
            s += f;
            if (f <= 1e-17 * s) {
                break;
            }
        }
        return min(1.0, max(0.0, s));
    }
    const f64 k_start = kf + 1.0;
    f64 f = hyge_pdf(k_start, m, big_k, n);
    f64 s = f;
    for (f64 j = k_start; j < k_max; j += 1.0) {
        const f64 num = (big_k - j) * (n - j);
        const f64 den = (j + 1.0) * (m - big_k - n + j + 1.0);
        f = (den == 0.0) ? 0.0 : (f * num / den);
        s += f;
        if (f <= 1e-17 * s) {
            break;
        }
    }
    return min(1.0, max(0.0, 1.0 - s));
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
    const f64 target = q - tol;
    if (hyge_cdf(k_min, m, big_k, n) >= target) {
        return k_min;
    }
    f64 lo = k_min;
    f64 hi = k_max;
    for (i32 it = 0; it < 64 && hi - lo > 1.0; ++it) {
        const f64 mid = floor(lo + 0.5 * (hi - lo));
        if (mid <= lo || mid >= hi) {
            break;
        }
        if (hyge_cdf(mid, m, big_k, n) >= target) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return hi;
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
