#include "keel/stats.h"

#include "keel/specfun.h"

namespace keel::stats {

namespace {

constexpr f64 kSqrt2 = 1.41421356237309504880;
constexpr f64 kSqrt2Pi = 2.50662827463100050241;
constexpr f64 kEulerGamma = 0.5772156649015328606065120900824024;
constexpr f64 kPi2Over6 = 1.6449340668482264364724151666460252;

[[nodiscard]] constexpr f64 qnan() noexcept { return Limits<f64>::quiet_nan(); }
[[nodiscard]] constexpr f64 pinf() noexcept { return Limits<f64>::infinity(); }

} // namespace

// ============================================================================
// 1. Normal
// ============================================================================

f64 norm_pdf(f64 x, f64 mu, f64 sigma) noexcept {
    if (!(sigma > 0.0) || is_nan(x) || is_nan(mu)) return qnan();
    const f64 d = (x - mu) / sigma;
    return exp(-0.5 * d * d) / (sigma * kSqrt2Pi);
}

f64 norm_cdf(f64 x, f64 mu, f64 sigma) noexcept {
    if (!(sigma > 0.0) || is_nan(x) || is_nan(mu)) return qnan();
    return 0.5 * specfun::erfc(-(x - mu) / (sigma * kSqrt2));
}

f64 norm_inv(f64 p, f64 mu, f64 sigma) noexcept {
    if (!(sigma > 0.0) || is_nan(mu) || is_nan(p) || p < 0.0 || p > 1.0) return qnan();
    if (p == 0.0) return -pinf();
    if (p == 1.0) return pinf();
    return mu + sigma * kSqrt2 * specfun::erfinv(2.0 * p - 1.0);
}

Moments norm_stat(f64 mu, f64 sigma) noexcept {
    if (!(sigma > 0.0) || is_nan(mu)) return {qnan(), qnan()};
    return {mu, sigma * sigma};
}

// ============================================================================
// 2. Student's t
// ============================================================================

f64 t_pdf(f64 x, f64 nu) noexcept {
    if (!(nu > 0.0) || is_nan(x)) return qnan();
    if (is_inf(nu)) return norm_pdf(x, 0.0, 1.0);
    const f64 log_norm = specfun::gammaln(0.5 * (nu + 1.0)) - specfun::gammaln(0.5 * nu) - 0.5 * log(nu * kPi);
    const f64 exponent = -0.5 * (nu + 1.0);
    return exp(log_norm + exponent * log1p((x * x) / nu));
}

f64 t_cdf(f64 x, f64 nu) noexcept {
    if (!(nu > 0.0) || is_nan(x)) return qnan();
    if (is_inf(nu)) return norm_cdf(x, 0.0, 1.0);
    if (is_inf(x)) return (x > 0.0) ? 1.0 : 0.0;
    const f64 z = nu / (nu + x * x);
    const f64 iz = specfun::betainc(z, 0.5 * nu, 0.5);
    return (x >= 0.0) ? (1.0 - 0.5 * iz) : (0.5 * iz);
}

f64 t_inv(f64 p, f64 nu) noexcept {
    if (!(nu > 0.0) || is_nan(p) || p < 0.0 || p > 1.0) return qnan();
    if (p == 0.0) return -pinf();
    if (p == 1.0) return pinf();
    if (is_inf(nu)) return norm_inv(p, 0.0, 1.0);
    const f64 q = (p >= 0.5) ? (2.0 * (1.0 - p)) : (2.0 * p);
    const f64 z = specfun::betaincinv(q, 0.5 * nu, 0.5);
    if (z <= 0.0) return (p >= 0.5) ? pinf() : -pinf();
    const f64 mag = sqrt(nu * (1.0 / z - 1.0));
    return (p >= 0.5) ? mag : -mag;
}

Moments t_stat(f64 nu) noexcept {
    if (!(nu > 0.0)) return {qnan(), qnan()};
    const f64 m = (nu > 1.0) ? 0.0 : qnan();
    const f64 v = (nu > 2.0) ? (is_inf(nu) ? 1.0 : (nu / (nu - 2.0))) : qnan();
    return {m, v};
}

// ============================================================================
// 3. Chi-squared
// ============================================================================

f64 chi2_pdf(f64 x, f64 k) noexcept {
    if (is_nan(x) || is_nan(k) || k < 0.0) return qnan();
    if (k == 0.0) return 0.0;
    if (x < 0.0) return 0.0;
    if (x == 0.0) {
        if (k == 2.0) return 0.5;
        return (k > 2.0) ? 0.0 : pinf();
    }
    const f64 half_k = 0.5 * k;
    const f64 log_norm = -half_k * log(2.0) - specfun::gammaln(half_k);
    return exp(log_norm + (half_k - 1.0) * log(x) - 0.5 * x);
}

f64 chi2_cdf(f64 x, f64 k) noexcept {
    if (!(k > 0.0) || is_nan(x)) return qnan();
    if (x <= 0.0) return 0.0;
    return specfun::gammainc(0.5 * x, 0.5 * k);
}

f64 chi2_inv(f64 p, f64 k) noexcept {
    if (is_nan(k) || k < 0.0 || is_nan(p) || p < 0.0 || p > 1.0) return qnan();
    if (k == 0.0) return 0.0;
    return 2.0 * specfun::gammaincinv(p, 0.5 * k);
}

Moments chi2_stat(f64 k) noexcept {
    if (!(k > 0.0)) return {qnan(), qnan()};
    return {k, 2.0 * k};
}

// ============================================================================
// 4. Fisher F
// ============================================================================

f64 f_pdf(f64 x, f64 v1, f64 v2) noexcept {
    if (!(v1 > 0.0) || !(v2 > 0.0) || is_nan(x)) return qnan();
    if (x < 0.0) return 0.0;
    const f64 a = 0.5 * v1;
    const f64 b = 0.5 * v2;
    const f64 lb = specfun::betaln(a, b);
    const f64 lv1 = log(v1);
    const f64 lv2 = log(v2);
    if (x == 0.0) {
        if (v1 < 2.0) return pinf();
        if (v1 == 2.0) return exp(a * lv1 + b * lv2 - (a + b) * lv2 - lb);
        return 0.0;
    }
    const f64 lp = a * lv1 + b * lv2 + (a - 1.0) * log(x) - (a + b) * log(v2 + v1 * x) - lb;
    return exp(lp);
}

f64 f_cdf(f64 x, f64 v1, f64 v2) noexcept {
    if (!(v1 > 0.0) || !(v2 > 0.0) || is_nan(x)) return qnan();
    if (x <= 0.0) return 0.0;
    if (is_inf(x)) return 1.0;
    const f64 z = (v1 * x) / (v1 * x + v2);
    return specfun::betainc(z, 0.5 * v1, 0.5 * v2);
}

f64 f_inv(f64 p, f64 v1, f64 v2) noexcept {
    if (!(v1 > 0.0) || !(v2 > 0.0) || is_nan(p) || p < 0.0 || p > 1.0) return qnan();
    if (p == 0.0) return 0.0;
    if (p == 1.0) return pinf();
    const f64 z = specfun::betaincinv(p, 0.5 * v1, 0.5 * v2);
    if (z <= 0.0) return 0.0;
    if (z >= 1.0) return pinf();
    return (v2 / v1) * z / (1.0 - z);
}

Moments f_stat(f64 v1, f64 v2) noexcept {
    if (!(v1 > 0.0) || !(v2 > 0.0)) return {qnan(), qnan()};
    const f64 m = (v2 > 2.0) ? (v2 / (v2 - 2.0)) : qnan();
    f64 v = qnan();
    if (v2 > 4.0) {
        const f64 num = 2.0 * v2 * v2 * (v1 + v2 - 2.0);
        const f64 den = v1 * (v2 - 2.0) * (v2 - 2.0) * (v2 - 4.0);
        v = num / den;
    }
    return {m, v};
}

// ============================================================================
// 5. Gamma (shape a, scale b)
// ============================================================================

f64 gam_pdf(f64 x, f64 a, f64 b) noexcept {
    if (is_nan(x) || is_nan(a) || !(b > 0.0) || a < 0.0) return qnan();
    if (a == 0.0) return 0.0;
    if (x < 0.0) return 0.0;
    const f64 log_b = log(b);
    const f64 lga = specfun::gammaln(a);
    if (x == 0.0) {
        if (a < 1.0) return pinf();
        if (a > 1.0) return 0.0;
        return exp(-lga - log_b);
    }
    return exp((a - 1.0) * log(x) - x / b - a * log_b - lga);
}

f64 gam_cdf(f64 x, f64 a, f64 b) noexcept {
    if (!(a > 0.0) || !(b > 0.0) || is_nan(x)) return qnan();
    if (x <= 0.0) return 0.0;
    return specfun::gammainc(x / b, a);
}

f64 gam_inv(f64 p, f64 a, f64 b) noexcept {
    if (is_nan(a) || a < 0.0 || !(b > 0.0) || is_nan(p) || p < 0.0 || p > 1.0) return qnan();
    if (a == 0.0) return 0.0;
    return b * specfun::gammaincinv(p, a);
}

Moments gam_stat(f64 a, f64 b) noexcept {
    if (!(a > 0.0) || !(b > 0.0)) return {qnan(), qnan()};
    return {a * b, a * b * b};
}

// ============================================================================
// 6. Beta
// ============================================================================

f64 beta_pdf(f64 x, f64 a, f64 b) noexcept {
    if (!(a > 0.0) || !(b > 0.0) || is_nan(x)) return qnan();
    if (x < 0.0 || x > 1.0) return 0.0;
    const f64 lb = specfun::betaln(a, b);
    if (x == 0.0) {
        if (a == 1.0) return exp(-lb);
        return (a > 1.0) ? 0.0 : pinf();
    }
    if (x == 1.0) {
        if (b == 1.0) return exp(-lb);
        return (b > 1.0) ? 0.0 : pinf();
    }
    return exp((a - 1.0) * log(x) + (b - 1.0) * log1p(-x) - lb);
}

f64 beta_cdf(f64 x, f64 a, f64 b) noexcept {
    if (!(a > 0.0) || !(b > 0.0) || is_nan(x)) return qnan();
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    return specfun::betainc(x, a, b);
}

f64 beta_inv(f64 p, f64 a, f64 b) noexcept {
    if (!(a > 0.0) || !(b > 0.0) || is_nan(p) || p < 0.0 || p > 1.0) return qnan();
    return specfun::betaincinv(p, a, b);
}

Moments beta_stat(f64 a, f64 b) noexcept {
    if (!(a > 0.0) || !(b > 0.0)) return {qnan(), qnan()};
    const f64 s = a + b;
    return {a / s, (a * b) / (s * s * (s + 1.0))};
}

// ============================================================================
// 7. Exponential (mean mu)
// ============================================================================

f64 exp_pdf(f64 x, f64 mu) noexcept {
    if (!(mu > 0.0) || is_nan(x)) return qnan();
    if (x < 0.0) return 0.0;
    const f64 inv_mu = 1.0 / mu;
    return inv_mu * exp(-x * inv_mu);
}

f64 exp_cdf(f64 x, f64 mu) noexcept {
    if (!(mu > 0.0) || is_nan(x)) return qnan();
    if (x <= 0.0) return 0.0;
    return -expm1(-x / mu);
}

f64 exp_inv(f64 p, f64 mu) noexcept {
    if (!(mu > 0.0) || is_nan(p) || p < 0.0 || p > 1.0) return qnan();
    if (p == 0.0) return 0.0;
    if (p >= 1.0) return pinf();
    return -mu * log1p(-p);
}

Moments exp_stat(f64 mu) noexcept {
    if (!(mu > 0.0)) return {qnan(), qnan()};
    return {mu, mu * mu};
}

// ============================================================================
// 8. Weibull (scale a, shape b)
// ============================================================================

f64 wbl_pdf(f64 x, f64 a, f64 b) noexcept {
    if (!(a > 0.0) || !(b > 0.0) || is_nan(x)) return qnan();
    if (x < 0.0) return 0.0;
    if (x == 0.0) {
        if (b < 1.0) return pinf();
        if (b > 1.0) return 0.0;
        return 1.0 / a;
    }
    const f64 r = x / a;
    return (b / a) * pow(r, b - 1.0) * exp(-pow(r, b));
}

f64 wbl_cdf(f64 x, f64 a, f64 b) noexcept {
    if (!(a > 0.0) || !(b > 0.0) || is_nan(x)) return qnan();
    if (x <= 0.0) return 0.0;
    return -expm1(-pow(x / a, b));
}

f64 wbl_inv(f64 p, f64 a, f64 b) noexcept {
    if (!(a > 0.0) || !(b > 0.0) || is_nan(p) || p < 0.0 || p > 1.0) return qnan();
    if (p == 0.0) return 0.0;
    if (p >= 1.0) return pinf();
    return a * pow(-log1p(-p), 1.0 / b);
}

Moments wbl_stat(f64 a, f64 b) noexcept {
    if (!(a > 0.0) || !(b > 0.0)) return {qnan(), qnan()};
    const f64 g1 = specfun::gamma(1.0 + 1.0 / b);
    const f64 g2 = specfun::gamma(1.0 + 2.0 / b);
    return {a * g1, a * a * (g2 - g1 * g1)};
}

// ============================================================================
// 9. Lognormal
// ============================================================================

f64 logn_pdf(f64 x, f64 mu, f64 sigma) noexcept {
    if (!(sigma > 0.0) || is_nan(x) || is_nan(mu)) return qnan();
    if (x <= 0.0) return 0.0;
    const f64 z = (log(x) - mu) / sigma;
    return exp(-0.5 * z * z) / (x * sigma * kSqrt2Pi);
}

f64 logn_cdf(f64 x, f64 mu, f64 sigma) noexcept {
    if (!(sigma > 0.0) || is_nan(x) || is_nan(mu)) return qnan();
    if (x <= 0.0) return 0.0;
    return norm_cdf((log(x) - mu) / sigma, 0.0, 1.0);
}

f64 logn_inv(f64 p, f64 mu, f64 sigma) noexcept {
    if (!(sigma > 0.0) || is_nan(mu) || is_nan(p) || p < 0.0 || p > 1.0) return qnan();
    if (p == 0.0) return 0.0;
    if (p == 1.0) return pinf();
    return exp(mu + sigma * norm_inv(p, 0.0, 1.0));
}

Moments logn_stat(f64 mu, f64 sigma) noexcept {
    if (!(sigma > 0.0) || is_nan(mu)) return {qnan(), qnan()};
    const f64 s2 = sigma * sigma;
    return {exp(mu + 0.5 * s2), expm1(s2) * exp(2.0 * mu + s2)};
}

// ============================================================================
// 10. Continuous Uniform
// ============================================================================

f64 unif_pdf(f64 x, f64 a, f64 b) noexcept {
    if (!(a < b) || is_nan(x)) return qnan();
    return (x >= a && x <= b) ? (1.0 / (b - a)) : 0.0;
}

f64 unif_cdf(f64 x, f64 a, f64 b) noexcept {
    if (!(a < b) || is_nan(x)) return qnan();
    if (x < a) return 0.0;
    if (x >= b) return 1.0;
    return (x - a) / (b - a);
}

f64 unif_inv(f64 p, f64 a, f64 b) noexcept {
    if (!(a < b) || is_nan(p) || p < 0.0 || p > 1.0) return qnan();
    return a + p * (b - a);
}

Moments unif_stat(f64 a, f64 b) noexcept {
    if (!(a < b)) return {qnan(), qnan()};
    const f64 d = b - a;
    return {0.5 * (a + b), (d * d) / 12.0};
}

// ============================================================================
// 11. Rayleigh
// ============================================================================

f64 rayl_pdf(f64 x, f64 b) noexcept {
    if (!(b > 0.0) || is_nan(x)) return qnan();
    if (x < 0.0) return 0.0;
    const f64 inv_b2 = 1.0 / (b * b);
    return x * inv_b2 * exp(-0.5 * x * x * inv_b2);
}

f64 rayl_cdf(f64 x, f64 b) noexcept {
    if (!(b > 0.0) || is_nan(x)) return qnan();
    if (x <= 0.0) return 0.0;
    const f64 inv_b2 = 1.0 / (b * b);
    return -expm1(-0.5 * x * x * inv_b2);
}

f64 rayl_inv(f64 p, f64 b) noexcept {
    if (!(b > 0.0) || is_nan(p) || p < 0.0 || p > 1.0) return qnan();
    if (p == 0.0) return 0.0;
    if (p >= 1.0) return pinf();
    return b * sqrt(-2.0 * log1p(-p));
}

Moments rayl_stat(f64 b) noexcept {
    if (!(b > 0.0)) return {qnan(), qnan()};
    return {b * sqrt(0.5 * kPi), (2.0 - 0.5 * kPi) * b * b};
}

// ============================================================================
// 12. Generalized Extreme Value (GEV)
// ============================================================================

f64 gev_pdf(f64 x, f64 k, f64 sigma, f64 mu) noexcept {
    if (!(sigma > 0.0) || is_nan(x) || is_nan(k) || is_nan(mu)) return qnan();
    const f64 inv_s = 1.0 / sigma;
    const f64 z = (x - mu) * inv_s;
    if (k == 0.0) return inv_s * exp(-z - exp(-z));
    const f64 t = 1.0 + k * z;
    if (t <= 0.0) return 0.0;
    const f64 tinvk = pow(t, -1.0 / k);
    return inv_s * pow(t, -1.0 / k - 1.0) * exp(-tinvk);
}

f64 gev_cdf(f64 x, f64 k, f64 sigma, f64 mu) noexcept {
    if (!(sigma > 0.0) || is_nan(x) || is_nan(k) || is_nan(mu)) return qnan();
    const f64 z = (x - mu) / sigma;
    if (k == 0.0) return exp(-exp(-z));
    const f64 t = 1.0 + k * z;
    if (t <= 0.0) return (k > 0.0) ? 0.0 : 1.0;
    return exp(-pow(t, -1.0 / k));
}

f64 gev_inv(f64 p, f64 k, f64 sigma, f64 mu) noexcept {
    if (!(sigma > 0.0) || is_nan(k) || is_nan(mu) || is_nan(p) || p < 0.0 || p > 1.0) return qnan();
    if (p == 0.0) return (k > 0.0) ? (mu - sigma / k) : -pinf();
    if (p == 1.0) return (k < 0.0) ? (mu - sigma / k) : pinf();
    if (k == 0.0) return mu - sigma * log(-log(p));
    return mu + sigma * (pow(-log(p), -k) - 1.0) / k;
}

Moments gev_stat(f64 k, f64 sigma, f64 mu) noexcept {
    if (!(sigma > 0.0) || is_nan(k) || is_nan(mu)) return {qnan(), qnan()};
    if (k == 0.0) return {mu + sigma * kEulerGamma, sigma * sigma * kPi2Over6};
    if (!(k < 1.0)) return {pinf(), qnan()};
    const f64 g1 = specfun::gamma(1.0 - k);
    const f64 m = mu + sigma * (g1 - 1.0) / k;
    if (!(k < 0.5)) return {m, pinf()};
    const f64 g2 = specfun::gamma(1.0 - 2.0 * k);
    return {m, (sigma * sigma / (k * k)) * (g2 - g1 * g1)};
}

// ============================================================================
// 13. Poisson
// ============================================================================

f64 poiss_pdf(f64 k, f64 lambda) noexcept {
    if (is_nan(lambda) || lambda < 0.0 || is_nan(k)) return qnan();
    if (k < 0.0 || floor(k) != k) return 0.0;
    if (lambda == 0.0) return (k == 0.0) ? 1.0 : 0.0;
    return exp(k * log(lambda) - lambda - specfun::gammaln(k + 1.0));
}

f64 poiss_cdf(f64 k, f64 lambda) noexcept {
    if (is_nan(lambda) || lambda < 0.0 || is_nan(k)) return qnan();
    if (k < 0.0) return 0.0;
    if (lambda == 0.0) return 1.0;
    if (is_inf(k)) return 1.0;
    return 1.0 - specfun::gammainc(lambda, floor(k) + 1.0);
}

f64 poiss_inv(f64 p, f64 lambda) noexcept {
    if (is_nan(lambda) || lambda < 0.0 || is_nan(p) || p < 0.0 || p > 1.0) return qnan();
    if (p == 0.0) return 0.0;
    if (p >= 1.0) return pinf();
    if (lambda == 0.0) return 0.0;

    f64 k = 0.0;
    if (lambda >= 10.0) {
        const f64 z = norm_inv(p, 0.0, 1.0);
        k = floor(lambda + sqrt(lambda) * z);
        if (k < 0.0) k = 0.0;
    }

    const f64 tol = max(1e-13, p * 1e-13);
    f64 cur = poiss_cdf(k, lambda);
    if (cur >= p - tol) {
        while (k > 0.0) {
            const f64 prev = cur - poiss_pdf(k, lambda);
            if (prev < p - tol) return k;
            cur = prev;
            k -= 1.0;
        }
        return 0.0;
    }
    while (cur < p - tol && k < 1e18) {
        k += 1.0;
        cur += poiss_pdf(k, lambda);
        if (cur >= p - tol) return k;
    }
    return k;
}

Moments poiss_stat(f64 lambda) noexcept {
    if (!(lambda > 0.0)) return {qnan(), qnan()};
    return {lambda, lambda};
}

// ============================================================================
// 14. Binomial
// ============================================================================

f64 bino_pdf(f64 k, f64 n, f64 p) noexcept {
    if (is_nan(n) || n < 0.0 || floor(n) != n || is_nan(p) || p < 0.0 || p > 1.0 || is_nan(k)) return qnan();
    if (k < 0.0 || k > n || floor(k) != k) return 0.0;
    if (p == 0.0) return (k == 0.0) ? 1.0 : 0.0;
    if (p == 1.0) return (k == n) ? 1.0 : 0.0;
    const f64 lc = specfun::gammaln(n + 1.0) - specfun::gammaln(k + 1.0) - specfun::gammaln(n - k + 1.0);
    return exp(lc + k * log(p) + (n - k) * log1p(-p));
}

f64 bino_cdf(f64 k, f64 n, f64 p) noexcept {
    if (is_nan(n) || n < 0.0 || floor(n) != n || is_nan(p) || p < 0.0 || p > 1.0 || is_nan(k)) return qnan();
    if (k < 0.0) return 0.0;
    if (k >= n) return 1.0;
    if (p == 0.0) return 1.0;
    if (p == 1.0) return 0.0;
    const f64 kf = floor(k);
    return specfun::betainc(1.0 - p, n - kf, kf + 1.0);
}

f64 bino_inv(f64 q, f64 n, f64 p) noexcept {
    if (is_nan(n) || n < 0.0 || floor(n) != n || is_nan(p) || p < 0.0 || p > 1.0 || is_nan(q) || q < 0.0 || q > 1.0) {
        return qnan();
    }
    if (q == 0.0) return 0.0;
    if (q >= 1.0) return n;
    if (n == 0.0) return 0.0;
    if (p == 0.0) return 0.0;
    if (p == 1.0) return n;

    const f64 r = p / (1.0 - p);
    f64 pmf = pow(1.0 - p, n);
    f64 cdf = pmf;
    const f64 tol = max(1e-13, q * 1e-13);
    if (cdf >= q - tol) return 0.0;
    for (f64 j = 0.0; j < n; j += 1.0) {
        pmf *= ((n - j) / (j + 1.0)) * r;
        cdf += pmf;
        if (cdf >= q - tol) return j + 1.0;
    }
    return n;
}

Moments bino_stat(f64 n, f64 p) noexcept {
    if (is_nan(n) || n < 0.0 || floor(n) != n || is_nan(p) || p < 0.0 || p > 1.0) return {qnan(), qnan()};
    return {n * p, n * p * (1.0 - p)};
}

} // namespace keel::stats
