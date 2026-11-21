#pragma once

#include "keel/array.h"
#include "keel/core.h"
#include "keel/math.h"
#include "keel/memory.h"
#include "keel/ref.h"

namespace keel::stats {

struct Moments {
    f64 mean = 0.0;
    f64 var = 0.0;
};

// ============================================================================
// Continuous Probability Distributions (PDF, CDF, ICDF, Moments)
// ============================================================================

// Normal N(mu, sigma^2), sigma > 0
[[nodiscard]] f64 norm_pdf(f64 x, f64 mu = 0.0, f64 sigma = 1.0) noexcept;
[[nodiscard]] f64 norm_cdf(f64 x, f64 mu = 0.0, f64 sigma = 1.0) noexcept;
[[nodiscard]] f64 norm_inv(f64 p, f64 mu = 0.0, f64 sigma = 1.0) noexcept;
[[nodiscard]] Moments norm_stat(f64 mu = 0.0, f64 sigma = 1.0) noexcept;

// Student's t(nu), nu > 0 (nu = +Inf reduces to standard normal)
[[nodiscard]] f64 t_pdf(f64 x, f64 nu) noexcept;
[[nodiscard]] f64 t_cdf(f64 x, f64 nu) noexcept;
[[nodiscard]] f64 t_inv(f64 p, f64 nu) noexcept;
[[nodiscard]] Moments t_stat(f64 nu) noexcept;

// Chi-squared(k), k >= 0 (k == 0 degenerate at 0)
[[nodiscard]] f64 chi2_pdf(f64 x, f64 k) noexcept;
[[nodiscard]] f64 chi2_cdf(f64 x, f64 k) noexcept;
[[nodiscard]] f64 chi2_inv(f64 p, f64 k) noexcept;
[[nodiscard]] Moments chi2_stat(f64 k) noexcept;

// Fisher F(v1, v2), v1 > 0, v2 > 0
[[nodiscard]] f64 f_pdf(f64 x, f64 v1, f64 v2) noexcept;
[[nodiscard]] f64 f_cdf(f64 x, f64 v1, f64 v2) noexcept;
[[nodiscard]] f64 f_inv(f64 p, f64 v1, f64 v2) noexcept;
[[nodiscard]] Moments f_stat(f64 v1, f64 v2) noexcept;

// Gamma(shape = a, scale = b), a >= 0, b > 0
[[nodiscard]] f64 gam_pdf(f64 x, f64 a, f64 b = 1.0) noexcept;
[[nodiscard]] f64 gam_cdf(f64 x, f64 a, f64 b = 1.0) noexcept;
[[nodiscard]] f64 gam_inv(f64 p, f64 a, f64 b = 1.0) noexcept;
[[nodiscard]] Moments gam_stat(f64 a, f64 b = 1.0) noexcept;

// Beta(a, b), a > 0, b > 0
[[nodiscard]] f64 beta_pdf(f64 x, f64 a, f64 b) noexcept;
[[nodiscard]] f64 beta_cdf(f64 x, f64 a, f64 b) noexcept;
[[nodiscard]] f64 beta_inv(f64 p, f64 a, f64 b) noexcept;
[[nodiscard]] Moments beta_stat(f64 a, f64 b) noexcept;

// Exponential(mean = mu), mu > 0
[[nodiscard]] f64 exp_pdf(f64 x, f64 mu = 1.0) noexcept;
[[nodiscard]] f64 exp_cdf(f64 x, f64 mu = 1.0) noexcept;
[[nodiscard]] f64 exp_inv(f64 p, f64 mu = 1.0) noexcept;
[[nodiscard]] Moments exp_stat(f64 mu = 1.0) noexcept;

// Weibull(scale = a, shape = b), a > 0, b > 0
[[nodiscard]] f64 wbl_pdf(f64 x, f64 a = 1.0, f64 b = 1.0) noexcept;
[[nodiscard]] f64 wbl_cdf(f64 x, f64 a = 1.0, f64 b = 1.0) noexcept;
[[nodiscard]] f64 wbl_inv(f64 p, f64 a = 1.0, f64 b = 1.0) noexcept;
[[nodiscard]] Moments wbl_stat(f64 a = 1.0, f64 b = 1.0) noexcept;

// Lognormal(mu, sigma), sigma > 0
[[nodiscard]] f64 logn_pdf(f64 x, f64 mu = 0.0, f64 sigma = 1.0) noexcept;
[[nodiscard]] f64 logn_cdf(f64 x, f64 mu = 0.0, f64 sigma = 1.0) noexcept;
[[nodiscard]] f64 logn_inv(f64 p, f64 mu = 0.0, f64 sigma = 1.0) noexcept;
[[nodiscard]] Moments logn_stat(f64 mu = 0.0, f64 sigma = 1.0) noexcept;

// Continuous Uniform on [a, b], a < b
[[nodiscard]] f64 unif_pdf(f64 x, f64 a = 0.0, f64 b = 1.0) noexcept;
[[nodiscard]] f64 unif_cdf(f64 x, f64 a = 0.0, f64 b = 1.0) noexcept;
[[nodiscard]] f64 unif_inv(f64 p, f64 a = 0.0, f64 b = 1.0) noexcept;
[[nodiscard]] Moments unif_stat(f64 a = 0.0, f64 b = 1.0) noexcept;

// Rayleigh(scale = b), b > 0
[[nodiscard]] f64 rayl_pdf(f64 x, f64 b = 1.0) noexcept;
[[nodiscard]] f64 rayl_cdf(f64 x, f64 b = 1.0) noexcept;
[[nodiscard]] f64 rayl_inv(f64 p, f64 b = 1.0) noexcept;
[[nodiscard]] Moments rayl_stat(f64 b = 1.0) noexcept;

// Generalized Extreme Value GEV(shape = k, scale = sigma, loc = mu), sigma > 0
[[nodiscard]] f64 gev_pdf(f64 x, f64 k, f64 sigma = 1.0, f64 mu = 0.0) noexcept;
[[nodiscard]] f64 gev_cdf(f64 x, f64 k, f64 sigma = 1.0, f64 mu = 0.0) noexcept;
[[nodiscard]] f64 gev_inv(f64 p, f64 k, f64 sigma = 1.0, f64 mu = 0.0) noexcept;
[[nodiscard]] Moments gev_stat(f64 k, f64 sigma = 1.0, f64 mu = 0.0) noexcept;

// ============================================================================
// Discrete Probability Distributions (PMF, CDF, ICDF, Moments)
// ============================================================================

// Poisson(lambda), lambda >= 0
[[nodiscard]] f64 poiss_pdf(f64 k, f64 lambda) noexcept;
[[nodiscard]] f64 poiss_cdf(f64 k, f64 lambda) noexcept;
[[nodiscard]] f64 poiss_inv(f64 p, f64 lambda) noexcept;
[[nodiscard]] Moments poiss_stat(f64 lambda) noexcept;

// Binomial(n, p), n >= 0 integer, 0 <= p <= 1
[[nodiscard]] f64 bino_pdf(f64 k, f64 n, f64 p) noexcept;
[[nodiscard]] f64 bino_cdf(f64 k, f64 n, f64 p) noexcept;
[[nodiscard]] f64 bino_inv(f64 q, f64 n, f64 p) noexcept;
[[nodiscard]] Moments bino_stat(f64 n, f64 p) noexcept;

// ============================================================================
// Descriptive Statistics & Quantiles
// ============================================================================

[[nodiscard]] f64 mean(ArrayRef<f64> x) noexcept;
[[nodiscard]] f64 var(ArrayRef<f64> x, bool population = false) noexcept;
[[nodiscard]] f64 stddev(ArrayRef<f64> x, bool population = false) noexcept;
[[nodiscard]] f64 skewness(ArrayRef<f64> x, bool bias_corrected = false) noexcept;
[[nodiscard]] f64 kurtosis(ArrayRef<f64> x, bool bias_corrected = false) noexcept;
[[nodiscard]] f64 cov(ArrayRef<f64> x, ArrayRef<f64> y, bool population = false) noexcept;
[[nodiscard]] f64 corrcoef(ArrayRef<f64> x, ArrayRef<f64> y) noexcept;

/// Quantile at probability `p` in [0, 1] using linear interpolation on (i + 0.5) / n.
[[nodiscard]] f64 quantile(ArrayRef<f64> x, f64 p, Allocator* alloc = nullptr);
[[nodiscard]] f64 median(ArrayRef<f64> x, Allocator* alloc = nullptr);
[[nodiscard]] f64 iqr(ArrayRef<f64> x, Allocator* alloc = nullptr);

/// Standardize `x` into `out` as (x - mean) / std.
void zscore(ArrayRef<f64> x, MutArrayRef<f64> out, bool population = false) noexcept;

} // namespace keel::stats
