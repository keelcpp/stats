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

// ============================================================================
// Pairwise Distances & Mahalanobis
// ============================================================================

enum class StatsStatus : u8 {
    kOk = 0,
    kInvalidArgument,
    kSingularMatrix,
};

enum class DistanceMetric : u8 {
    kEuclidean = 0,
    kSqEuclidean,
    kCityblock,
    kChebychev,
    kMinkowski,
    kCosine,
    kCorrelation,
    kHamming,
    kJaccard,
    kMahalanobis,
    kSeuclidean,
    kSpearman,
};

// Pairwise condensed distance vector of `rows x cols` column-major `x`.
// `out` has length `rows * (rows - 1) / 2`.
// For `kMahalanobis`, optional `extra_param` is `cols x cols` column-major covariance.
// For `kSeuclidean`, optional `extra_param` is length-`cols` scale vector.
[[nodiscard]] StatsStatus pdist(ArrayRef<f64> x_col_major, usize rows, usize cols,
                                DistanceMetric metric, f64 minkowski_p, ArrayRef<f64> extra_param,
                                MutArrayRef<f64> out, Allocator* alloc = nullptr);

// Cross-distance matrix (`mx x my` column-major) between `x` (`mx x cols`) and `y` (`my x cols`).
[[nodiscard]] StatsStatus pdist2(ArrayRef<f64> x_col_major, usize mx, ArrayRef<f64> y_col_major,
                                 usize my, usize cols, DistanceMetric metric, f64 minkowski_p,
                                 ArrayRef<f64> extra_param, MutArrayRef<f64> out_col_major,
                                 Allocator* alloc = nullptr);

// Condensed <-> square symmetric distance matrix conversions.
[[nodiscard]] StatsStatus squareform_to_matrix(ArrayRef<f64> condensed, usize n,
                                               MutArrayRef<f64> out_col_major) noexcept;
[[nodiscard]] StatsStatus squareform_to_condensed(ArrayRef<f64> mat_col_major, usize n,
                                                  MutArrayRef<f64> out_condensed) noexcept;

// Rank-safe QR-based squared Mahalanobis distance of each row of `y` (`my x cols`)
// relative to reference sample `x` (`mx x cols`, `mx >= 2`).
[[nodiscard]] StatsStatus mahal_qr(ArrayRef<f64> y_col_major, usize my, ArrayRef<f64> x_col_major,
                                   usize mx, usize cols, MutArrayRef<f64> out_d2,
                                   Allocator* alloc = nullptr);

// ============================================================================
// Hierarchical Clustering (Linkage, Cophenet, Inconsistent, Cluster)
// ============================================================================

enum class LinkageMethod : u8 {
    kSingle = 0,
    kComplete,
    kAverage,
    kWeighted,
    kCentroid,
    kMedian,
    kWard,
};

// Agglomerative linkage on condensed `pdist` vector of length `n*(n-1)/2`.
// Writes `(n - 1) x 3` column-major tree `[id_a, id_b, dist]` (1-based node IDs).
[[nodiscard]] StatsStatus linkage(ArrayRef<f64> pdist_condensed, usize n, LinkageMethod method,
                                  MutArrayRef<f64> out_z_col_major, Allocator* alloc = nullptr);

// Inconsistency coefficients (`num_merges x 4` column-major: mean, std, count, coeff).
[[nodiscard]] StatsStatus inconsistent(ArrayRef<f64> z_col_major, usize num_merges, i32 depth,
                                       MutArrayRef<f64> out_col_major, Allocator* alloc = nullptr);

// Flatten linkage tree into `max_clust` clusters (writes 1-based labels of length `num_merges + 1`).
[[nodiscard]] StatsStatus cluster_maxclust(ArrayRef<f64> z_col_major, usize num_merges,
                                           i32 max_clust, MutArrayRef<i32> out_labels,
                                           Allocator* alloc = nullptr);

// Flatten linkage tree by distance (`by_distance == true`) or inconsistency cutoff.
[[nodiscard]] StatsStatus cluster_cutoff(ArrayRef<f64> z_col_major, usize num_merges, f64 cutoff,
                                         bool by_distance, i32 depth, MutArrayRef<i32> out_labels,
                                         Allocator* alloc = nullptr);

// Cophenetic correlation coefficient & condensed cophenetic distances (`out_dcoph` may be empty).
[[nodiscard]] StatsStatus cophenet(ArrayRef<f64> z_col_major, usize num_merges,
                                   ArrayRef<f64> pdist_condensed, MutArrayRef<f64> out_dcoph,
                                   f64& out_corr, Allocator* alloc = nullptr);

// ============================================================================
// Regularized Regression (LASSO / Elastic-Net & GLM-LASSO)
// ============================================================================

enum class GlmFamily : u8 {
    kNormal = 0,
    kBinomial,
    kPoisson,
};

// Cyclic coordinate-descent LASSO / elastic-net (`alpha in [0, 1]`).
// `out_b_col_major` is `p x lambdas.size()`; `out_intercept` has length `lambdas.size()`.
[[nodiscard]] StatsStatus lasso(ArrayRef<f64> x_col_major, usize n, usize p, ArrayRef<f64> y,
                                ArrayRef<f64> lambdas, f64 alpha, MutArrayRef<f64> out_b_col_major,
                                MutArrayRef<f64> out_intercept, Allocator* alloc = nullptr);

// IRLS + coordinate-descent regularized GLM (Normal / Binomial / Poisson).
[[nodiscard]] StatsStatus lassoglm(ArrayRef<f64> x_col_major, usize n, usize p, ArrayRef<f64> y,
                                   GlmFamily family, ArrayRef<f64> lambdas, f64 alpha,
                                   MutArrayRef<f64> out_b_col_major, MutArrayRef<f64> out_intercept,
                                   Allocator* alloc = nullptr);

// ============================================================================
// Analysis of Variance & Nonparametric Rank Tests
// ============================================================================

struct Anova1Result {
    f64 p = 0.0;
    f64 f = 0.0;
    f64 df_between = 0.0;
    f64 df_within = 0.0;
    f64 ss_between = 0.0;
    f64 ss_within = 0.0;
};

struct Anova2Result {
    f64 p_cols = 0.0;
    f64 p_rows = 0.0;
    f64 f_cols = 0.0;
    f64 f_rows = 0.0;
    f64 df_cols = 0.0;
    f64 df_rows = 0.0;
    f64 df_err = 0.0;
    f64 ss_cols = 0.0;
    f64 ss_rows = 0.0;
    f64 ss_err = 0.0;
};

struct KruskalResult {
    f64 p = 0.0;
    f64 h = 0.0;
    f64 df = 0.0;
    f64 sum_r2_n = 0.0;
};

struct FriedmanResult {
    f64 p = 0.0;
    f64 q = 0.0;
    f64 df = 0.0;
};

// One-way ANOVA on concatenated group values (`group_offsets` has length K + 1).
[[nodiscard]] StatsStatus anova1_grouped(ArrayRef<f64> values, ArrayRef<usize> group_offsets,
                                         Anova1Result& out) noexcept;

// Two-way ANOVA without replication on `rows x cols` column-major matrix `y`.
[[nodiscard]] StatsStatus anova2(ArrayRef<f64> y_col_major, usize rows, usize cols,
                                 Anova2Result& out, Allocator* alloc = nullptr);

// Kruskal-Wallis nonparametric one-way rank test on concatenated group values.
[[nodiscard]] StatsStatus kruskal_wallis(ArrayRef<f64> values, ArrayRef<usize> group_offsets,
                                         KruskalResult& out, Allocator* alloc = nullptr);

// Friedman nonparametric two-way block rank test (`reps >= 1` replicates per cell).
[[nodiscard]] StatsStatus friedman(ArrayRef<f64> x_col_major, usize rows, usize cols, i32 reps,
                                   FriedmanResult& out, Allocator* alloc = nullptr);

} // namespace keel::stats
