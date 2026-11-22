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

// ============================================================================
// Linear, Weighted, Ridge & Generalized Linear Regression
// ============================================================================

struct RegressStats {
    f64 r2 = 0.0;
    f64 f = 0.0;
    f64 p_f = 0.0;
    f64 sigma2 = 0.0;
};

// OLS regression `y ~ X * beta` (`x_col_major` is `n x p`, `y` has length `n`).
// `out_beta` (p), `out_bint` (p x 2 col-major), `out_r` (n), `out_rint` (n x 2 col-major).
[[nodiscard]] StatsStatus regress(ArrayRef<f64> y, ArrayRef<f64> x_col_major, usize n, usize p,
                                  f64 alpha, MutArrayRef<f64> out_beta,
                                  MutArrayRef<f64> out_bint_col_major, MutArrayRef<f64> out_r,
                                  MutArrayRef<f64> out_rint_col_major, RegressStats& out_stats,
                                  Allocator* alloc = nullptr);

// Weighted or generalized least squares `A * x ~ b`.
// Pass empty `weights` for OLS, length-`n` diagonal weights (`is_full_cov == false`),
// or `n x n` column-major SPD covariance (`is_full_cov == true`).
[[nodiscard]] StatsStatus lscov(ArrayRef<f64> a_col_major, ArrayRef<f64> b, usize n, usize p,
                                ArrayRef<f64> weights, bool is_full_cov, MutArrayRef<f64> out_x,
                                MutArrayRef<f64> out_stdx, f64& out_mse,
                                MutArrayRef<f64> out_s_col_major, Allocator* alloc = nullptr);

// Ridge regression (`out_b_col_major` is `(scaled ? p : p + 1) x k_vals.size()`).
[[nodiscard]] StatsStatus ridge(ArrayRef<f64> y, ArrayRef<f64> x_col_major, usize n, usize p,
                                ArrayRef<f64> k_vals, bool scaled,
                                MutArrayRef<f64> out_b_col_major, Allocator* alloc = nullptr);

enum class GlmDist : u8 {
    kNormal = 0,
    kBinomial,
    kPoisson,
    kGamma,
    kInverseGaussian,
};

enum class GlmLink : u8 {
    kIdentity = 0,
    kLogit,
    kLog,
    kReciprocal,
    kProbit,
};

struct GlmFitOut {
    Array<f64> beta;      // length p = p_in + 1 (intercept first)
    f64 deviance = 0.0;
    f64 dfe = 0.0;
    f64 sfit = 0.0;
    f64 s = 1.0;
    bool estdisp = false;
    Array<f64> covb;      // p x p col-major
    Array<f64> se;        // p
    Array<f64> coeffcorr; // p x p col-major
    Array<f64> t_stat;    // p
    Array<f64> resid;     // n
    Array<f64> wts;       // n

    explicit GlmFitOut(Allocator* alloc = nullptr)
        : beta(alloc), covb(alloc), se(alloc), coeffcorr(alloc), t_stat(alloc), resid(alloc),
          wts(alloc) {}
};

[[nodiscard]] StatsStatus glmfit(ArrayRef<f64> x_col_major, ArrayRef<f64> y, usize n, usize p_in,
                                 GlmDist distr, GlmLink link, GlmFitOut& out,
                                 Allocator* alloc = nullptr);

[[nodiscard]] StatsStatus glmval(ArrayRef<f64> beta, ArrayRef<f64> x_col_major, usize m, usize p_in,
                                 GlmLink link, MutArrayRef<f64> out_y) noexcept;

// ============================================================================
// PCA, Linear/Quadratic Discriminant Analysis & Nearest Correlation Matrix
// ============================================================================

// PCA on `n x d` column-major `x`.
// Outputs: `coeff` (`d x d`), `score` (`n x d`), `latent` (`d`), `tsq` (`n`),
// `explained` (`d`), `mu` (`d`). Any output slice except `coeff`/`latent` may be empty.
[[nodiscard]] StatsStatus pca(ArrayRef<f64> x_col_major, usize n, usize d,
                              MutArrayRef<f64> out_coeff_col_major,
                              MutArrayRef<f64> out_score_col_major, MutArrayRef<f64> out_latent,
                              MutArrayRef<f64> out_tsq, MutArrayRef<f64> out_explained,
                              MutArrayRef<f64> out_mu, Allocator* alloc = nullptr);

// PCA directly from a `d x d` column-major covariance matrix `c`.
[[nodiscard]] StatsStatus pcacov(ArrayRef<f64> c_col_major, usize d,
                                 MutArrayRef<f64> out_coeff_col_major, MutArrayRef<f64> out_latent,
                                 MutArrayRef<f64> out_explained, Allocator* alloc = nullptr);

// PCA reconstruction and residuals keeping `ndim` principal components.
[[nodiscard]] StatsStatus pcares(ArrayRef<f64> x_col_major, usize n, usize d, usize ndim,
                                 MutArrayRef<f64> out_res_col_major,
                                 MutArrayRef<f64> out_recon_col_major, Allocator* alloc = nullptr);

enum class DiscrimType : u8 {
    kLinear = 0,
    kDiagLinear,
    kQuadratic,
    kDiagQuadratic,
};

struct LdaResult {
    Array<f64> pred;      // n_samp
    f64 train_error = 0.0;
    Array<f64> posterior; // n_samp x K col-major
    Array<f64> logp;      // n_samp
    usize num_classes = 0;

    explicit LdaResult(Allocator* alloc = nullptr)
        : pred(alloc), posterior(alloc), logp(alloc) {}
};

[[nodiscard]] StatsStatus lda_classify(ArrayRef<f64> sample_col_major, usize n_samp,
                                       ArrayRef<f64> training_col_major, usize n_tr, usize d,
                                       ArrayRef<f64> group_labels, DiscrimType type,
                                       LdaResult& out, Allocator* alloc = nullptr);

// Higham (2002) alternating projections with Dykstra correction for nearest correlation matrix.
// Pass empty `w_scale_col_major` for unweighted, or `n x n` sqrt-weight matrix.
[[nodiscard]] StatsStatus nearcorr(ArrayRef<f64> a_col_major, usize n,
                                   ArrayRef<f64> w_scale_col_major, f64 tol, i64 max_iter,
                                   MutArrayRef<f64> out_col_major, Allocator* alloc = nullptr);

// ============================================================================
// Clustering Extras (k-medoids PAM, DBSCAN, Silhouette)
// ============================================================================

// Single replicate of PAM k-medoids given initial 0-based medoid indices `init_medoids` (length k).
// Writes 0-based `out_idx` (n), 0-based `out_medoids` (k), `out_sumd` (k), `out_total`, `out_iters`.
[[nodiscard]] StatsStatus kmedoids_pam_step(ArrayRef<f64> x_col_major, usize n, usize d,
                                            ArrayRef<i32> init_medoids, i32 max_iter,
                                            DistanceMetric metric, f64 minkowski_p,
                                            MutArrayRef<i32> out_idx, MutArrayRef<i32> out_medoids,
                                            MutArrayRef<f64> out_sumd, f64& out_total,
                                            i32& out_iters, Allocator* alloc = nullptr);

// DBSCAN density clustering. If `precomputed == true`, `x_col_major` is `n x n` distance matrix.
// Writes `out_labels` (`-1` for noise, `1..C` for clusters) and `out_core` (`0`/`1`).
[[nodiscard]] StatsStatus dbscan(ArrayRef<f64> x_col_major, usize n, usize d, f64 eps, i32 minpts,
                                 DistanceMetric metric, f64 minkowski_p, bool precomputed,
                                 MutArrayRef<i32> out_labels, MutArrayRef<u8> out_core,
                                 Allocator* alloc = nullptr);

// Silhouette score per point on `n x d` column-major `x` with integer cluster labels.
[[nodiscard]] StatsStatus silhouette(ArrayRef<f64> x_col_major, usize n, usize d,
                                     ArrayRef<i32> labels, DistanceMetric metric, f64 minkowski_p,
                                     MutArrayRef<f64> out_s, Allocator* alloc = nullptr);

// ============================================================================
// Hypothesis Tests (Anderson-Darling Normality & Durbin-Watson Autocorrelation)
// ============================================================================

struct AdTestResult {
    i32 h = 0;
    f64 p = 0.0;
    f64 adstat = 0.0;
    f64 cv = 0.752;
};

// Anderson-Darling test for normality (parameters estimated, NaNs dropped).
[[nodiscard]] StatsStatus adtest(ArrayRef<f64> x, f64 alpha, AdTestResult& out,
                                 Allocator* alloc = nullptr);

struct DwTestResult {
    f64 dw = 0.0;
    f64 p_left = 0.0;
    f64 p_both = 0.0;
};

// Durbin-Watson autocorrelation test (`exact == true` uses Imhof CF inversion).
[[nodiscard]] StatsStatus dwtest(ArrayRef<f64> r, ArrayRef<f64> x_col_major, usize n, usize k,
                                 bool exact, DwTestResult& out, Allocator* alloc = nullptr);

// ============================================================================
// Copulas (Gaussian, Student-t, Clayton, Frank, Gumbel)
// ============================================================================

[[nodiscard]] f64 bivnorm_cdf(f64 a, f64 b, f64 rho) noexcept;

[[nodiscard]] StatsStatus copula_pdf_gaussian(ArrayRef<f64> u_col_major, usize n, usize d,
                                              ArrayRef<f64> r_col_major, MutArrayRef<f64> out,
                                              Allocator* alloc = nullptr);
[[nodiscard]] StatsStatus copula_cdf_gaussian_2d(ArrayRef<f64> u_col_major, usize n, f64 rho,
                                                 MutArrayRef<f64> out) noexcept;

[[nodiscard]] StatsStatus copula_pdf_t(ArrayRef<f64> u_col_major, usize n, usize d,
                                       ArrayRef<f64> r_col_major, f64 nu, MutArrayRef<f64> out,
                                       Allocator* alloc = nullptr);

[[nodiscard]] StatsStatus copula_pdf_clayton(ArrayRef<f64> u_col_major, usize n, f64 alpha,
                                             MutArrayRef<f64> out) noexcept;
[[nodiscard]] StatsStatus copula_cdf_clayton(ArrayRef<f64> u_col_major, usize n, f64 alpha,
                                             MutArrayRef<f64> out) noexcept;

[[nodiscard]] StatsStatus copula_pdf_frank(ArrayRef<f64> u_col_major, usize n, f64 alpha,
                                           MutArrayRef<f64> out) noexcept;
[[nodiscard]] StatsStatus copula_cdf_frank(ArrayRef<f64> u_col_major, usize n, f64 alpha,
                                           MutArrayRef<f64> out) noexcept;

[[nodiscard]] StatsStatus copula_pdf_gumbel(ArrayRef<f64> u_col_major, usize n, f64 alpha,
                                            MutArrayRef<f64> out) noexcept;
[[nodiscard]] StatsStatus copula_cdf_gumbel(ArrayRef<f64> u_col_major, usize n, f64 alpha,
                                            MutArrayRef<f64> out) noexcept;

// ============================================================================
// Multivariate Densities (mvnpdf, mvtpdf, mnpdf)
// ============================================================================

// Multivariate normal PDF for centered `n x d` `x0_col_major` and `d x d` PSD `sigma_col_major`.
// Returns `kSingularMatrix` if `sigma` is not positive semi-definite.
[[nodiscard]] StatsStatus mvnpdf_full(ArrayRef<f64> x0_col_major, usize n, usize d,
                                      ArrayRef<f64> sigma_col_major, MutArrayRef<f64> out,
                                      Allocator* alloc = nullptr);

// Multivariate normal PDF for centered `n x d` `x0_col_major` and positive diagonal variances.
[[nodiscard]] StatsStatus mvnpdf_diag(ArrayRef<f64> x0_col_major, usize n, usize d,
                                      ArrayRef<f64> diag_var, MutArrayRef<f64> out) noexcept;

// Multivariate Student-t PDF for `n x d` `x_col_major`, `d x d` PSD `corr_col_major`
// (automatically standardized to unit diagonal), and length-`n` positive `df_per_row`.
[[nodiscard]] StatsStatus mvtpdf_full(ArrayRef<f64> x_col_major, usize n, usize d,
                                      ArrayRef<f64> corr_col_major, ArrayRef<f64> df_per_row,
                                      MutArrayRef<f64> out, Allocator* alloc = nullptr);

// Multinomial PMF for `m x k` counts `x_col_major` and `mm x k` probabilities `p_col_major`
// (`mm == 1` or `mm == m`).
[[nodiscard]] StatsStatus mnpdf_rows(ArrayRef<f64> x_col_major, usize m, usize k,
                                     ArrayRef<f64> p_col_major, usize mm,
                                     MutArrayRef<f64> out) noexcept;

// ============================================================================
// Hypothesis Tests (t, z, var, KS, JB, sign, Fisher, chi2gof, vartestn, runs, rank, Lilliefors)
// ============================================================================

enum class TestTail : u8 {
    kBoth = 0,
    kRight,
    kLeft,
};

struct TTestResult {
    i32 h = 0;
    f64 p = 1.0;
    f64 ci_lo = 0.0;
    f64 ci_hi = 0.0;
    f64 tstat = 0.0;
    f64 df = 0.0;
    f64 sd = 0.0;
};

struct TTest2Result {
    i32 h = 0;
    f64 p = 1.0;
    f64 ci_lo = 0.0;
    f64 ci_hi = 0.0;
    f64 tstat = 0.0;
    f64 df = 0.0;
    f64 sd_x = 0.0;
    f64 sd_y = 0.0;
    f64 sd_pooled = 0.0;
};

struct ZTestResult {
    i32 h = 0;
    f64 p = 1.0;
    f64 ci_lo = 0.0;
    f64 ci_hi = 0.0;
    f64 zval = 0.0;
};

struct VarTestResult {
    i32 h = 0;
    f64 p = 1.0;
    f64 ci_lo = 0.0;
    f64 ci_hi = 0.0;
    f64 chisqstat = 0.0;
    f64 df = 0.0;
};

struct VarTest2Result {
    i32 h = 0;
    f64 p = 1.0;
    f64 ci_lo = 0.0;
    f64 ci_hi = 0.0;
    f64 fstat = 0.0;
    f64 df1 = 0.0;
    f64 df2 = 0.0;
};

struct KsTestResult {
    i32 h = 0;
    f64 p = 1.0;
    f64 ksstat = 0.0;
    f64 cv = 0.0;
};

struct JbTestResult {
    i32 h = 0;
    f64 p = 1.0;
    f64 jbstat = 0.0;
    f64 cv = 0.0;
};

struct SignTestResult {
    f64 p = 1.0;
    i32 h = 0;
    i64 sign = 0;
};

struct FisherTestResult {
    i32 h = 0;
    f64 p = 1.0;
    f64 odds_ratio = 0.0;
    f64 ci_lo = 0.0;
    f64 ci_hi = 0.0;
};

struct Chi2GofResult {
    f64 p = 1.0;
    i32 h = 0;
    f64 chi2stat = 0.0;
    f64 df = 0.0;
};

enum class VarTestNMethod : u8 {
    kBartlett = 0,
    kLeveneQuadratic,
    kLeveneAbsolute,
    kBrownForsythe,
    kOBrien,
};

struct VarTestNResult {
    f64 p = 0.0;
    f64 stat = 0.0;
    f64 df1 = 0.0;
    f64 df2 = 0.0;
};

struct RunsTestResult {
    f64 p = 1.0;
    i32 h = 0;
    i32 nruns = 0;
    i32 n1 = 0;
    i32 n0 = 0;
    f64 zval = 0.0;
};

struct RankTestResult {
    f64 p = 1.0;
    i32 h = 0;
    f64 stat = 0.0;
    f64 zval = 0.0;
};

struct LillieTestResult {
    i32 h = 0;
    f64 p = 0.5;
    f64 kstat = 0.0;
    f64 cv = 0.0;
};

[[nodiscard]] StatsStatus ttest_1sample(ArrayRef<f64> x, f64 m, f64 alpha, TestTail tail,
                                        TTestResult& out) noexcept;
[[nodiscard]] StatsStatus ttest_2sample(ArrayRef<f64> x, ArrayRef<f64> y, f64 alpha, TestTail tail,
                                        bool equal_var, TTest2Result& out) noexcept;
[[nodiscard]] StatsStatus ztest_1sample(ArrayRef<f64> x, f64 m, f64 sigma, f64 alpha, TestTail tail,
                                        ZTestResult& out) noexcept;
[[nodiscard]] StatsStatus vartest_1sample(ArrayRef<f64> x, f64 v, f64 alpha, TestTail tail,
                                          VarTestResult& out) noexcept;
[[nodiscard]] StatsStatus vartest_2sample(ArrayRef<f64> x, ArrayRef<f64> y, f64 alpha,
                                          TestTail tail, VarTest2Result& out) noexcept;
// If `cdf_x` is empty, tests against standard normal N(0, 1).
[[nodiscard]] StatsStatus kstest_1sample(ArrayRef<f64> x, ArrayRef<f64> cdf_x, ArrayRef<f64> cdf_f,
                                         f64 alpha, TestTail tail, KsTestResult& out,
                                         Allocator* alloc = nullptr);
[[nodiscard]] StatsStatus kstest_2sample(ArrayRef<f64> x, ArrayRef<f64> y, f64 alpha, TestTail tail,
                                         KsTestResult& out, Allocator* alloc = nullptr);
// `mctol`: pass quiet_nan() for default (MC when n < 2000, chi2(2) asymptotic when n >= 2000).
[[nodiscard]] StatsStatus jbtest(ArrayRef<f64> x, f64 alpha, f64 mctol, JbTestResult& out,
                                 Allocator* alloc = nullptr);
// Paired when `!y.empty()`; otherwise tests median against `m0`.
[[nodiscard]] StatsStatus signtest(ArrayRef<f64> x, ArrayRef<f64> y, f64 m0, f64 alpha,
                                   TestTail tail, SignTestResult& out) noexcept;
[[nodiscard]] StatsStatus fishertest(f64 a, f64 b, f64 c, f64 d, f64 alpha, TestTail tail,
                                     FisherTestResult& out, Allocator* alloc = nullptr);
[[nodiscard]] StatsStatus chi2gof_freq(ArrayRef<f64> observed, ArrayRef<f64> expected, i32 nparams,
                                       f64 alpha, Chi2GofResult& out) noexcept;
[[nodiscard]] StatsStatus vartestn(ArrayRef<f64> x, ArrayRef<f64> group, VarTestNMethod method,
                                   VarTestNResult& out, Allocator* alloc = nullptr);
// `threshold`: pass quiet_nan() to default to median(x).
[[nodiscard]] StatsStatus runstest(ArrayRef<f64> x, f64 threshold, f64 alpha, TestTail tail,
                                   bool exact, RunsTestResult& out, Allocator* alloc = nullptr);
// `method_mode`: 0 = auto, 1 = exact, 2 = approximate.
[[nodiscard]] StatsStatus ranksum(ArrayRef<f64> x, ArrayRef<f64> y, f64 alpha, TestTail tail,
                                  i32 method_mode, RankTestResult& out, Allocator* alloc = nullptr);
[[nodiscard]] StatsStatus signrank(ArrayRef<f64> x, ArrayRef<f64> y, f64 m0, f64 alpha,
                                   TestTail tail, i32 method_mode, RankTestResult& out,
                                   Allocator* alloc = nullptr);
[[nodiscard]] StatsStatus ansaribradley(ArrayRef<f64> x, ArrayRef<f64> y, f64 alpha, TestTail tail,
                                        RankTestResult& out, Allocator* alloc = nullptr);
[[nodiscard]] StatsStatus lillietest(ArrayRef<f64> x, f64 alpha, LillieTestResult& out,
                                     Allocator* alloc = nullptr);

// ============================================================================
// Distribution MLE Fitters & Wald Confidence Intervals
// ============================================================================

struct Fit2Result {
    f64 p1 = 0.0;
    f64 p2 = 0.0;
    f64 ci1_lo = 0.0;
    f64 ci1_hi = 0.0;
    f64 ci2_lo = 0.0;
    f64 ci2_hi = 0.0;
};

struct Fit3Result {
    f64 p1 = 0.0;
    f64 p2 = 0.0;
    f64 p3 = 0.0;
    f64 ci1_lo = 0.0;
    f64 ci1_hi = 0.0;
    f64 ci2_lo = 0.0;
    f64 ci2_hi = 0.0;
    f64 ci3_lo = 0.0;
    f64 ci3_hi = 0.0;
};

// `cens` and `freq` may be empty (treated as all 0 and all 1 respectively).
[[nodiscard]] StatsStatus gamfit(ArrayRef<f64> x, ArrayRef<f64> cens, ArrayRef<f64> freq, f64 alpha,
                                 bool compute_ci, Fit2Result& out) noexcept;
[[nodiscard]] StatsStatus wblfit(ArrayRef<f64> x, ArrayRef<f64> cens, ArrayRef<f64> freq, f64 alpha,
                                 bool compute_ci, Fit2Result& out, Allocator* alloc = nullptr);
[[nodiscard]] StatsStatus betafit(ArrayRef<f64> x, f64 alpha, bool compute_ci,
                                  Fit2Result& out) noexcept;
[[nodiscard]] StatsStatus nbinfit(ArrayRef<f64> x, f64 alpha, bool compute_ci,
                                  Fit2Result& out) noexcept;
[[nodiscard]] StatsStatus evfit(ArrayRef<f64> x, ArrayRef<f64> cens, ArrayRef<f64> freq, f64 alpha,
                                bool compute_ci, Fit2Result& out) noexcept;
[[nodiscard]] StatsStatus gpfit(ArrayRef<f64> x, f64 alpha, bool compute_ci, Fit2Result& out,
                                Allocator* alloc = nullptr);
[[nodiscard]] StatsStatus gevfit(ArrayRef<f64> x, f64 alpha, bool compute_ci, Fit3Result& out,
                                 Allocator* alloc = nullptr);

} // namespace keel::stats
