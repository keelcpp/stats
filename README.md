# keel::stats

Zero-STL C++20 probability distributions and descriptive statistics library built on [`keel`](https://github.com/keelcpp/keel) and [`keel::specfun`](https://github.com/keelcpp/specfun).

## Features

- **14 univariate probability distributions** (PDF, CDF, inverse CDF / quantile, analytical mean & variance):
  - Continuous: Normal (`norm_*`), Student's t (`t_*`), Chi-squared (`chi2_*`), Fisher F (`f_*`), Gamma (`gam_*`), Beta (`beta_*`), Exponential (`exp_*`), Weibull (`wbl_*`), Lognormal (`logn_*`), Uniform (`unif_*`), Rayleigh (`rayl_*`), Generalized Extreme Value (`gev_*`)
  - Discrete: Poisson (`poiss_*`), Binomial (`bino_*`)
- **Descriptive statistics**: `mean`, `var`, `stddev`, `skewness`, `kurtosis`, `median`, `quantile`, `iqr`, `cov`, `corrcoef`, `zscore`
- **Pairwise distances & hierarchical clustering**: `pdist`, `pdist2` (12 metrics), `squareform_to_matrix`/`condensed`, rank-safe QR `mahal_qr`, `linkage` (7 Lance-Williams methods), `inconsistent`, `cluster_maxclust`, `cluster_cutoff`, `cophenet`
- **Regularized regression**: cyclic coordinate-descent `lasso` (LASSO / Elastic-Net) and IRLS `lassoglm` (Normal / Binomial / Poisson)
- **ANOVA & nonparametric rank tests**: `anova1_grouped`, `anova2`, `kruskal_wallis`, `friedman`
- Zero STL headers in public headers and implementation; full `-fno-exceptions -fno-rtti` support

## Build & Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```
