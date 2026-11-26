#include "keel/stats.h"

#include "keel/math.h"

namespace keel::stats {

namespace {

inline bool mul_overflows(usize a, usize b) noexcept {
    return a != 0u && b > Limits<usize>::kMax / a;
}

// Column-major Cholesky factorization M = L * L^T (lower-triangular L).
bool cholesky_col(const f64* M, f64* L, usize d) noexcept {
    for (usize i = 0; i < d * d; ++i) L[i] = 0.0;
    for (usize j = 0; j < d; ++j) {
        f64 s = M[j + j * d];
        for (usize k = 0; k < j; ++k) s -= L[j + k * d] * L[j + k * d];
        if (!(s > 0.0) || !is_finite(s)) return false;
        const f64 ljj = sqrt(s);
        L[j + j * d] = ljj;
        for (usize i = j + 1; i < d; ++i) {
            f64 t = M[i + j * d];
            for (usize k = 0; k < j; ++k) t -= L[i + k * d] * L[j + k * d];
            L[i + j * d] = t / ljj;
        }
    }
    return true;
}

void fwd_solve_col(const f64* L, f64* z, const f64* b, usize d) noexcept {
    for (usize i = 0; i < d; ++i) {
        f64 s = b[i];
        for (usize k = 0; k < i; ++k) s -= L[i + k * d] * z[k];
        z[i] = s / L[i + i * d];
    }
}

void back_solve_col(const f64* L, f64* x, const f64* z, usize d) noexcept {
    for (usize i = d; i-- > 0;) {
        f64 s = z[i];
        for (usize k = i + 1; k < d; ++k) s -= L[k + i * d] * x[k];
        x[i] = s / L[i + i * d];
    }
}

// In-place Gaussian elimination with partial pivoting on row-major p x p M, length-p b.
bool gauss_solve_row(f64* M, f64* b, usize p) noexcept {
    for (usize k = 0; k < p; ++k) {
        usize piv = k;
        f64 pmax = abs(M[k * p + k]);
        for (usize rr = k + 1; rr < p; ++rr) {
            const f64 v = abs(M[rr * p + k]);
            if (v > pmax) {
                pmax = v;
                piv = rr;
            }
        }
        if (!(pmax > 0.0) || !is_finite(pmax)) return false;
        if (piv != k) {
            for (usize j = 0; j < p; ++j) keel::swap(M[k * p + j], M[piv * p + j]);
            keel::swap(b[k], b[piv]);
        }
        const f64 pv = M[k * p + k];
        for (usize rr = k + 1; rr < p; ++rr) {
            const f64 f = M[rr * p + k] / pv;
            for (usize j = k; j < p; ++j) M[rr * p + j] -= f * M[k * p + j];
            b[rr] -= f * b[k];
        }
    }
    for (usize k = p; k-- > 0;) {
        f64 s = b[k];
        for (usize j = k + 1; j < p; ++j) s -= M[k * p + j] * b[j];
        b[k] = s / M[k * p + k];
    }
    return true;
}

inline f64 sigmoid(f64 x) noexcept {
    if (x > 0.0) return 1.0 / (1.0 + exp(-x));
    const f64 e = exp(x);
    return e / (1.0 + e);
}

inline f64 inv_link(f64 e, GlmLink link) noexcept {
    switch (link) {
        case GlmLink::kIdentity:
            return e;
        case GlmLink::kLogit:
            return sigmoid(e);
        case GlmLink::kLog:
            return exp(clamp(e, -500.0, 500.0));
        case GlmLink::kReciprocal: {
            const f64 r = (abs(e) > 1e-300) ? e : ((e < 0.0) ? -1e-300 : 1e-300);
            return 1.0 / r;
        }
        case GlmLink::kProbit:
            return norm_cdf(e, 0.0, 1.0);
    }
    return e;
}

inline f64 deta_dmu(f64 mu, GlmLink link) noexcept {
    switch (link) {
        case GlmLink::kIdentity:
            return 1.0;
        case GlmLink::kLogit: {
            const f64 m = clamp(mu, 1e-12, 1.0 - 1e-12);
            return 1.0 / (m * (1.0 - m));
        }
        case GlmLink::kLog:
            return 1.0 / max(mu, 1e-12);
        case GlmLink::kReciprocal: {
            const f64 m = (abs(mu) > 1e-300) ? mu : 1e-300;
            return -1.0 / (m * m);
        }
        case GlmLink::kProbit: {
            const f64 m = clamp(mu, 1e-12, 1.0 - 1e-12);
            const f64 z = norm_inv(m, 0.0, 1.0);
            const f64 phi = norm_pdf(z, 0.0, 1.0);
            return (phi > 1e-300) ? (1.0 / phi) : 1e300;
        }
    }
    return 1.0;
}

inline f64 variance_fn(f64 mu, GlmDist d) noexcept {
    switch (d) {
        case GlmDist::kNormal:
            return 1.0;
        case GlmDist::kBinomial: {
            const f64 m = clamp(mu, 1e-12, 1.0 - 1e-12);
            return m * (1.0 - m);
        }
        case GlmDist::kPoisson:
            return max(mu, 1e-12);
        case GlmDist::kGamma: {
            const f64 m = max(mu, 1e-12);
            return m * m;
        }
        case GlmDist::kInverseGaussian: {
            const f64 m = max(mu, 1e-12);
            return m * m * m;
        }
    }
    return 1.0;
}

inline f64 initial_mu(f64 yi, GlmDist d) noexcept {
    switch (d) {
        case GlmDist::kNormal:
            return yi;
        case GlmDist::kBinomial:
            return (clamp(yi, 0.0, 1.0) + 0.5) / 2.0;
        case GlmDist::kPoisson:
            return max(yi + 0.1, 0.1);
        case GlmDist::kGamma:
        case GlmDist::kInverseGaussian:
            return max(yi, 0.1);
    }
    return yi;
}

inline f64 dev_term(f64 y, f64 mu, GlmDist d) noexcept {
    switch (d) {
        case GlmDist::kNormal: {
            const f64 r = y - mu;
            return r * r;
        }
        case GlmDist::kBinomial: {
            const f64 m = clamp(mu, 1e-12, 1.0 - 1e-12);
            const f64 yy = clamp(y, 0.0, 1.0);
            f64 t = 0.0;
            if (yy > 0.0) t += yy * log(yy / m);
            if (yy < 1.0) t += (1.0 - yy) * log((1.0 - yy) / (1.0 - m));
            return 2.0 * t;
        }
        case GlmDist::kPoisson: {
            const f64 m = max(mu, 1e-12);
            const f64 yy = max(y, 0.0);
            const f64 term_y = (yy > 0.0) ? (yy * log(yy / m)) : 0.0;
            return 2.0 * (term_y - (yy - m));
        }
        case GlmDist::kGamma: {
            const f64 m = max(mu, 1e-12);
            const f64 yy = max(y, 1e-12);
            return 2.0 * (-log(yy / m) + (yy - m) / m);
        }
        case GlmDist::kInverseGaussian: {
            const f64 m = max(mu, 1e-12);
            const f64 yy = max(y, 1e-12);
            const f64 r = yy - m;
            return r * r / (yy * m * m);
        }
    }
    return 0.0;
}

inline GlmLink canonical_link(GlmDist d) noexcept {
    switch (d) {
        case GlmDist::kNormal:
            return GlmLink::kIdentity;
        case GlmDist::kBinomial:
            return GlmLink::kLogit;
        case GlmDist::kPoisson:
            return GlmLink::kLog;
        case GlmDist::kGamma:
        case GlmDist::kInverseGaussian:
            return GlmLink::kReciprocal;
    }
    return GlmLink::kIdentity;
}

} // namespace

StatsStatus regress(ArrayRef<f64> y, ArrayRef<f64> x_col_major, usize n, usize p, f64 alpha,
                    MutArrayRef<f64> out_beta, MutArrayRef<f64> out_bint_col_major,
                    MutArrayRef<f64> out_r, MutArrayRef<f64> out_rint_col_major,
                    RegressStats& out_stats, Allocator* alloc) {
    if (n == 0 || p == 0 || !(alpha > 0.0 && alpha < 1.0) || mul_overflows(n, p) ||
        mul_overflows(p, p) || y.size() != n || x_col_major.size() != n * p) {
        return StatsStatus::kInvalidArgument;
    }
    if (out_beta.size() != p || out_bint_col_major.size() != 2 * p || out_r.size() != n ||
        out_rint_col_major.size() != 2 * n) {
        return StatsStatus::kInvalidArgument;
    }

    Array<f64> xtx(p * p, 0.0, alloc);
    Array<f64> xty(p, 0.0, alloc);
    for (usize j = 0; j < p; ++j) {
        for (usize i = 0; i < n; ++i) {
            const f64 xij = x_col_major[i + j * n];
            xty[j] += xij * y[i];
            for (usize k = j; k < p; ++k) {
                xtx[k + j * p] += xij * x_col_major[i + k * n];
            }
        }
    }
    for (usize j = 0; j < p; ++j) {
        for (usize k = 0; k < j; ++k) xtx[k + j * p] = xtx[j + k * p];
    }

    Array<f64> L(p * p, 0.0, alloc);
    if (!cholesky_col(xtx.data(), L.data(), p)) return StatsStatus::kSingularMatrix;

    Array<f64> z(p, 0.0, alloc);
    fwd_solve_col(L.data(), z.data(), xty.data(), p);
    back_solve_col(L.data(), out_beta.data(), z.data(), p);

    f64 ssr = 0.0;
    for (usize i = 0; i < n; ++i) {
        f64 pred = 0.0;
        for (usize j = 0; j < p; ++j) pred += x_col_major[i + j * n] * out_beta[j];
        const f64 ri = y[i] - pred;
        out_r[i] = ri;
        ssr += ri * ri;
    }

    const f64 nan_v = Limits<f64>::quiet_nan();
    const f64 df_err = as<f64>(n) - as<f64>(p);
    const f64 sigma2 = (df_err > 0.0) ? (ssr / df_err) : nan_v;

    Array<f64> inv_mat(p * p, 0.0, alloc);
    Array<f64> inv_diag(p, 0.0, alloc);
    {
        Array<f64> ej(p, 0.0, alloc);
        Array<f64> zj(p, 0.0, alloc);
        Array<f64> mj(p, 0.0, alloc);
        for (usize j = 0; j < p; ++j) {
            for (usize i = 0; i < p; ++i) ej[i] = 0.0;
            ej[j] = 1.0;
            fwd_solve_col(L.data(), zj.data(), ej.data(), p);
            back_solve_col(L.data(), mj.data(), zj.data(), p);
            for (usize i = 0; i < p; ++i) inv_mat[i + j * p] = mj[i];
            inv_diag[j] = mj[j];
        }
    }

    const f64 tcrit = (df_err > 0.0) ? t_inv(1.0 - alpha / 2.0, df_err) : 0.0;
    for (usize j = 0; j < p; ++j) {
        const f64 se = (!is_nan(sigma2) && inv_diag[j] > 0.0) ? sqrt(sigma2 * inv_diag[j]) : nan_v;
        out_bint_col_major[j] = out_beta[j] - tcrit * se;
        out_bint_col_major[j + p] = out_beta[j] + tcrit * se;
    }

    f64 y_mean = 0.0;
    for (usize i = 0; i < n; ++i) y_mean += y[i];
    y_mean /= as<f64>(n);
    f64 ss_tot = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 d = y[i] - y_mean;
        ss_tot += d * d;
    }
    const f64 r2 = (ss_tot > 0.0) ? (1.0 - ssr / ss_tot) : nan_v;
    const f64 df_model = as<f64>(p) - 1.0;
    f64 f_stat = nan_v;
    f64 p_f = nan_v;
    if (df_model > 0.0 && df_err > 0.0 && !is_nan(r2)) {
        if (ssr == 0.0 && ss_tot > 0.0) {
            f_stat = Limits<f64>::infinity();
            p_f = 0.0;
        } else if ((1.0 - r2) > 0.0) {
            f_stat = (r2 / df_model) / ((1.0 - r2) / df_err);
            p_f = max(0.0, 1.0 - f_cdf(f_stat, df_model, df_err));
        }
    }
    out_stats = RegressStats{r2, f_stat, p_f, sigma2};

    const f64 nu = df_err;
    const f64 rmse = (!is_nan(sigma2) && sigma2 > 0.0) ? sqrt(sigma2) : 0.0;
    const f64 tcrit_r = (nu > 1.0) ? t_inv(1.0 - alpha / 2.0, nu - 1.0) : tcrit;
    constexpr f64 kEpsThresh = 1.4901161193847656e-08;
    for (usize i = 0; i < n; ++i) {
        f64 hii = 0.0;
        for (usize j = 0; j < p; ++j) {
            f64 row = 0.0;
            for (usize k = 0; k < p; ++k) row += inv_mat[j + k * p] * x_col_major[i + k * n];
            hii += x_col_major[i + j * n] * row;
        }
        const f64 one_minus_h = 1.0 - hii;
        f64 ser = 0.0;
        if (one_minus_h > kEpsThresh && is_finite(one_minus_h)) {
            if (nu > 1.0) {
                const f64 denom = (nu - 1.0) * one_minus_h;
                const f64 sigma_i =
                    sqrt(max(0.0, (nu * sigma2 / (nu - 1.0)) - (out_r[i] * out_r[i]) / denom));
                ser = sqrt(one_minus_h) * sigma_i;
            } else {
                ser = sqrt(one_minus_h) * rmse;
            }
        }
        out_rint_col_major[i] = out_r[i] - tcrit_r * ser;
        out_rint_col_major[i + n] = out_r[i] + tcrit_r * ser;
    }

    return StatsStatus::kOk;
}

StatsStatus lscov(ArrayRef<f64> a_col_major, ArrayRef<f64> b, usize n, usize p,
                  ArrayRef<f64> weights, bool is_full_cov, MutArrayRef<f64> out_x,
                  MutArrayRef<f64> out_stdx, f64& out_mse, MutArrayRef<f64> out_s_col_major,
                  Allocator* alloc) {
    if (n == 0 || p == 0 || mul_overflows(n, p) || mul_overflows(p, p) || b.size() != n ||
        a_col_major.size() != n * p) {
        return StatsStatus::kInvalidArgument;
    }
    if (out_x.size() != p || out_stdx.size() != p || out_s_col_major.size() != p * p) {
        return StatsStatus::kInvalidArgument;
    }
    const bool weighted = !weights.empty();
    const bool full_cov = weighted && is_full_cov;
    if (full_cov && (mul_overflows(n, n) || weights.size() != n * n)) {
        return StatsStatus::kInvalidArgument;
    }
    if (weighted && !full_cov && weights.size() != n) return StatsStatus::kInvalidArgument;
    if (weighted && !full_cov) {
        for (usize i = 0; i < n; ++i) {
            if (!(weights[i] >= 0.0) || !is_finite(weights[i])) {
                return StatsStatus::kInvalidArgument;
            }
        }
    }

    Array<f64> xtwx(p * p, 0.0, alloc);
    Array<f64> xtwy(p, 0.0, alloc);
    Array<f64> a_tilde(alloc);
    Array<f64> b_tilde(alloc);

    if (full_cov) {
        Array<f64> V(n * n, 0.0, alloc);
        for (usize j = 0; j < n; ++j) {
            for (usize i = 0; i < n; ++i) {
                const f64 vij = weights[i + j * n];
                const f64 vji = weights[j + i * n];
                if (!is_finite(vij) || abs(vij - vji) > 1e-10 * (1.0 + abs(vij))) {
                    return StatsStatus::kInvalidArgument;
                }
                V[i + j * n] = 0.5 * (vij + vji);
            }
        }
        Array<f64> lv(n * n, 0.0, alloc);
        if (!cholesky_col(V.data(), lv.data(), n)) return StatsStatus::kInvalidArgument;

        a_tilde.resize(n * p, 0.0);
        b_tilde.resize(n, 0.0);
        Array<f64> col_in(n, 0.0, alloc);
        for (usize j = 0; j < p; ++j) {
            for (usize i = 0; i < n; ++i) col_in[i] = a_col_major[i + j * n];
            fwd_solve_col(lv.data(), a_tilde.data() + j * n, col_in.data(), n);
        }
        for (usize i = 0; i < n; ++i) col_in[i] = b[i];
        fwd_solve_col(lv.data(), b_tilde.data(), col_in.data(), n);

        for (usize i = 0; i < n; ++i) {
            const f64 bi = b_tilde[i];
            for (usize j = 0; j < p; ++j) {
                const f64 xij = a_tilde[i + j * n];
                xtwy[j] += xij * bi;
                for (usize k = j; k < p; ++k) xtwx[k + j * p] += xij * a_tilde[i + k * n];
            }
        }
    } else {
        for (usize i = 0; i < n; ++i) {
            const f64 wii = weighted ? weights[i] : 1.0;
            const f64 bi = b[i];
            for (usize j = 0; j < p; ++j) {
                const f64 xij = a_col_major[i + j * n];
                xtwy[j] += wii * xij * bi;
                for (usize k = j; k < p; ++k) xtwx[k + j * p] += wii * xij * a_col_major[i + k * n];
            }
        }
    }
    for (usize j = 0; j < p; ++j) {
        for (usize k = 0; k < j; ++k) xtwx[k + j * p] = xtwx[j + k * p];
    }

    Array<f64> L(p * p, 0.0, alloc);
    if (!cholesky_col(xtwx.data(), L.data(), p)) return StatsStatus::kSingularMatrix;

    Array<f64> z(p, 0.0, alloc);
    fwd_solve_col(L.data(), z.data(), xtwy.data(), p);
    back_solve_col(L.data(), out_x.data(), z.data(), p);

    f64 ssr_w = 0.0;
    if (full_cov) {
        for (usize i = 0; i < n; ++i) {
            f64 pred = 0.0;
            for (usize j = 0; j < p; ++j) pred += a_tilde[i + j * n] * out_x[j];
            const f64 ri = b_tilde[i] - pred;
            ssr_w += ri * ri;
        }
    } else {
        for (usize i = 0; i < n; ++i) {
            f64 pred = 0.0;
            for (usize j = 0; j < p; ++j) pred += a_col_major[i + j * n] * out_x[j];
            const f64 ri = b[i] - pred;
            const f64 wii = weighted ? weights[i] : 1.0;
            ssr_w += wii * ri * ri;
        }
    }

    const f64 df_err = as<f64>(n) - as<f64>(p);
    out_mse = (df_err > 0.0) ? (ssr_w / df_err) : Limits<f64>::quiet_nan();

    Array<f64> ej(p, 0.0, alloc);
    Array<f64> zj(p, 0.0, alloc);
    Array<f64> mj(p, 0.0, alloc);
    for (usize j = 0; j < p; ++j) {
        for (usize i = 0; i < p; ++i) ej[i] = 0.0;
        ej[j] = 1.0;
        fwd_solve_col(L.data(), zj.data(), ej.data(), p);
        back_solve_col(L.data(), mj.data(), zj.data(), p);
        for (usize i = 0; i < p; ++i) out_s_col_major[i + j * p] = out_mse * mj[i];
    }
    for (usize j = 0; j < p; ++j) {
        const f64 v = out_s_col_major[j + j * p];
        out_stdx[j] = is_nan(v) ? v : sqrt(max(v, 0.0));
    }
    return StatsStatus::kOk;
}

StatsStatus ridge(ArrayRef<f64> y, ArrayRef<f64> x_col_major, usize n, usize p,
                  ArrayRef<f64> k_vals, bool scaled, MutArrayRef<f64> out_b_col_major,
                  Allocator* alloc) {
    const usize nk = k_vals.size();
    const usize out_rows = scaled ? p : (p + 1);
    if (n == 0 || p == 0 || nk == 0 || mul_overflows(n, p) || mul_overflows(p, p) ||
        mul_overflows(out_rows, nk) || y.size() != n || x_col_major.size() != n * p ||
        out_b_col_major.size() != out_rows * nk) {
        return StatsStatus::kInvalidArgument;
    }
    for (usize i = 0; i < nk; ++i) {
        if (!(k_vals[i] >= 0.0) || !is_finite(k_vals[i])) return StatsStatus::kInvalidArgument;
    }

    f64 y_mean = 0.0;
    for (usize i = 0; i < n; ++i) y_mean += y[i];
    y_mean /= as<f64>(n);
    Array<f64> yc(n, 0.0, alloc);
    for (usize i = 0; i < n; ++i) yc[i] = y[i] - y_mean;

    Array<f64> x_mean(p, 0.0, alloc);
    Array<f64> x_std(p, 0.0, alloc);
    Array<f64> xs(n * p, 0.0, alloc);
    for (usize j = 0; j < p; ++j) {
        for (usize i = 0; i < n; ++i) x_mean[j] += x_col_major[i + j * n];
        x_mean[j] /= as<f64>(n);
        f64 sq = 0.0;
        for (usize i = 0; i < n; ++i) {
            const f64 d = x_col_major[i + j * n] - x_mean[j];
            xs[i + j * n] = d;
            sq += d * d;
        }
        const f64 s = (n > 1) ? sqrt(sq / as<f64>(n - 1)) : 1.0;
        x_std[j] = s;
        if (s > 0.0) {
            for (usize i = 0; i < n; ++i) xs[i + j * n] /= s;
        }
    }

    Array<f64> xtx(p * p, 0.0, alloc);
    Array<f64> xty(p, 0.0, alloc);
    for (usize i = 0; i < n; ++i) {
        const f64 yi = yc[i];
        for (usize j = 0; j < p; ++j) {
            const f64 xij = xs[i + j * n];
            xty[j] += xij * yi;
            for (usize kk = j; kk < p; ++kk) xtx[kk + j * p] += xij * xs[i + kk * n];
        }
    }
    for (usize j = 0; j < p; ++j) {
        for (usize kk = 0; kk < j; ++kk) xtx[kk + j * p] = xtx[j + kk * p];
    }

    Array<f64> M(p * p, 0.0, alloc);
    Array<f64> L(p * p, 0.0, alloc);
    Array<f64> beta(p, 0.0, alloc);
    Array<f64> z(p, 0.0, alloc);

    for (usize k_idx = 0; k_idx < nk; ++k_idx) {
        const f64 k = k_vals[k_idx];
        for (usize j = 0; j < p * p; ++j) M[j] = xtx[j];
        for (usize j = 0; j < p; ++j) M[j + j * p] += k;
        if (!cholesky_col(M.data(), L.data(), p)) return StatsStatus::kSingularMatrix;
        fwd_solve_col(L.data(), z.data(), xty.data(), p);
        back_solve_col(L.data(), beta.data(), z.data(), p);
        if (scaled) {
            for (usize j = 0; j < p; ++j) out_b_col_major[j + k_idx * out_rows] = beta[j];
        } else {
            f64 intercept = y_mean;
            for (usize j = 0; j < p; ++j) {
                const f64 bj = (x_std[j] > 0.0) ? (beta[j] / x_std[j]) : 0.0;
                out_b_col_major[(j + 1) + k_idx * out_rows] = bj;
                intercept -= bj * x_mean[j];
            }
            out_b_col_major[0 + k_idx * out_rows] = intercept;
        }
    }
    return StatsStatus::kOk;
}

StatsStatus glmfit(ArrayRef<f64> x_col_major, ArrayRef<f64> y, usize n, usize p_in, GlmDist distr,
                   GlmLink link, GlmFitOut& out, Allocator* alloc) {
    const usize p = p_in + 1;
    if (n == 0 || mul_overflows(n, p) || mul_overflows(p, p) || y.size() != n ||
        x_col_major.size() != n * p_in || n <= p) {
        return StatsStatus::kInvalidArgument;
    }

    if (link == GlmLink::kIdentity && distr != GlmDist::kNormal) {
        link = canonical_link(distr);
    }

    Array<f64> xb(n * p, 0.0, alloc);
    for (usize i = 0; i < n; ++i) xb[i] = 1.0;
    for (usize j = 0; j < p_in; ++j) {
        for (usize i = 0; i < n; ++i) xb[(j + 1) * n + i] = x_col_major[j * n + i];
    }

    Array<f64> mu(n, 0.0, alloc);
    Array<f64> eta(n, 0.0, alloc);
    for (usize i = 0; i < n; ++i) {
        mu[i] = initial_mu(y[i], distr);
        switch (link) {
            case GlmLink::kIdentity:
                eta[i] = mu[i];
                break;
            case GlmLink::kLogit: {
                const f64 m = clamp(mu[i], 1e-12, 1.0 - 1e-12);
                eta[i] = log(m / (1.0 - m));
                break;
            }
            case GlmLink::kLog:
                eta[i] = log(max(mu[i], 1e-12));
                break;
            case GlmLink::kReciprocal:
                eta[i] = 1.0 / max(mu[i], 1e-12);
                break;
            case GlmLink::kProbit: {
                const f64 m = clamp(mu[i], 1e-12, 1.0 - 1e-12);
                eta[i] = norm_inv(m, 0.0, 1.0);
                break;
            }
        }
    }

    Array<f64> beta(p, 0.0, alloc);
    Array<f64> beta_prev(p, 0.0, alloc);
    Array<f64> z(n, 0.0, alloc);
    Array<f64> w(n, 0.0, alloc);
    Array<f64> M(p * p, 0.0, alloc);
    Array<f64> b(p, 0.0, alloc);

    constexpr i32 kMaxIter = 100;
    constexpr f64 kTol = 1e-8;
    for (i32 it = 0; it < kMaxIter; ++it) {
        for (usize i = 0; i < n; ++i) {
            const f64 de = deta_dmu(mu[i], link);
            const f64 v = variance_fn(mu[i], distr);
            w[i] = 1.0 / (v * de * de + 1e-30);
            z[i] = eta[i] + (y[i] - mu[i]) * de;
        }
        for (usize i = 0; i < p; ++i) {
            for (usize j = 0; j < p; ++j) {
                f64 s = 0.0;
                for (usize k = 0; k < n; ++k) s += xb[i * n + k] * w[k] * xb[j * n + k];
                M[i * p + j] = s;
            }
            f64 s = 0.0;
            for (usize k = 0; k < n; ++k) s += xb[i * n + k] * w[k] * z[k];
            b[i] = s;
        }
        if (!gauss_solve_row(M.data(), b.data(), p)) {
            if (it == 0) return StatsStatus::kSingularMatrix;
            break;
        }

        for (usize j = 0; j < p; ++j) {
            beta_prev[j] = beta[j];
            beta[j] = b[j];
        }
        for (usize i = 0; i < n; ++i) {
            f64 e = 0.0;
            for (usize j = 0; j < p; ++j) e += xb[j * n + i] * beta[j];
            eta[i] = e;
            mu[i] = inv_link(e, link);
        }

        f64 max_step = 0.0;
        f64 max_beta = 0.0;
        for (usize j = 0; j < p; ++j) {
            max_step = max(max_step, abs(beta[j] - beta_prev[j]));
            max_beta = max(max_beta, abs(beta[j]));
        }
        if (it > 0 && max_step < kTol * max(max_beta, 1.0)) break;
    }

    f64 dev = 0.0;
    f64 pearson_chi2 = 0.0;
    out.wts.resize(n);
    out.resid.resize(n);
    for (usize i = 0; i < n; ++i) {
        dev += dev_term(y[i], mu[i], distr);
        const f64 de = deta_dmu(mu[i], link);
        const f64 v = max(variance_fn(mu[i], distr), 1e-30);
        out.wts[i] = 1.0 / (v * de * de + 1e-30);
        const f64 r = y[i] - mu[i];
        out.resid[i] = r;
        pearson_chi2 += (r * r) / v;
    }

    const f64 dfe = as<f64>(n - p);
    const f64 sfit = sqrt(max(0.0, dev) / dfe);
    const bool estdisp = (distr == GlmDist::kNormal || distr == GlmDist::kGamma ||
                          distr == GlmDist::kInverseGaussian);
    const f64 s_disp =
        !estdisp ? 1.0 : ((distr == GlmDist::kNormal) ? sfit : sqrt(pearson_chi2 / dfe));

    Array<f64> J(p * p, 0.0, alloc);
    for (usize i = 0; i < p; ++i) {
        for (usize j = 0; j < p; ++j) {
            f64 acc = 0.0;
            for (usize k = 0; k < n; ++k) acc += xb[i * n + k] * out.wts[k] * xb[j * n + k];
            J[i * p + j] = acc;
        }
    }

    out.beta.resize(p);
    for (usize j = 0; j < p; ++j) out.beta[j] = beta[j];
    out.deviance = dev;
    out.dfe = dfe;
    out.sfit = sfit;
    out.s = s_disp;
    out.estdisp = estdisp;

    out.covb.resize(p * p);
    for (usize i = 0; i < p * p; ++i) out.covb[i] = 0.0;
    Array<f64> rhs(p, 0.0, alloc);
    for (usize col = 0; col < p; ++col) {
        for (usize i = 0; i < p * p; ++i) M[i] = J[i];
        for (usize i = 0; i < p; ++i) rhs[i] = 0.0;
        rhs[col] = 1.0;
        if (gauss_solve_row(M.data(), rhs.data(), p)) {
            for (usize row = 0; row < p; ++row) {
                out.covb[row + col * p] = (s_disp * s_disp) * rhs[row];
            }
        }
    }

    out.se.resize(p);
    out.t_stat.resize(p);
    for (usize j = 0; j < p; ++j) {
        const f64 sej = sqrt(max(0.0, out.covb[j + j * p]));
        out.se[j] = sej;
        out.t_stat[j] = (sej > 0.0) ? (beta[j] / sej) : 0.0;
    }

    out.coeffcorr.resize(p * p);
    for (usize j = 0; j < p; ++j) {
        for (usize k = 0; k < p; ++k) {
            const f64 sej = out.se[j];
            const f64 sek = out.se[k];
            out.coeffcorr[j + k * p] =
                (sej > 0.0 && sek > 0.0) ? (out.covb[j + k * p] / (sej * sek)) : (j == k ? 1.0 : 0.0);
        }
    }
    return StatsStatus::kOk;
}

StatsStatus glmval(ArrayRef<f64> beta, ArrayRef<f64> x_col_major, usize m, usize p_in,
                   GlmLink link, MutArrayRef<f64> out_y) noexcept {
    if (mul_overflows(m, p_in) || beta.size() != p_in + 1 || x_col_major.size() != m * p_in ||
        out_y.size() != m) {
        return StatsStatus::kInvalidArgument;
    }
    for (usize i = 0; i < m; ++i) {
        f64 eta = beta[0];
        for (usize j = 0; j < p_in; ++j) eta += x_col_major[j * m + i] * beta[j + 1];
        out_y[i] = inv_link(eta, link);
    }
    return StatsStatus::kOk;
}

} // namespace keel::stats
