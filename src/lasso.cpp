#include "keel/stats.h"

#include "keel/algo.h"
#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"

namespace keel::stats {

namespace {

inline bool mul_overflows(usize a, usize b) noexcept {
    return a != 0u && b > Limits<usize>::kMax / a;
}

inline f64 soft_threshold(f64 z, f64 g) noexcept {
    if (z > g) return z - g;
    if (z < -g) return z + g;
    return 0.0;
}

void standardise_cols(MutArrayRef<f64> x, usize n, usize p, MutArrayRef<f64> mu,
                      MutArrayRef<f64> sd) noexcept {
    for (usize j = 0; j < p; ++j) {
        f64 s = 0.0;
        for (usize i = 0; i < n; ++i) s += x[j * n + i];
        mu[j] = s / as<f64>(n);
        f64 s2 = 0.0;
        for (usize i = 0; i < n; ++i) {
            const f64 d = x[j * n + i] - mu[j];
            s2 += d * d;
        }
        sd[j] = keel::sqrt(s2 / as<f64>(n));
        if (sd[j] < 1e-12) sd[j] = 1.0;
        for (usize i = 0; i < n; ++i) {
            x[j * n + i] = (x[j * n + i] - mu[j]) / sd[j];
        }
    }
}

bool coord_descent_sweep(ArrayRef<f64> xs, usize n, usize p, MutArrayRef<f64> r,
                         MutArrayRef<f64> beta, ArrayRef<f64> xj_norm_sq, f64 lambda_eff,
                         f64 lambda_l2, f64 tol) noexcept {
    f64 max_change = 0.0;
    const f64 fn = as<f64>(n);
    for (usize j = 0; j < p; ++j) {
        const f64 denom = xj_norm_sq[j] / fn + lambda_l2;
        if (!(denom > 0.0) || xj_norm_sq[j] == 0.0) {
            const f64 old_b = beta[j];
            if (old_b != 0.0) {
                for (usize i = 0; i < n; ++i) r[i] += old_b * xs[j * n + i];
                beta[j] = 0.0;
                const f64 ad = keel::abs(old_b);
                if (ad > max_change) max_change = ad;
            }
            continue;
        }
        const f64 old_b = beta[j];
        f64 xr = 0.0;
        for (usize i = 0; i < n; ++i) xr += xs[j * n + i] * r[i];
        const f64 z = xr + xj_norm_sq[j] * old_b;
        const f64 new_b = soft_threshold(z / fn, lambda_eff) / denom;
        beta[j] = new_b;
        const f64 delta = new_b - old_b;
        if (delta != 0.0) {
            for (usize i = 0; i < n; ++i) r[i] -= delta * xs[j * n + i];
            const f64 ad = keel::abs(delta);
            if (ad > max_change) max_change = ad;
        }
    }
    return max_change < tol;
}

inline f64 inv_link(f64 e, GlmFamily fam) noexcept {
    switch (fam) {
        case GlmFamily::kNormal:
            return e;
        case GlmFamily::kBinomial: {
            if (e > 0.0) return 1.0 / (1.0 + keel::exp(-e));
            const f64 ex = keel::exp(e);
            return ex / (1.0 + ex);
        }
        case GlmFamily::kPoisson:
            return keel::exp(clamp(e, -500.0, 500.0));
    }
    return e;
}

inline f64 d_eta(f64 mu, GlmFamily fam) noexcept {
    switch (fam) {
        case GlmFamily::kNormal:
            return 1.0;
        case GlmFamily::kBinomial: {
            const f64 m = keel::min(keel::max(mu, 1e-12), 1.0 - 1e-12);
            return 1.0 / (m * (1.0 - m));
        }
        case GlmFamily::kPoisson:
            return 1.0 / keel::max(mu, 1e-12);
    }
    return 1.0;
}

inline f64 var_glm(f64 mu, GlmFamily fam) noexcept {
    switch (fam) {
        case GlmFamily::kNormal:
            return 1.0;
        case GlmFamily::kBinomial: {
            const f64 m = keel::min(keel::max(mu, 1e-12), 1.0 - 1e-12);
            return m * (1.0 - m);
        }
        case GlmFamily::kPoisson:
            return keel::max(mu, 1e-12);
    }
    return 1.0;
}

inline f64 init_mu(f64 yi, GlmFamily fam) noexcept {
    switch (fam) {
        case GlmFamily::kNormal:
            return yi;
        case GlmFamily::kBinomial:
            return 0.5 * (clamp(yi, 0.0, 1.0) + 0.5);
        case GlmFamily::kPoisson:
            return keel::max(yi + 0.1, 0.1);
    }
    return yi;
}

} // namespace

StatsStatus lasso(ArrayRef<f64> x_col_major, usize n, usize p, ArrayRef<f64> y,
                  ArrayRef<f64> lambdas, f64 alpha, MutArrayRef<f64> out_b_col_major,
                  MutArrayRef<f64> out_intercept, Allocator* alloc) {
    const usize nl = lambdas.size();
    if (!(alpha >= 0.0 && alpha <= 1.0) || n == 0u || nl == 0u || mul_overflows(n, p) ||
        mul_overflows(p, nl) || x_col_major.size() < n * p || y.size() != n ||
        out_b_col_major.size() < p * nl || out_intercept.size() < nl) {
        return StatsStatus::kInvalidArgument;
    }
    for (usize i = 0; i < nl; ++i) {
        if (!(lambdas[i] >= 0.0) || !is_finite(lambdas[i])) return StatsStatus::kInvalidArgument;
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> xs(n * p, 0.0, scratch.allocator());
    for (usize i = 0; i < n * p; ++i) xs[i] = x_col_major[i];
    Array<f64> mu_x(p, 0.0, scratch.allocator());
    Array<f64> sd_x(p, 1.0, scratch.allocator());
    standardise_cols(xs.as_mut_ref(), n, p, mu_x.as_mut_ref(), sd_x.as_mut_ref());

    Array<f64> yc(n, 0.0, scratch.allocator());
    f64 y_mean = 0.0;
    for (usize i = 0; i < n; ++i) y_mean += y[i];
    y_mean /= as<f64>(n);
    for (usize i = 0; i < n; ++i) yc[i] = y[i] - y_mean;

    Array<f64> xj_norm_sq(p, 0.0, scratch.allocator());
    for (usize j = 0; j < p; ++j) {
        f64 s = 0.0;
        for (usize i = 0; i < n; ++i) s += xs[j * n + i] * xs[j * n + i];
        xj_norm_sq[j] = s;
    }

    Array<usize> idx(nl, 0u, scratch.allocator());
    for (usize i = 0; i < nl; ++i) idx[i] = i;
    keel::sort(idx.as_mut_ref(), [&](usize a, usize b) noexcept { return lambdas[a] > lambdas[b]; });

    Array<f64> beta(p, 0.0, scratch.allocator());
    Array<f64> r = Array<f64>::from(yc.as_ref(), scratch.allocator());
    constexpr i32 kMaxIter = 1000;
    constexpr f64 kTol = 1e-7;

    for (usize ii = 0; ii < nl; ++ii) {
        const usize k = idx[ii];
        const f64 lambda = lambdas[k];
        const f64 lambda_eff = alpha * lambda;
        const f64 lambda_l2 = (1.0 - alpha) * lambda;
        for (i32 it = 0; it < kMaxIter; ++it) {
            if (coord_descent_sweep(xs.as_ref(), n, p, r.as_mut_ref(), beta.as_mut_ref(),
                                    xj_norm_sq.as_ref(), lambda_eff, lambda_l2, kTol)) {
                break;
            }
        }
        f64 interc = y_mean;
        for (usize j = 0; j < p; ++j) {
            const f64 b_orig = beta[j] / sd_x[j];
            out_b_col_major[k * p + j] = b_orig;
            interc -= b_orig * mu_x[j];
        }
        out_intercept[k] = interc;
    }
    return StatsStatus::kOk;
}

StatsStatus lassoglm(ArrayRef<f64> x_col_major, usize n, usize p, ArrayRef<f64> y, GlmFamily family,
                     ArrayRef<f64> lambdas, f64 alpha, MutArrayRef<f64> out_b_col_major,
                     MutArrayRef<f64> out_intercept, Allocator* alloc) {
    const usize nl = lambdas.size();
    if (!(alpha >= 0.0 && alpha <= 1.0) || n == 0u || nl == 0u || mul_overflows(n, p) ||
        mul_overflows(p, nl) || x_col_major.size() < n * p || y.size() != n ||
        out_b_col_major.size() < p * nl || out_intercept.size() < nl) {
        return StatsStatus::kInvalidArgument;
    }
    for (usize i = 0; i < nl; ++i) {
        if (!(lambdas[i] >= 0.0) || !is_finite(lambdas[i])) return StatsStatus::kInvalidArgument;
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> xs(n * p, 0.0, scratch.allocator());
    for (usize i = 0; i < n * p; ++i) xs[i] = x_col_major[i];
    Array<f64> mu_x(p, 0.0, scratch.allocator());
    Array<f64> sd_x(p, 1.0, scratch.allocator());
    standardise_cols(xs.as_mut_ref(), n, p, mu_x.as_mut_ref(), sd_x.as_mut_ref());

    Array<usize> idx(nl, 0u, scratch.allocator());
    for (usize i = 0; i < nl; ++i) idx[i] = i;
    keel::sort(idx.as_mut_ref(), [&](usize a, usize b) noexcept { return lambdas[a] > lambdas[b]; });

    Array<f64> beta(p, 0.0, scratch.allocator());
    Array<f64> mu(n, 0.0, scratch.allocator());
    Array<f64> eta(n, 0.0, scratch.allocator());
    for (usize i = 0; i < n; ++i) {
        mu[i] = init_mu(y[i], family);
        switch (family) {
            case GlmFamily::kNormal:
                eta[i] = mu[i];
                break;
            case GlmFamily::kBinomial: {
                const f64 m = keel::min(keel::max(mu[i], 1e-12), 1.0 - 1e-12);
                eta[i] = keel::log(m / (1.0 - m));
                break;
            }
            case GlmFamily::kPoisson:
                eta[i] = keel::log(keel::max(mu[i], 1e-12));
                break;
        }
    }

    Array<f64> z(n, 0.0, scratch.allocator());
    Array<f64> w(n, 0.0, scratch.allocator());
    Array<f64> zc(n, 0.0, scratch.allocator());
    Array<f64> xbar_w(p, 0.0, scratch.allocator());
    Array<f64> xw(n * p, 0.0, scratch.allocator());
    Array<f64> zw(n, 0.0, scratch.allocator());
    Array<f64> xj_norm(p, 0.0, scratch.allocator());
    Array<f64> r(n, 0.0, scratch.allocator());

    f64 b0 = 0.0;
    for (usize ii = 0; ii < nl; ++ii) {
        const usize k = idx[ii];
        const f64 lambda = lambdas[k];
        const f64 lambda_eff = alpha * lambda;
        const f64 lambda_l2 = (1.0 - alpha) * lambda;

        constexpr i32 kOuterIter = 25;
        for (i32 outer = 0; outer < kOuterIter; ++outer) {
            f64 wsum = 0.0;
            f64 wz = 0.0;
            for (usize i = 0; i < n; ++i) {
                const f64 de = d_eta(mu[i], family);
                const f64 v = var_glm(mu[i], family);
                w[i] = 1.0 / (v * de * de + 1e-30);
                z[i] = eta[i] + (y[i] - mu[i]) * de;
                wsum += w[i];
                wz += w[i] * z[i];
            }
            const f64 inv_wsum = 1.0 / keel::max(wsum, 1e-30);
            const f64 z_mean = wz * inv_wsum;
            for (usize i = 0; i < n; ++i) zc[i] = z[i] - z_mean;
            for (usize j = 0; j < p; ++j) {
                f64 wx = 0.0;
                for (usize i = 0; i < n; ++i) wx += w[i] * xs[j * n + i];
                xbar_w[j] = wx * inv_wsum;
            }

            for (usize i = 0; i < n; ++i) {
                const f64 sw = keel::sqrt(w[i]);
                zw[i] = sw * zc[i];
                for (usize j = 0; j < p; ++j) xw[j * n + i] = sw * (xs[j * n + i] - xbar_w[j]);
            }
            for (usize j = 0; j < p; ++j) {
                f64 s = 0.0;
                for (usize i = 0; i < n; ++i) s += xw[j * n + i] * xw[j * n + i];
                xj_norm[j] = s;
            }
            for (usize i = 0; i < n; ++i) {
                f64 ri = zw[i];
                for (usize j = 0; j < p; ++j) ri -= xw[j * n + i] * beta[j];
                r[i] = ri;
            }

            constexpr i32 kInnerIter = 500;
            constexpr f64 kTol = 1e-6;
            for (i32 inner = 0; inner < kInnerIter; ++inner) {
                if (coord_descent_sweep(xw.as_ref(), n, p, r.as_mut_ref(), beta.as_mut_ref(),
                                        xj_norm.as_ref(), lambda_eff, lambda_l2, kTol)) {
                    break;
                }
            }

            b0 = z_mean;
            for (usize j = 0; j < p; ++j) b0 -= beta[j] * xbar_w[j];
            for (usize i = 0; i < n; ++i) {
                f64 e = b0;
                for (usize j = 0; j < p; ++j) e += beta[j] * xs[j * n + i];
                eta[i] = e;
                mu[i] = inv_link(e, family);
            }
        }

        f64 interc = b0;
        for (usize j = 0; j < p; ++j) {
            const f64 b_orig = beta[j] / sd_x[j];
            out_b_col_major[k * p + j] = b_orig;
            interc -= b_orig * mu_x[j];
        }
        out_intercept[k] = interc;
    }
    return StatsStatus::kOk;
}

} // namespace keel::stats
