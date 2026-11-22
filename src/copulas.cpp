// keelcpp/stats/src/copulas.cpp
// Pure zero-STL copula PDF and CDF implementations (Gaussian, Student-t, Clayton, Frank, Gumbel).

#include "keel/stats.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"
#include "keel/specfun.h"

namespace keel::stats {

namespace {

constexpr f64 kPi = 3.14159265358979323846;

bool chol_lower_inplace(f64* mat, usize d, f64& det_out) noexcept {
    f64 det = 1.0;
    for (usize j = 0; j < d; ++j) {
        f64 diag = mat[j * d + j];
        for (usize k = 0; k < j; ++k) {
            diag -= mat[j * d + k] * mat[j * d + k];
        }
        if (!(diag > 0.0)) {
            return false;
        }
        const f64 ljj = sqrt(diag);
        mat[j * d + j] = ljj;
        det *= diag;
        for (usize i = j + 1; i < d; ++i) {
            f64 s = mat[i * d + j];
            for (usize k = 0; k < j; ++k) {
                s -= mat[i * d + k] * mat[j * d + k];
            }
            mat[i * d + j] = s / ljj;
        }
        for (usize k = j + 1; k < d; ++k) {
            mat[j * d + k] = 0.0;
        }
    }
    det_out = det;
    return true;
}

void chol_lower_solve(const f64* l_mat, f64* y, usize d) noexcept {
    for (usize i = 0; i < d; ++i) {
        f64 s = y[i];
        for (usize k = 0; k < i; ++k) {
            s -= l_mat[i * d + k] * y[k];
        }
        y[i] = s / l_mat[i * d + i];
    }
}

} // namespace

f64 bivnorm_cdf(f64 a, f64 b, f64 rho) noexcept {
    const f64 inv_sqrt2 = 1.0 / sqrt(2.0);
    const f64 phi_a = 0.5 * (1.0 + specfun::erf(a * inv_sqrt2));
    const f64 phi_b = 0.5 * (1.0 + specfun::erf(b * inv_sqrt2));
    if (rho == 0.0) {
        return phi_a * phi_b;
    }

    constexpr i32 kPanels = 8192;
    const f64 half_inv_pi = 1.0 / (2.0 * kPi);
    const f64 a2_p_b2 = a * a + b * b;
    const f64 ab = a * b;
    f64 total = 0.0;
    for (i32 i = 0; i <= kPanels; ++i) {
        const f64 r = rho * as<f64>(i) / as<f64>(kPanels);
        const f64 one_m_r2 = 1.0 - r * r;
        const f64 v = half_inv_pi / sqrt(one_m_r2) * exp(-(a2_p_b2 - 2.0 * r * ab) / (2.0 * one_m_r2));
        if (i == 0 || i == kPanels) {
            total += 0.5 * v;
        } else {
            total += v;
        }
    }
    total *= rho / as<f64>(kPanels);
    return phi_a * phi_b + total;
}

StatsStatus copula_pdf_gaussian(ArrayRef<f64> u_col_major, usize n, usize d,
                                ArrayRef<f64> r_col_major, MutArrayRef<f64> out, Allocator* alloc) {
    if (d < 2 || u_col_major.size() < n * d || r_col_major.size() < d * d || out.size() < n) {
        return StatsStatus::kInvalidArgument;
    }
    if (n == 0) {
        return StatsStatus::kOk;
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> l_mat(d * d, 0.0, scratch.allocator());
    for (usize i = 0; i < d; ++i) {
        for (usize j = 0; j < d; ++j) {
            l_mat[i * d + j] = r_col_major[j * d + i];
        }
    }
    f64 det_r = 0.0;
    if (!chol_lower_inplace(l_mat.data(), d, det_r)) {
        return StatsStatus::kSingularMatrix;
    }
    const f64 pref = 1.0 / sqrt(det_r);
    Array<f64> z(d, 0.0, scratch.allocator());
    Array<f64> y(d, 0.0, scratch.allocator());

    for (usize i = 0; i < n; ++i) {
        for (usize j = 0; j < d; ++j) {
            z[j] = norm_inv(u_col_major[j * n + i], 0.0, 1.0);
            y[j] = z[j];
        }
        chol_lower_solve(l_mat.data(), y.data(), d);
        f64 yty = 0.0;
        f64 ztz = 0.0;
        for (usize j = 0; j < d; ++j) {
            yty += y[j] * y[j];
            ztz += z[j] * z[j];
        }
        out[i] = pref * exp(-0.5 * (yty - ztz));
    }
    return StatsStatus::kOk;
}

StatsStatus copula_cdf_gaussian_2d(ArrayRef<f64> u_col_major, usize n, f64 rho,
                                   MutArrayRef<f64> out) noexcept {
    if (u_col_major.size() < n * 2 || out.size() < n || !(rho > -1.0 && rho < 1.0)) {
        return StatsStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n; ++i) {
        const f64 z1 = norm_inv(u_col_major[i], 0.0, 1.0);
        const f64 z2 = norm_inv(u_col_major[n + i], 0.0, 1.0);
        out[i] = bivnorm_cdf(z1, z2, rho);
    }
    return StatsStatus::kOk;
}

StatsStatus copula_pdf_t(ArrayRef<f64> u_col_major, usize n, usize d, ArrayRef<f64> r_col_major,
                         f64 nu, MutArrayRef<f64> out, Allocator* alloc) {
    if (d < 2 || !(nu > 0.0) || u_col_major.size() < n * d || r_col_major.size() < d * d ||
        out.size() < n) {
        return StatsStatus::kInvalidArgument;
    }
    if (n == 0) {
        return StatsStatus::kOk;
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> l_mat(d * d, 0.0, scratch.allocator());
    for (usize i = 0; i < d; ++i) {
        for (usize j = 0; j < d; ++j) {
            l_mat[i * d + j] = r_col_major[j * d + i];
        }
    }
    f64 det_r = 0.0;
    if (!chol_lower_inplace(l_mat.data(), d, det_r)) {
        return StatsStatus::kSingularMatrix;
    }

    const f64 dd = as<f64>(d);
    const f64 log_norm_mv = specfun::gammaln(0.5 * (nu + dd)) - specfun::gammaln(0.5 * nu) -
                            0.5 * dd * log(nu * kPi) - 0.5 * log(det_r);
    const f64 log_norm_uni =
        specfun::gammaln(0.5 * (nu + 1.0)) - specfun::gammaln(0.5 * nu) - 0.5 * log(nu * kPi);

    Array<f64> z(d, 0.0, scratch.allocator());
    Array<f64> y(d, 0.0, scratch.allocator());
    for (usize i = 0; i < n; ++i) {
        for (usize j = 0; j < d; ++j) {
            z[j] = t_inv(u_col_major[j * n + i], nu);
            y[j] = z[j];
        }
        chol_lower_solve(l_mat.data(), y.data(), d);
        f64 yty = 0.0;
        for (usize j = 0; j < d; ++j) {
            yty += y[j] * y[j];
        }
        const f64 log_mv = log_norm_mv - 0.5 * (nu + dd) * log1p(yty / nu);
        f64 log_unis = 0.0;
        for (usize j = 0; j < d; ++j) {
            log_unis += log_norm_uni - 0.5 * (nu + 1.0) * log1p(z[j] * z[j] / nu);
        }
        out[i] = exp(log_mv - log_unis);
    }
    return StatsStatus::kOk;
}

StatsStatus copula_pdf_clayton(ArrayRef<f64> u_col_major, usize n, f64 alpha,
                               MutArrayRef<f64> out) noexcept {
    if (!(alpha > 0.0) || u_col_major.size() < n * 2 || out.size() < n) {
        return StatsStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n; ++i) {
        const f64 u = u_col_major[i];
        const f64 v = u_col_major[n + i];
        const f64 t = pow(u, -alpha) + pow(v, -alpha) - 1.0;
        out[i] = (1.0 + alpha) * pow(u * v, -(1.0 + alpha)) * pow(t, -(2.0 + 1.0 / alpha));
    }
    return StatsStatus::kOk;
}

StatsStatus copula_cdf_clayton(ArrayRef<f64> u_col_major, usize n, f64 alpha,
                               MutArrayRef<f64> out) noexcept {
    if (!(alpha > 0.0) || u_col_major.size() < n * 2 || out.size() < n) {
        return StatsStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n; ++i) {
        const f64 u = u_col_major[i];
        const f64 v = u_col_major[n + i];
        const f64 t = pow(u, -alpha) + pow(v, -alpha) - 1.0;
        out[i] = pow(t, -1.0 / alpha);
    }
    return StatsStatus::kOk;
}

StatsStatus copula_pdf_frank(ArrayRef<f64> u_col_major, usize n, f64 alpha,
                             MutArrayRef<f64> out) noexcept {
    if (u_col_major.size() < n * 2 || out.size() < n) {
        return StatsStatus::kInvalidArgument;
    }
    const f64 one_m_em_a = 1.0 - exp(-alpha);
    for (usize i = 0; i < n; ++i) {
        const f64 u = u_col_major[i];
        const f64 v = u_col_major[n + i];
        const f64 em_au = exp(-alpha * u);
        const f64 em_av = exp(-alpha * v);
        const f64 denom = one_m_em_a - (1.0 - em_au) * (1.0 - em_av);
        out[i] = alpha * one_m_em_a * em_au * em_av / (denom * denom);
    }
    return StatsStatus::kOk;
}

StatsStatus copula_cdf_frank(ArrayRef<f64> u_col_major, usize n, f64 alpha,
                             MutArrayRef<f64> out) noexcept {
    if (alpha == 0.0 || u_col_major.size() < n * 2 || out.size() < n) {
        return StatsStatus::kInvalidArgument;
    }
    const f64 em_a = exp(-alpha);
    for (usize i = 0; i < n; ++i) {
        const f64 u = u_col_major[i];
        const f64 v = u_col_major[n + i];
        const f64 num = (exp(-alpha * u) - 1.0) * (exp(-alpha * v) - 1.0);
        const f64 inside = 1.0 + num / (em_a - 1.0);
        out[i] = -log(inside) / alpha;
    }
    return StatsStatus::kOk;
}

StatsStatus copula_pdf_gumbel(ArrayRef<f64> u_col_major, usize n, f64 alpha,
                              MutArrayRef<f64> out) noexcept {
    if (!(alpha >= 1.0) || u_col_major.size() < n * 2 || out.size() < n) {
        return StatsStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n; ++i) {
        const f64 u = u_col_major[i];
        const f64 v = u_col_major[n + i];
        const f64 lu = -log(u);
        const f64 lv = -log(v);
        const f64 a_val = pow(lu, alpha) + pow(lv, alpha);
        const f64 a_inv_a = pow(a_val, 1.0 / alpha);
        const f64 c_val = exp(-a_inv_a);
        out[i] = c_val / (u * v) * pow(lu * lv, alpha - 1.0) * pow(a_val, 1.0 / alpha - 2.0) *
                 ((alpha - 1.0) + a_inv_a);
    }
    return StatsStatus::kOk;
}

StatsStatus copula_cdf_gumbel(ArrayRef<f64> u_col_major, usize n, f64 alpha,
                              MutArrayRef<f64> out) noexcept {
    if (!(alpha >= 1.0) || u_col_major.size() < n * 2 || out.size() < n) {
        return StatsStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n; ++i) {
        const f64 u = u_col_major[i];
        const f64 v = u_col_major[n + i];
        const f64 lu = -log(u);
        const f64 lv = -log(v);
        const f64 a_val = pow(lu, alpha) + pow(lv, alpha);
        out[i] = exp(-pow(a_val, 1.0 / alpha));
    }
    return StatsStatus::kOk;
}

} // namespace keel::stats
