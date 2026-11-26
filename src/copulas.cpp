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

inline f64 qnan() noexcept { return Limits<f64>::quiet_nan(); }

inline bool mul_overflows(usize a, usize b) noexcept {
    return a != 0u && b > Limits<usize>::kMax / a;
}

// 16-point Gauss-Legendre positive nodes and weights on [-1, 1].
constexpr f64 kGl16X[8] = {
    0.0950125098376374401853, 0.2816035507792589132305, 0.4580167776572273863424,
    0.6178762444026437484467, 0.7554044083550030338951, 0.8656312023878317438805,
    0.9445750230732325760780, 0.9894009349916499325961};
constexpr f64 kGl16W[8] = {
    0.1894506104550684962854, 0.1826034150449235888668, 0.1691565193950025381893,
    0.1495959888165767320815, 0.1246289712555338720525, 0.0951585116824927848099,
    0.0622535239386478928628, 0.0271524594117540948518};

bool chol_lower_inplace(f64* mat, usize d, f64& log_half_det_out) noexcept {
    f64 log_half_det = 0.0;
    for (usize j = 0; j < d; ++j) {
        f64 diag = mat[j * d + j];
        for (usize k = 0; k < j; ++k) {
            diag -= mat[j * d + k] * mat[j * d + k];
        }
        if (!(diag > 0.0) || !is_finite(diag)) {
            return false;
        }
        const f64 ljj = sqrt(diag);
        mat[j * d + j] = ljj;
        log_half_det += log(ljj);
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
    log_half_det_out = log_half_det;
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

inline f64 std_phi(f64 x) noexcept {
    if (is_inf(x)) {
        return (x > 0.0) ? 1.0 : 0.0;
    }
    return 0.5 * specfun::erfc(-x * kInvSqrt2);
}

} // namespace

f64 bivnorm_cdf(f64 a, f64 b, f64 rho) noexcept {
    if (is_nan(a) || is_nan(b) || !(rho >= -1.0 && rho <= 1.0)) {
        return qnan();
    }
    if (is_inf(a)) {
        return (a < 0.0) ? 0.0 : std_phi(b);
    }
    if (is_inf(b)) {
        return (b < 0.0) ? 0.0 : std_phi(a);
    }
    if (rho == 0.0) {
        return std_phi(a) * std_phi(b);
    }
    if (rho == 1.0) {
        return std_phi(min(a, b));
    }
    if (rho == -1.0) {
        return max(0.0, std_phi(a) - std_phi(-b));
    }

    // Drezner-Wesolowsky / Plackett representation with singularity-free substitution r = sin(t):
    //   Phi_2(a, b; rho) = Phi(a)*Phi(b) + (1 / (2*pi)) * integral_0^{asin(rho)}
    //                      exp(-(a^2 + b^2 - 2*a*b*sin(t)) / (2*cos^2(t))) dt.
    // For |rho| near 1, integrate from asin(rho) to +/-pi/2 using the exact limit at |rho| = 1.
    const f64 theta_rho = asin(rho);
    auto integrand = [&](f64 t) noexcept -> f64 {
        const f64 st = sin(t);
        const f64 ct = cos(t);
        const f64 ct2 = ct * ct;
        if (!(ct2 > 0.0)) {
            return 0.0;
        }
        const f64 num = (a - b * st) * (a - b * st) + b * b * ct2;
        const f64 exponent = -0.5 * num / ct2;
        return (exponent < -745.0) ? 0.0 : exp(exponent);
    };
    auto integrate_gl = [&](f64 lo, f64 hi, i32 panels) noexcept -> f64 {
        const f64 step = (hi - lo) / as<f64>(panels);
        f64 sum = 0.0;
        for (i32 p = 0; p < panels; ++p) {
            const f64 x0 = lo + as<f64>(p) * step;
            const f64 x1 = x0 + step;
            const f64 mid = 0.5 * (x0 + x1);
            const f64 half = 0.5 * (x1 - x0);
            f64 s = 0.0;
            for (i32 i = 0; i < 8; ++i) {
                const f64 dx = half * kGl16X[i];
                s += kGl16W[i] * (integrand(mid - dx) + integrand(mid + dx));
            }
            sum += s * half;
        }
        return sum;
    };

    const f64 inv_2pi = 1.0 / (2.0 * kPi);
    if (abs(rho) <= 0.925) {
        const f64 val = std_phi(a) * std_phi(b) + inv_2pi * integrate_gl(0.0, theta_rho, 4);
        return clamp(val, 0.0, 1.0);
    }
    if (rho > 0.0) {
        const f64 base = std_phi(min(a, b));
        const f64 tail = inv_2pi * integrate_gl(theta_rho, 0.5 * kPi, 8);
        return clamp(base - tail, 0.0, 1.0);
    }
    const f64 base = max(0.0, std_phi(a) - std_phi(-b));
    const f64 tail = inv_2pi * integrate_gl(-0.5 * kPi, theta_rho, 8);
    return clamp(base + tail, 0.0, 1.0);
}

StatsStatus copula_pdf_gaussian(ArrayRef<f64> u_col_major, usize n, usize d,
                                ArrayRef<f64> r_col_major, MutArrayRef<f64> out, Allocator* alloc) {
    if (d < 2 || mul_overflows(n, d) || mul_overflows(d, d) || u_col_major.size() < n * d ||
        r_col_major.size() < d * d || out.size() < n) {
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
    f64 log_half_det = 0.0;
    if (!chol_lower_inplace(l_mat.data(), d, log_half_det)) {
        return StatsStatus::kSingularMatrix;
    }
    Array<f64> z(d, 0.0, scratch.allocator());
    Array<f64> y(d, 0.0, scratch.allocator());

    for (usize i = 0; i < n; ++i) {
        bool has_nan = false;
        bool outside = false;
        for (usize j = 0; j < d; ++j) {
            const f64 uij = u_col_major[j * n + i];
            if (is_nan(uij)) {
                has_nan = true;
                break;
            }
            if (uij <= 0.0 || uij >= 1.0) {
                outside = true;
            }
        }
        if (has_nan) {
            out[i] = qnan();
            continue;
        }
        if (outside) {
            out[i] = 0.0;
            continue;
        }
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
        out[i] = exp(-log_half_det - 0.5 * (yty - ztz));
    }
    return StatsStatus::kOk;
}

StatsStatus copula_cdf_gaussian_2d(ArrayRef<f64> u_col_major, usize n, f64 rho,
                                   MutArrayRef<f64> out) noexcept {
    if (mul_overflows(n, 2u) || u_col_major.size() < n * 2 || out.size() < n ||
        !(rho >= -1.0 && rho <= 1.0)) {
        return StatsStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n; ++i) {
        const f64 u1 = u_col_major[i];
        const f64 u2 = u_col_major[n + i];
        if (is_nan(u1) || is_nan(u2)) {
            out[i] = qnan();
            continue;
        }
        if (u1 <= 0.0 || u2 <= 0.0) {
            out[i] = 0.0;
            continue;
        }
        if (u1 >= 1.0) {
            out[i] = min(1.0, u2);
            continue;
        }
        if (u2 >= 1.0) {
            out[i] = min(1.0, u1);
            continue;
        }
        const f64 z1 = norm_inv(u1, 0.0, 1.0);
        const f64 z2 = norm_inv(u2, 0.0, 1.0);
        out[i] = bivnorm_cdf(z1, z2, rho);
    }
    return StatsStatus::kOk;
}

StatsStatus copula_pdf_t(ArrayRef<f64> u_col_major, usize n, usize d, ArrayRef<f64> r_col_major,
                         f64 nu, MutArrayRef<f64> out, Allocator* alloc) {
    if (d < 2 || !(nu > 0.0) || mul_overflows(n, d) || mul_overflows(d, d) ||
        u_col_major.size() < n * d || r_col_major.size() < d * d || out.size() < n) {
        return StatsStatus::kInvalidArgument;
    }
    if (is_inf(nu)) {
        return copula_pdf_gaussian(u_col_major, n, d, r_col_major, out, alloc);
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
    f64 log_half_det = 0.0;
    if (!chol_lower_inplace(l_mat.data(), d, log_half_det)) {
        return StatsStatus::kSingularMatrix;
    }

    const f64 dd = as<f64>(d);
    const f64 log_norm_mv = specfun::gammaln(0.5 * (nu + dd)) - specfun::gammaln(0.5 * nu) -
                            0.5 * dd * log(nu * kPi) - log_half_det;
    const f64 log_norm_uni =
        specfun::gammaln(0.5 * (nu + 1.0)) - specfun::gammaln(0.5 * nu) - 0.5 * log(nu * kPi);

    Array<f64> z(d, 0.0, scratch.allocator());
    Array<f64> y(d, 0.0, scratch.allocator());
    for (usize i = 0; i < n; ++i) {
        bool has_nan = false;
        bool outside = false;
        for (usize j = 0; j < d; ++j) {
            const f64 uij = u_col_major[j * n + i];
            if (is_nan(uij)) {
                has_nan = true;
                break;
            }
            if (uij <= 0.0 || uij >= 1.0) {
                outside = true;
            }
        }
        if (has_nan) {
            out[i] = qnan();
            continue;
        }
        if (outside) {
            out[i] = 0.0;
            continue;
        }
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
    if (!(alpha > 0.0) || !is_finite(alpha) || mul_overflows(n, 2u) || u_col_major.size() < n * 2 ||
        out.size() < n) {
        return StatsStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n; ++i) {
        const f64 u = u_col_major[i];
        const f64 v = u_col_major[n + i];
        if (is_nan(u) || is_nan(v)) {
            out[i] = qnan();
            continue;
        }
        if (u <= 0.0 || v <= 0.0 || u > 1.0 || v > 1.0) {
            out[i] = 0.0;
            continue;
        }
        const f64 lu = log(u);
        const f64 lv = log(v);
        const f64 a_lu = -alpha * lu;
        const f64 a_lv = -alpha * lv;
        const f64 m = max(a_lu, a_lv);
        // log(u^(-alpha) + v^(-alpha) - 1) = m + log(exp(a_lu - m) + exp(a_lv - m) - exp(-m))
        const f64 inner = exp(a_lu - m) + exp(a_lv - m) - exp(-m);
        if (!(inner > 0.0)) {
            out[i] = 0.0;
            continue;
        }
        const f64 log_t = m + log(inner);
        const f64 log_pdf =
            log1p(alpha) - (1.0 + alpha) * (lu + lv) - (2.0 + 1.0 / alpha) * log_t;
        out[i] = exp(log_pdf);
    }
    return StatsStatus::kOk;
}

StatsStatus copula_cdf_clayton(ArrayRef<f64> u_col_major, usize n, f64 alpha,
                               MutArrayRef<f64> out) noexcept {
    if (!(alpha > 0.0) || !is_finite(alpha) || mul_overflows(n, 2u) || u_col_major.size() < n * 2 ||
        out.size() < n) {
        return StatsStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n; ++i) {
        const f64 u = u_col_major[i];
        const f64 v = u_col_major[n + i];
        if (is_nan(u) || is_nan(v)) {
            out[i] = qnan();
            continue;
        }
        if (u <= 0.0 || v <= 0.0) {
            out[i] = 0.0;
            continue;
        }
        if (u >= 1.0) {
            out[i] = min(1.0, v);
            continue;
        }
        if (v >= 1.0) {
            out[i] = min(1.0, u);
            continue;
        }
        const f64 a_lu = -alpha * log(u);
        const f64 a_lv = -alpha * log(v);
        const f64 m = max(a_lu, a_lv);
        const f64 inner = exp(a_lu - m) + exp(a_lv - m) - exp(-m);
        if (!(inner > 0.0)) {
            out[i] = 0.0;
            continue;
        }
        out[i] = exp(-(m + log(inner)) / alpha);
    }
    return StatsStatus::kOk;
}

StatsStatus copula_pdf_frank(ArrayRef<f64> u_col_major, usize n, f64 alpha,
                             MutArrayRef<f64> out) noexcept {
    if (!is_finite(alpha) || mul_overflows(n, 2u) || u_col_major.size() < n * 2 || out.size() < n) {
        return StatsStatus::kInvalidArgument;
    }
    const f64 em_a_m1 = expm1(-alpha); // exp(-alpha) - 1
    for (usize i = 0; i < n; ++i) {
        const f64 u = u_col_major[i];
        const f64 v = u_col_major[n + i];
        if (is_nan(u) || is_nan(v)) {
            out[i] = qnan();
            continue;
        }
        if (u < 0.0 || v < 0.0 || u > 1.0 || v > 1.0) {
            out[i] = 0.0;
            continue;
        }
        if (abs(alpha) < 1e-8) {
            // Smooth Taylor limit around independence (alpha == 0).
            out[i] = 1.0 + alpha * (1.0 - 2.0 * u) * (1.0 - 2.0 * v);
            continue;
        }
        const f64 em_au_m1 = expm1(-alpha * u);
        const f64 em_av_m1 = expm1(-alpha * v);
        const f64 denom = -em_a_m1 - em_au_m1 * em_av_m1;
        out[i] = -alpha * em_a_m1 * exp(-alpha * (u + v)) / (denom * denom);
    }
    return StatsStatus::kOk;
}

StatsStatus copula_cdf_frank(ArrayRef<f64> u_col_major, usize n, f64 alpha,
                             MutArrayRef<f64> out) noexcept {
    if (!is_finite(alpha) || mul_overflows(n, 2u) || u_col_major.size() < n * 2 || out.size() < n) {
        return StatsStatus::kInvalidArgument;
    }
    const f64 em_a_m1 = expm1(-alpha);
    for (usize i = 0; i < n; ++i) {
        const f64 u = u_col_major[i];
        const f64 v = u_col_major[n + i];
        if (is_nan(u) || is_nan(v)) {
            out[i] = qnan();
            continue;
        }
        if (u <= 0.0 || v <= 0.0) {
            out[i] = 0.0;
            continue;
        }
        if (u >= 1.0) {
            out[i] = min(1.0, v);
            continue;
        }
        if (v >= 1.0) {
            out[i] = min(1.0, u);
            continue;
        }
        if (abs(alpha) < 1e-8) {
            out[i] = u * v * (1.0 + 0.5 * alpha * (1.0 - u) * (1.0 - v));
            continue;
        }
        const f64 ratio = (expm1(-alpha * u) * expm1(-alpha * v)) / em_a_m1;
        out[i] = -log1p(ratio) / alpha;
    }
    return StatsStatus::kOk;
}

StatsStatus copula_pdf_gumbel(ArrayRef<f64> u_col_major, usize n, f64 alpha,
                              MutArrayRef<f64> out) noexcept {
    if (!(alpha >= 1.0) || !is_finite(alpha) || mul_overflows(n, 2u) ||
        u_col_major.size() < n * 2 || out.size() < n) {
        return StatsStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n; ++i) {
        const f64 u = u_col_major[i];
        const f64 v = u_col_major[n + i];
        if (is_nan(u) || is_nan(v)) {
            out[i] = qnan();
            continue;
        }
        if (u < 0.0 || v < 0.0 || u > 1.0 || v > 1.0) {
            out[i] = 0.0;
            continue;
        }
        if (alpha == 1.0) {
            out[i] = 1.0;
            continue;
        }
        if (u <= 0.0 || v <= 0.0 || u >= 1.0 || v >= 1.0) {
            out[i] = 0.0;
            continue;
        }
        const f64 lu = -log(u);
        const f64 lv = -log(v);
        // Scale by m = max(lu, lv) so pow(lu, alpha) never overflows for large alpha.
        const f64 m = max(lu, lv);
        const f64 ru = lu / m;
        const f64 rv = lv / m;
        const f64 s = pow(ru, alpha) + pow(rv, alpha);
        const f64 log_a = alpha * log(m) + log(s);
        const f64 a_inv_a = m * pow(s, 1.0 / alpha);
        const f64 log_pdf = -a_inv_a - log(u) - log(v) + (alpha - 1.0) * (log(lu) + log(lv)) +
                            (1.0 / alpha - 2.0) * log_a + log((alpha - 1.0) + a_inv_a);
        out[i] = exp(log_pdf);
    }
    return StatsStatus::kOk;
}

StatsStatus copula_cdf_gumbel(ArrayRef<f64> u_col_major, usize n, f64 alpha,
                              MutArrayRef<f64> out) noexcept {
    if (!(alpha >= 1.0) || !is_finite(alpha) || mul_overflows(n, 2u) ||
        u_col_major.size() < n * 2 || out.size() < n) {
        return StatsStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n; ++i) {
        const f64 u = u_col_major[i];
        const f64 v = u_col_major[n + i];
        if (is_nan(u) || is_nan(v)) {
            out[i] = qnan();
            continue;
        }
        if (u <= 0.0 || v <= 0.0) {
            out[i] = 0.0;
            continue;
        }
        if (u >= 1.0) {
            out[i] = min(1.0, v);
            continue;
        }
        if (v >= 1.0) {
            out[i] = min(1.0, u);
            continue;
        }
        if (alpha == 1.0) {
            out[i] = u * v;
            continue;
        }
        const f64 lu = -log(u);
        const f64 lv = -log(v);
        const f64 m = max(lu, lv);
        const f64 s = pow(lu / m, alpha) + pow(lv / m, alpha);
        out[i] = exp(-m * pow(s, 1.0 / alpha));
    }
    return StatsStatus::kOk;
}

} // namespace keel::stats
