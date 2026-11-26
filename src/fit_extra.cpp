// keelcpp/stats/src/fit_extra.cpp
// Pure zero-STL closed-form/censored MLE fitters & negative log-likelihoods:
//   normfit, expfit, poissfit, unifit, binofit_scalar, raylfit,
//   normlike, explike, lognlike, gamlike, betalike, wbllike, evlike, gevlike, gplike.

#include "keel/stats.h"

#include "keel/math.h"
#include "keel/specfun.h"

namespace keel::stats {

namespace {

constexpr f64 kLog2Pi = 1.8378770664093454835606594728112352;
constexpr f64 kSqrt2OverPi = 0.7978845608028653558798921198687637;

inline f64 qnan() noexcept { return Limits<f64>::quiet_nan(); }
inline f64 pinf() noexcept { return Limits<f64>::infinity(); }

// Inverse Mills ratio lambda(a) = phi(a) / Phi(-a) for right-censored normal MLE.
// Uses erfcx(a / sqrt(2)) for a >= 0 so extreme right tails never underflow to 0/0.
inline f64 inv_mills(f64 a) noexcept {
    if (a >= 0.0) {
        const f64 ex = specfun::erfcx(a * kInvSqrt2);
        return (ex > 0.0) ? (kSqrt2OverPi / ex) : (a + 1.0 / a);
    }
    const f64 pdf = kInvSqrt2Pi * exp(-0.5 * a * a);
    const f64 surv = 0.5 * specfun::erfc(a * kInvSqrt2);
    return (surv > 0.0) ? (pdf / surv) : 0.0;
}

// -log(Phi(-z)) = -log(0.5 * erfc(z / sqrt(2))) without underflow in the right tail.
inline f64 neg_log_norm_surv(f64 z) noexcept {
    if (z <= 0.0) {
        return -log(0.5 * specfun::erfc(z * kInvSqrt2));
    }
    const f64 ex = specfun::erfcx(z * kInvSqrt2);
    return 0.5 * z * z - log(0.5 * ex);
}

inline f64 z_norm_crit(f64 alpha) noexcept {
    return -norm_inv(0.5 * alpha, 0.0, 1.0);
}

} // namespace

// ============================================================================
// Distribution MLE Fitters
// ============================================================================

StatsStatus normfit(ArrayRef<f64> x, ArrayRef<u8> cens, ArrayRef<f64> freq, f64 alpha,
                    Fit2Result& out) noexcept {
    const f64 nanv = qnan();
    out = {nanv, nanv, nanv, nanv, nanv, nanv};
    const usize n = x.size();
    if (n == 0 || !(alpha > 0.0 && alpha < 1.0)) {
        return StatsStatus::kInvalidArgument;
    }
    if (!cens.empty() && cens.size() != n) {
        return StatsStatus::kInvalidArgument;
    }
    if (!freq.empty() && freq.size() != n) {
        return StatsStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n; ++i) {
        if (!is_finite(x[i])) {
            return StatsStatus::kInvalidArgument;
        }
        if (!freq.empty() && (!(freq[i] >= 0.0) || !is_finite(freq[i]))) {
            return StatsStatus::kInvalidArgument;
        }
    }
    if (n == 1) {
        out.p1 = x[0];
        out.p2 = 0.0;
        return StatsStatus::kOk;
    }
    const bool has_c = !cens.empty();
    const bool has_f = !freq.empty();

    f64 sumf = 0.0;
    usize nuncens = 0;
    for (usize i = 0; i < n; ++i) {
        const f64 fi = has_f ? freq[i] : 1.0;
        sumf += fi;
        if (!(has_c && cens[i] != 0) && fi > 0.0) {
            ++nuncens;
        }
    }
    if (sumf < 2.0 || nuncens < 1) {
        return StatsStatus::kInvalidArgument;
    }

    bool any_cens = false;
    if (has_c) {
        for (usize i = 0; i < n; ++i) {
            const f64 fi = has_f ? freq[i] : 1.0;
            if (cens[i] != 0 && fi > 0.0) {
                any_cens = true;
                break;
            }
        }
    }
    if (!any_cens) {
        f64 s = 0.0;
        for (usize i = 0; i < n; ++i) {
            const f64 fi = has_f ? freq[i] : 1.0;
            s += fi * x[i];
        }
        const f64 mu = s / sumf;
        f64 sq = 0.0;
        for (usize i = 0; i < n; ++i) {
            const f64 fi = has_f ? freq[i] : 1.0;
            const f64 d = x[i] - mu;
            sq += fi * d * d;
        }
        const f64 dof = sumf - 1.0;
        const f64 sd = sqrt(sq / dof);
        const f64 t = t_inv(1.0 - 0.5 * alpha, dof);
        const f64 sem = sd / sqrt(sumf);
        const f64 chi_u = chi2_inv(1.0 - 0.5 * alpha, dof);
        const f64 chi_l = chi2_inv(0.5 * alpha, dof);
        out.p1 = mu;
        out.p2 = sd;
        out.ci1_lo = mu - t * sem;
        out.ci1_hi = mu + t * sem;
        out.ci2_lo = sqrt(dof * sd * sd / chi_u);
        out.ci2_hi = sqrt(dof * sd * sd / chi_l);
        return StatsStatus::kOk;
    }

    // Censored MLE via EM on truncated-normal moments.
    f64 s_un = 0.0;
    f64 w_un = 0.0;
    for (usize i = 0; i < n; ++i) {
        if (!cens[i]) {
            const f64 fi = has_f ? freq[i] : 1.0;
            s_un += fi * x[i];
            w_un += fi;
        }
    }
    f64 mu = s_un / w_un;
    f64 sq_un = 0.0;
    for (usize i = 0; i < n; ++i) {
        if (!cens[i]) {
            const f64 fi = has_f ? freq[i] : 1.0;
            const f64 d = x[i] - mu;
            sq_un += fi * d * d;
        }
    }
    f64 sd = sqrt(max(sq_un / max(w_un - 1.0, 1.0), 1e-12));

    for (i32 it = 0; it < 200; ++it) {
        f64 sumfy = 0.0;
        f64 sumfy2 = 0.0;
        for (usize i = 0; i < n; ++i) {
            const f64 yi = x[i];
            const f64 fi = has_f ? freq[i] : 1.0;
            if (!cens[i]) {
                sumfy += fi * yi;
                sumfy2 += fi * yi * yi;
            } else {
                const f64 a = (yi - mu) / sd;
                const f64 lam = inv_mills(a);
                const f64 ey = mu + sd * lam;
                const f64 eym2 = sd * sd * (1.0 + a * lam);
                const f64 ey2 = eym2 + 2.0 * mu * sd * lam + mu * mu;
                sumfy += fi * ey;
                sumfy2 += fi * ey2;
            }
        }
        const f64 mu_new = sumfy / sumf;
        const f64 var_new = max(sumfy2 / sumf - mu_new * mu_new, 1e-300);
        const f64 sd_new = sqrt(var_new);
        const f64 delta = abs(mu_new - mu) + abs(sd_new - sd);
        mu = mu_new;
        sd = sd_new;
        if (delta < 1e-10) {
            break;
        }
    }

    // Analytic observed Fisher information at MLE.
    f64 imm = 0.0;
    f64 ims = 0.0;
    f64 iss = 0.0;
    const f64 sd2 = sd * sd;
    const f64 sd3 = sd2 * sd;
    const f64 sd4 = sd2 * sd2;
    for (usize i = 0; i < n; ++i) {
        const f64 fi = has_f ? freq[i] : 1.0;
        const f64 d = x[i] - mu;
        const f64 a = d / sd;
        if (!cens[i]) {
            imm += fi / sd2;
            ims += 2.0 * fi * d / sd3;
            iss += -fi / sd2 + 3.0 * fi * d * d / sd4;
        } else {
            const f64 m = inv_mills(a);
            const f64 mp = m * (m - a);
            imm += fi * mp / sd2;
            ims += fi * (a * mp + m) / sd2;
            iss += fi * a * (a * mp + 2.0 * m) / sd2;
        }
    }
    const f64 det = imm * iss - ims * ims;
    f64 se_mu = nanv;
    f64 se_sigma = nanv;
    if (det > 0.0) {
        se_mu = sqrt(iss / det);
        se_sigma = sqrt(imm / det);
    }
    const f64 z = z_norm_crit(alpha);
    const f64 se_logs = (sd > 0.0) ? (se_sigma / sd) : nanv;

    out.p1 = mu;
    out.p2 = sd;
    out.ci1_lo = mu - z * se_mu;
    out.ci1_hi = mu + z * se_mu;
    out.ci2_lo = sd * exp(-z * se_logs);
    out.ci2_hi = sd * exp(z * se_logs);
    return StatsStatus::kOk;
}

StatsStatus expfit(ArrayRef<f64> x, ArrayRef<f64> cens, ArrayRef<f64> freq, f64 alpha,
                   Fit1Result& out) noexcept {
    const f64 nanv = qnan();
    out = {nanv, nanv, nanv};
    const usize n = x.size();
    if (n == 0 || !(alpha > 0.0 && alpha < 1.0)) {
        return StatsStatus::kInvalidArgument;
    }
    if (!cens.empty() && cens.size() != n) {
        return StatsStatus::kInvalidArgument;
    }
    if (!freq.empty() && freq.size() != n) {
        return StatsStatus::kInvalidArgument;
    }
    const bool has_c = !cens.empty();
    const bool has_f = !freq.empty();
    f64 total_t = 0.0;
    f64 events_d = 0.0;
    for (usize i = 0; i < n; ++i) {
        if (!(x[i] >= 0.0) || !is_finite(x[i])) {
            return StatsStatus::kInvalidArgument;
        }
        const f64 fi = has_f ? freq[i] : 1.0;
        if (!(fi >= 0.0) || !is_finite(fi)) {
            return StatsStatus::kInvalidArgument;
        }
        const f64 ci = (has_c && cens[i] != 0.0) ? 1.0 : 0.0;
        total_t += fi * x[i];
        events_d += fi * (1.0 - ci);
    }
    if (!(events_d > 0.0)) {
        return StatsStatus::kInvalidArgument;
    }
    const f64 mu = total_t / events_d;
    const f64 dof = 2.0 * events_d;
    const f64 chi_u = chi2_inv(1.0 - 0.5 * alpha, dof);
    const f64 chi_l = chi2_inv(0.5 * alpha, dof);
    out.p1 = mu;
    out.ci_lo = 2.0 * total_t / chi_u;
    out.ci_hi = 2.0 * total_t / chi_l;
    return StatsStatus::kOk;
}

StatsStatus poissfit(ArrayRef<f64> x, f64 alpha, Fit1Result& out) noexcept {
    const f64 nanv = qnan();
    out = {nanv, nanv, nanv};
    const usize n = x.size();
    if (n == 0 || !(alpha > 0.0 && alpha < 1.0)) {
        return StatsStatus::kInvalidArgument;
    }
    f64 s = 0.0;
    for (usize i = 0; i < n; ++i) {
        if (!(x[i] >= 0.0) || !is_finite(x[i])) {
            return StatsStatus::kInvalidArgument;
        }
        s += x[i];
    }
    const f64 nd = as<f64>(n);
    out.p1 = s / nd;
    out.ci_lo = (s == 0.0) ? 0.0 : (chi2_inv(0.5 * alpha, 2.0 * s) / (2.0 * nd));
    out.ci_hi = chi2_inv(1.0 - 0.5 * alpha, 2.0 * (s + 1.0)) / (2.0 * nd);
    return StatsStatus::kOk;
}

StatsStatus unifit(ArrayRef<f64> x, f64 alpha, Fit2Result& out) noexcept {
    const f64 nanv = qnan();
    out = {nanv, nanv, nanv, nanv, nanv, nanv};
    const usize n = x.size();
    if (n == 0 || !(alpha > 0.0 && alpha < 1.0)) {
        return StatsStatus::kInvalidArgument;
    }
    if (!is_finite(x[0])) {
        return StatsStatus::kInvalidArgument;
    }
    f64 mn = x[0];
    f64 mx = mn;
    for (usize i = 1; i < n; ++i) {
        if (!is_finite(x[i])) {
            return StatsStatus::kInvalidArgument;
        }
        if (x[i] < mn) {
            mn = x[i];
        }
        if (x[i] > mx) {
            mx = x[i];
        }
    }
    const f64 delta = (mx - mn) * (pow(alpha, -1.0 / as<f64>(n)) - 1.0);
    out.p1 = mn;
    out.p2 = mx;
    out.ci1_lo = mn - delta;
    out.ci1_hi = mn;
    out.ci2_lo = mx;
    out.ci2_hi = mx + delta;
    return StatsStatus::kOk;
}

StatsStatus binofit_scalar(f64 k, f64 n, f64 alpha, Fit1Result& out) noexcept {
    const f64 nanv = qnan();
    if (!(n > 0.0) || !is_finite(n) || floor(n) != n || !(k >= 0.0 && k <= n) ||
        !is_finite(k) || floor(k) != k || !(alpha > 0.0 && alpha < 1.0)) {
        out = {nanv, nanv, nanv};
        return StatsStatus::kInvalidArgument;
    }
    out.p1 = k / n;
    out.ci_lo = (k == 0.0) ? 0.0 : beta_inv(0.5 * alpha, k, n - k + 1.0);
    out.ci_hi = (k == n) ? 1.0 : beta_inv(1.0 - 0.5 * alpha, k + 1.0, n - k);
    return StatsStatus::kOk;
}

StatsStatus raylfit(ArrayRef<f64> x, f64 alpha, Fit1Result& out) noexcept {
    const f64 nanv = qnan();
    out = {nanv, nanv, nanv};
    const usize n = x.size();
    if (n == 0 || !(alpha > 0.0 && alpha < 1.0)) {
        return StatsStatus::kInvalidArgument;
    }
    f64 s2 = 0.0;
    for (usize i = 0; i < n; ++i) {
        if (!(x[i] >= 0.0) || !is_finite(x[i])) {
            return StatsStatus::kInvalidArgument;
        }
        s2 += x[i] * x[i];
    }
    const f64 two_n = 2.0 * as<f64>(n);
    const f64 sigma = sqrt(s2 / two_n);
    const f64 chi_u = chi2_inv(1.0 - 0.5 * alpha, two_n);
    const f64 chi_l = chi2_inv(0.5 * alpha, two_n);
    out.p1 = sigma;
    out.ci_lo = sigma * sqrt(two_n / chi_u);
    out.ci_hi = sigma * sqrt(two_n / chi_l);
    return StatsStatus::kOk;
}

// ============================================================================
// Negative Log-Likelihoods
// ============================================================================

f64 normlike(f64 mu, f64 sigma, ArrayRef<f64> x, ArrayRef<f64> cens, ArrayRef<f64> freq) noexcept {
    const usize n = x.size();
    if (n == 0) {
        return 0.0;
    }
    if (!(sigma > 0.0) || is_nan(mu)) {
        return qnan();
    }
    const bool use_c = !cens.empty();
    const bool use_f = !freq.empty();
    if ((use_c && cens.size() != n) || (use_f && freq.size() != n)) {
        return qnan();
    }
    const f64 inv_s = 1.0 / sigma;
    const f64 inv_2s2 = 0.5 * inv_s * inv_s;
    const f64 log_s = log(sigma);
    const f64 half_log2pi = 0.5 * kLog2Pi;
    f64 nl = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 w = use_f ? freq[i] : 1.0;
        if (w == 0.0) {
            continue;
        }
        const f64 xi = x[i];
        if (is_nan(xi)) {
            return qnan();
        }
        const f64 d = xi - mu;
        const bool is_cens = use_c && (cens[i] != 0.0);
        if (is_cens) {
            nl += w * neg_log_norm_surv(d * inv_s);
        } else {
            nl += w * (log_s + half_log2pi + d * d * inv_2s2);
        }
    }
    return nl;
}

f64 explike(f64 mu, ArrayRef<f64> x, ArrayRef<f64> cens, ArrayRef<f64> freq,
            f64* avar_out) noexcept {
    const f64 nanv = qnan();
    const usize n = x.size();
    if (n == 0) {
        if (avar_out) {
            *avar_out = nanv;
        }
        return 0.0;
    }
    if (!(mu > 0.0)) {
        if (avar_out) {
            *avar_out = nanv;
        }
        return nanv;
    }
    const bool use_c = !cens.empty();
    const bool use_f = !freq.empty();
    if ((use_c && cens.size() != n) || (use_f && freq.size() != n)) {
        if (avar_out) {
            *avar_out = nanv;
        }
        return nanv;
    }
    const f64 inv_mu = 1.0 / mu;
    const f64 inv_mu2 = inv_mu * inv_mu;
    const f64 inv_mu3 = inv_mu2 * inv_mu;
    const f64 log_mu = log(mu);

    f64 nl = 0.0;
    f64 info = 0.0;
    bool any = false;
    for (usize i = 0; i < n; ++i) {
        const f64 w = use_f ? freq[i] : 1.0;
        if (w == 0.0) {
            continue;
        }
        const f64 xi = x[i];
        if (!(xi >= 0.0)) {
            if (avar_out) {
                *avar_out = nanv;
            }
            return nanv;
        }
        any = true;
        const bool is_cens = use_c && (cens[i] != 0.0);
        if (is_cens) {
            nl += w * (xi * inv_mu);
            info += w * (2.0 * xi * inv_mu3);
        } else {
            nl += w * (log_mu + xi * inv_mu);
            info += w * (-inv_mu2 + 2.0 * xi * inv_mu3);
        }
    }
    if (avar_out) {
        *avar_out = (!any || info == 0.0 || !is_finite(info)) ? nanv : (1.0 / info);
    }
    return any ? nl : 0.0;
}

f64 lognlike(f64 mu, f64 sigma, ArrayRef<f64> x, ArrayRef<f64> cens, ArrayRef<f64> freq,
             f64* avar_out_2x2) noexcept {
    const f64 nanv = qnan();
    auto set_avar_nan = [&]() {
        if (avar_out_2x2) {
            for (i32 i = 0; i < 4; ++i) {
                avar_out_2x2[i] = nanv;
            }
        }
    };
    const usize n = x.size();
    if (n == 0) {
        set_avar_nan();
        return 0.0;
    }
    if (!(sigma > 0.0) || is_nan(mu)) {
        set_avar_nan();
        return nanv;
    }
    const bool use_c = !cens.empty();
    const bool use_f = !freq.empty();
    if ((use_c && cens.size() != n) || (use_f && freq.size() != n)) {
        set_avar_nan();
        return nanv;
    }

    const f64 inv_s = 1.0 / sigma;
    const f64 inv_s2 = inv_s * inv_s;
    const f64 inv_s3 = inv_s2 * inv_s;
    const f64 inv_s4 = inv_s2 * inv_s2;
    const f64 inv_2s2 = 0.5 * inv_s2;
    const f64 log_s = log(sigma);
    const f64 half_l2pi = 0.5 * kLog2Pi;

    f64 nl = 0.0;
    f64 i00 = 0.0;
    f64 i01 = 0.0;
    f64 i11 = 0.0;
    bool any = false;
    for (usize i = 0; i < n; ++i) {
        const f64 w = use_f ? freq[i] : 1.0;
        if (w == 0.0) {
            continue;
        }
        const f64 xi = x[i];
        if (!(xi > 0.0)) {
            set_avar_nan();
            return nanv;
        }
        any = true;
        const f64 lx = log(xi);
        const f64 d = lx - mu;
        const f64 z = d * inv_s;
        const bool is_cens = use_c && (cens[i] != 0.0);
        if (!is_cens) {
            nl += w * (lx + log_s + half_l2pi + d * d * inv_2s2);
            i00 += w * inv_s2;
            i11 += w * (-inv_s2 + 3.0 * d * d * inv_s4);
            i01 += w * 2.0 * d * inv_s3;
        } else {
            nl += w * neg_log_norm_surv(z);
            const f64 h = inv_mills(z);
            const f64 hp = h * (h - z);
            i00 += w * hp * inv_s2;
            i11 += w * (2.0 * z * h + z * z * hp) * inv_s2;
            i01 += w * (z * hp + h) * inv_s2;
        }
    }
    if (avar_out_2x2) {
        if (!any) {
            set_avar_nan();
        } else {
            const f64 det = i00 * i11 - i01 * i01;
            if (det == 0.0 || !is_finite(det)) {
                set_avar_nan();
            } else {
                const f64 inv = 1.0 / det;
                avar_out_2x2[0] = i11 * inv;
                avar_out_2x2[1] = -i01 * inv;
                avar_out_2x2[2] = -i01 * inv;
                avar_out_2x2[3] = i00 * inv;
            }
        }
    }
    return any ? nl : 0.0;
}

f64 gamlike(f64 a, f64 b, ArrayRef<f64> x) noexcept {
    const usize n = x.size();
    if (n == 0) {
        return pinf();
    }
    if (!(a > 0.0) || !(b > 0.0)) {
        return qnan();
    }
    f64 sum_log_x = 0.0;
    f64 sx = 0.0;
    for (usize i = 0; i < n; ++i) {
        if (is_nan(x[i])) {
            return qnan();
        }
        if (x[i] <= 0.0) {
            return pinf();
        }
        sum_log_x += log(x[i]);
        sx += x[i];
    }
    const f64 nd = as<f64>(n);
    return -(a - 1.0) * sum_log_x + sx / b + nd * a * log(b) + nd * specfun::gammaln(a);
}

f64 betalike(f64 a, f64 b, ArrayRef<f64> x) noexcept {
    const usize n = x.size();
    if (n == 0) {
        return pinf();
    }
    if (!(a > 0.0) || !(b > 0.0)) {
        return qnan();
    }
    f64 sum_log = 0.0;
    f64 sum_log1m = 0.0;
    for (usize i = 0; i < n; ++i) {
        if (!(x[i] > 0.0 && x[i] < 1.0)) {
            return qnan();
        }
        sum_log += log(x[i]);
        sum_log1m += log1p(-x[i]);
    }
    const f64 log_beta = specfun::betaln(a, b);
    return -(a - 1.0) * sum_log - (b - 1.0) * sum_log1m + as<f64>(n) * log_beta;
}

f64 wbllike(f64 scale, f64 shape, ArrayRef<f64> x, ArrayRef<f64> cens,
            ArrayRef<f64> freq) noexcept {
    const usize n = x.size();
    if (n == 0) {
        return 0.0;
    }
    if (!(scale > 0.0) || !(shape > 0.0)) {
        return qnan();
    }
    const bool use_c = !cens.empty();
    const bool use_f = !freq.empty();
    if ((use_c && cens.size() != n) || (use_f && freq.size() != n)) {
        return qnan();
    }
    const f64 log_a = log(scale);
    const f64 log_k = log(shape);
    f64 nl = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 w = use_f ? freq[i] : 1.0;
        if (w == 0.0) {
            continue;
        }
        const f64 xi = x[i];
        if (!(xi > 0.0)) {
            return qnan();
        }
        const f64 pow_term = pow(xi / scale, shape);
        const bool is_cens = use_c && (cens[i] != 0.0);
        if (is_cens) {
            nl += w * pow_term;
        } else {
            nl += w * (-log_k + shape * log_a - (shape - 1.0) * log(xi) + pow_term);
        }
    }
    return nl;
}

f64 evlike(f64 mu, f64 sigma, ArrayRef<f64> x, ArrayRef<f64> cens, ArrayRef<f64> freq) noexcept {
    const usize n = x.size();
    if (n == 0) {
        return 0.0;
    }
    if (!(sigma > 0.0) || is_nan(mu)) {
        return qnan();
    }
    const bool use_c = !cens.empty();
    const bool use_f = !freq.empty();
    if ((use_c && cens.size() != n) || (use_f && freq.size() != n)) {
        return qnan();
    }
    const f64 inv_s = 1.0 / sigma;
    const f64 log_s = log(sigma);
    f64 nl = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 w = use_f ? freq[i] : 1.0;
        if (w == 0.0) {
            continue;
        }
        const f64 xi = x[i];
        if (is_nan(xi)) {
            return qnan();
        }
        const f64 z = (xi - mu) * inv_s;
        const f64 ez = exp(z);
        const bool is_cens = use_c && (cens[i] != 0.0);
        if (is_cens) {
            nl += w * ez;
        } else {
            nl += w * (log_s - z + ez);
        }
    }
    return nl;
}

f64 gevlike(f64 k, f64 sigma, f64 mu, ArrayRef<f64> x) noexcept {
    const usize n = x.size();
    if (n == 0) {
        return pinf();
    }
    if (!(sigma > 0.0) || is_nan(k) || is_nan(mu)) {
        return qnan();
    }
    if (k == 0.0) {
        f64 sz = 0.0;
        f64 sexp = 0.0;
        for (usize i = 0; i < n; ++i) {
            if (is_nan(x[i])) {
                return qnan();
            }
            const f64 z = (x[i] - mu) / sigma;
            sz += z;
            sexp += exp(-z);
        }
        return as<f64>(n) * log(sigma) + sz + sexp;
    }
    f64 slog_t = 0.0;
    f64 st_invk = 0.0;
    for (usize i = 0; i < n; ++i) {
        if (is_nan(x[i])) {
            return qnan();
        }
        const f64 z = (x[i] - mu) / sigma;
        const f64 kz = k * z;
        if (kz <= -1.0) {
            return qnan();
        }
        const f64 lt = log1p(kz);
        slog_t += lt;
        st_invk += exp(-lt / k);
    }
    return as<f64>(n) * log(sigma) + (1.0 / k + 1.0) * slog_t + st_invk;
}

f64 gplike(f64 k, f64 sigma, ArrayRef<f64> x) noexcept {
    const usize n = x.size();
    if (n == 0) {
        return pinf();
    }
    if (sigma == 0.0 || is_nan(sigma) || is_nan(k)) {
        return qnan();
    }
    if (sigma < 0.0) {
        return -pinf();
    }
    if (k == 0.0) {
        f64 sx = 0.0;
        for (usize i = 0; i < n; ++i) {
            if (is_nan(x[i])) {
                return qnan();
            }
            sx += x[i];
        }
        return as<f64>(n) * log(sigma) + sx / sigma;
    }
    f64 slog_t = 0.0;
    for (usize i = 0; i < n; ++i) {
        if (is_nan(x[i])) {
            return qnan();
        }
        const f64 kz = k * x[i] / sigma;
        if (kz <= -1.0) {
            return pinf();
        }
        slog_t += log1p(kz);
    }
    return as<f64>(n) * log(sigma) + (1.0 / k + 1.0) * slog_t;
}

} // namespace keel::stats
