// keelcpp/stats/src/dist_fits.cpp
// Pure zero-STL MLE / PWM fitters and Wald confidence intervals for
// Gamma, Weibull, Beta, Negative Binomial, Type-I Extreme Value,
// Generalized Pareto, and Generalized Extreme Value distributions.

#include "keel/stats.h"

#include "keel/algo.h"
#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"
#include "keel/specfun.h"

namespace keel::stats {

namespace {

constexpr f64 kPi = 3.14159265358979323846;
constexpr f64 kEulerGamma = 0.57721566490153286;

f64 digamma_local(f64 x) noexcept {
    f64 s = 0.0;
    while (x < 6.0) {
        s -= 1.0 / x;
        x += 1.0;
    }
    const f64 x2 = 1.0 / (x * x);
    f64 r = log(x) - 0.5 / x;
    r -= x2 * (1.0 / 12.0 - x2 * (1.0 / 120.0 - x2 * (1.0 / 252.0)));
    return r + s;
}

f64 trigamma_local(f64 x) noexcept {
    f64 s = 0.0;
    while (x < 6.0) {
        s += 1.0 / (x * x);
        x += 1.0;
    }
    const f64 x2 = 1.0 / (x * x);
    f64 r = 1.0 / x + 0.5 / (x * x);
    r += (x2 / x) * (1.0 / 6.0 - x2 * (1.0 / 30.0 - x2 * (1.0 / 42.0)));
    return r + s;
}

enum class CiTransform : u8 {
    kLinear = 0,
    kLog,
    kLogit,
};

void apply_ci_1d(f64 t, f64 se, f64 z, CiTransform tr, f64& lo, f64& hi) noexcept {
    const f64 qnan = Limits<f64>::quiet_nan();
    switch (tr) {
        case CiTransform::kLinear:
            lo = t - z * se;
            hi = t + z * se;
            return;
        case CiTransform::kLog: {
            if (!(t > 0.0)) {
                lo = qnan;
                hi = qnan;
                return;
            }
            const f64 se_log = se / t;
            lo = t * exp(-z * se_log);
            hi = t * exp(z * se_log);
            return;
        }
        case CiTransform::kLogit: {
            if (!(t > 0.0 && t < 1.0)) {
                lo = qnan;
                hi = qnan;
                return;
            }
            const f64 se_logit = se / (t * (1.0 - t));
            const f64 logit = log(t / (1.0 - t));
            lo = 1.0 / (1.0 + exp(-(logit - z * se_logit)));
            hi = 1.0 / (1.0 + exp(-(logit + z * se_logit)));
            return;
        }
    }
}

void wald_ci_from_hessian_2d(f64 t1, f64 t2, f64 h11, f64 h22, f64 h12, CiTransform tr1,
                             CiTransform tr2, f64 alpha, Fit2Result& out) noexcept {
    const f64 qnan = Limits<f64>::quiet_nan();
    const f64 det = h11 * h22 - h12 * h12;
    if (abs(det) < 1e-300 || h11 < 0.0 || h22 < 0.0) {
        out.ci1_lo = qnan;
        out.ci1_hi = qnan;
        out.ci2_lo = qnan;
        out.ci2_hi = qnan;
        return;
    }
    const f64 se1 = sqrt(max(h22 / det, 0.0));
    const f64 se2 = sqrt(max(h11 / det, 0.0));
    const f64 z = norm_inv(1.0 - 0.5 * alpha, 0.0, 1.0);
    apply_ci_1d(t1, se1, z, tr1, out.ci1_lo, out.ci1_hi);
    apply_ci_1d(t2, se2, z, tr2, out.ci2_lo, out.ci2_hi);
}

template <typename NllFn>
void wald_ci_fd_2d(NllFn&& nll, f64 t1, f64 t2, CiTransform tr1, CiTransform tr2, f64 alpha,
                   Fit2Result& out) noexcept {
    const f64 h1 = max(abs(t1), 1.0) * 1e-4;
    const f64 h2 = max(abs(t2), 1.0) * 1e-4;
    const f64 f00 = nll(t1, t2);
    const f64 f1p = nll(t1 + h1, t2);
    const f64 f1m = nll(t1 - h1, t2);
    const f64 f2p = nll(t1, t2 + h2);
    const f64 f2m = nll(t1, t2 - h2);
    const f64 f12pp = nll(t1 + h1, t2 + h2);
    const f64 f12mm = nll(t1 - h1, t2 - h2);
    const f64 f12pm = nll(t1 + h1, t2 - h2);
    const f64 f12mp = nll(t1 - h1, t2 + h2);
    const f64 h11 = (f1p - 2.0 * f00 + f1m) / (h1 * h1);
    const f64 h22 = (f2p - 2.0 * f00 + f2m) / (h2 * h2);
    const f64 h12 = (f12pp - f12pm - f12mp + f12mm) / (4.0 * h1 * h2);
    wald_ci_from_hessian_2d(t1, t2, h11, h22, h12, tr1, tr2, alpha, out);
}

f64 nll_gam_simple(f64 a, f64 b, ArrayRef<f64> xv) noexcept {
    if (!(a > 0.0) || !(b > 0.0)) {
        return Limits<f64>::infinity();
    }
    const f64 dn = as<f64>(xv.size());
    f64 sx = 0.0;
    f64 slx = 0.0;
    for (usize i = 0; i < xv.size(); ++i) {
        sx += xv[i];
        slx += log(xv[i]);
    }
    return dn * specfun::gammaln(a) + dn * a * log(b) - (a - 1.0) * slx + sx / b;
}

f64 nll_gam_full(f64 a, f64 b, ArrayRef<f64> xv, ArrayRef<f64> cens, ArrayRef<f64> freq) noexcept {
    if (!(a > 0.0) || !(b > 0.0)) {
        return Limits<f64>::infinity();
    }
    f64 nuw = 0.0;
    f64 lxw = 0.0;
    f64 sxw = 0.0;
    f64 s_cens = 0.0;
    for (usize i = 0; i < xv.size(); ++i) {
        const f64 f = freq.empty() ? 1.0 : freq[i];
        const f64 c = cens.empty() ? 0.0 : (cens[i] != 0.0 ? 1.0 : 0.0);
        const f64 u = 1.0 - c;
        nuw += f * u;
        lxw += f * u * log(xv[i]);
        sxw += f * u * xv[i];
        if (c > 0.5 && f > 0.0) {
            const f64 q = 1.0 - specfun::gammainc(xv[i] / b, a);
            if (!(q > 0.0)) {
                return Limits<f64>::infinity();
            }
            s_cens -= f * log(q);
        }
    }
    return nuw * (specfun::gammaln(a) + a * log(b)) - (a - 1.0) * lxw + sxw / b + s_cens;
}

f64 nll_wbl_simple(f64 a, f64 b, ArrayRef<f64> xv) noexcept {
    if (!(a > 0.0) || !(b > 0.0)) {
        return Limits<f64>::infinity();
    }
    const f64 dn = as<f64>(xv.size());
    f64 slx = 0.0;
    f64 sw = 0.0;
    for (usize i = 0; i < xv.size(); ++i) {
        slx += log(xv[i]);
        sw += pow(xv[i] / a, b);
    }
    return -dn * log(b) + dn * b * log(a) - (b - 1.0) * slx + sw;
}

f64 nll_wbl_full(f64 a, f64 b, ArrayRef<f64> xv, ArrayRef<f64> cens, ArrayRef<f64> freq) noexcept {
    if (!(a > 0.0) || !(b > 0.0)) {
        return Limits<f64>::infinity();
    }
    f64 nuw = 0.0;
    f64 lxw = 0.0;
    f64 sum_w = 0.0;
    for (usize i = 0; i < xv.size(); ++i) {
        const f64 f = freq.empty() ? 1.0 : freq[i];
        const f64 c = cens.empty() ? 0.0 : (cens[i] != 0.0 ? 1.0 : 0.0);
        const f64 u = 1.0 - c;
        nuw += f * u;
        lxw += f * u * log(xv[i]);
        sum_w += f * pow(xv[i] / a, b);
    }
    return nuw * (-log(b) + b * log(a)) - (b - 1.0) * lxw + sum_w;
}

f64 nll_ev_full(f64 mu, f64 sigma, ArrayRef<f64> xv, ArrayRef<f64> cens,
                ArrayRef<f64> freq) noexcept {
    if (!(sigma > 0.0)) {
        return Limits<f64>::infinity();
    }
    f64 t_max = -Limits<f64>::infinity();
    for (usize i = 0; i < xv.size(); ++i) {
        const f64 t = (xv[i] - mu) / sigma;
        if (t > t_max) {
            t_max = t;
        }
    }
    f64 nuw = 0.0;
    f64 suw = 0.0;
    f64 wexp = 0.0;
    for (usize i = 0; i < xv.size(); ++i) {
        const f64 t = (xv[i] - mu) / sigma;
        const f64 f = freq.empty() ? 1.0 : freq[i];
        const f64 c = cens.empty() ? 0.0 : (cens[i] != 0.0 ? 1.0 : 0.0);
        const f64 u = 1.0 - c;
        nuw += f * u;
        suw += f * u * t;
        wexp += f * exp(t - t_max);
    }
    return nuw * log(sigma) - suw + exp(t_max) * wexp;
}

f64 nll_gp(f64 k, f64 sigma, ArrayRef<f64> xv) noexcept {
    if (!(sigma > 0.0)) {
        return Limits<f64>::infinity();
    }
    const usize n = xv.size();
    f64 nll = as<f64>(n) * log(sigma);
    if (abs(k) < 1e-10) {
        f64 s = 0.0;
        for (usize i = 0; i < n; ++i) {
            s += xv[i];
        }
        return nll + s / sigma;
    }
    const f64 inv_sigma = 1.0 / sigma;
    for (usize i = 0; i < n; ++i) {
        if (k * xv[i] * inv_sigma <= -1.0) {
            return Limits<f64>::infinity();
        }
    }
    f64 s = 0.0;
    for (usize i = 0; i < n; ++i) {
        s += log1p(k * xv[i] * inv_sigma);
    }
    return nll + (1.0 + 1.0 / k) * s;
}

void gp_nll_grad(f64 k, f64 sigma, ArrayRef<f64> xv, f64& gk, f64& gs) noexcept {
    const f64 hk = max(abs(k), 1.0) * 1e-7;
    const f64 hs = max(abs(sigma), 1.0) * 1e-7;
    gk = (nll_gp(k + hk, sigma, xv) - nll_gp(k - hk, sigma, xv)) / (2.0 * hk);
    gs = (nll_gp(k, sigma + hs, xv) - nll_gp(k, sigma - hs, xv)) / (2.0 * hs);
}

f64 nll_gev(f64 k, f64 sigma, f64 mu, ArrayRef<f64> xv) noexcept {
    if (!(sigma > 0.0)) {
        return Limits<f64>::infinity();
    }
    const usize n = xv.size();
    f64 nll = as<f64>(n) * log(sigma);
    if (abs(k) < 1e-10) {
        for (usize i = 0; i < n; ++i) {
            const f64 z = (xv[i] - mu) / sigma;
            nll += z + exp(-z);
        }
    } else {
        for (usize i = 0; i < n; ++i) {
            const f64 t = 1.0 + k * (xv[i] - mu) / sigma;
            if (!(t > 0.0)) {
                return Limits<f64>::infinity();
            }
            const f64 log_t = log(t);
            const f64 t_pow = exp(-log_t / k);
            nll += (1.0 + 1.0 / k) * log_t + t_pow;
        }
    }
    return nll;
}

} // namespace

StatsStatus gamfit(ArrayRef<f64> x, ArrayRef<f64> cens, ArrayRef<f64> freq, f64 alpha,
                   bool compute_ci, Fit2Result& out) noexcept {
    const usize n = x.size();
    if (n < 2) {
        return StatsStatus::kInvalidArgument;
    }
    if ((!cens.empty() && cens.size() != n) || (!freq.empty() && freq.size() != n)) {
        return StatsStatus::kInvalidArgument;
    }
    bool has_cens = false;
    bool freq_trivial = true;
    for (usize i = 0; i < n; ++i) {
        if (!(x[i] > 0.0)) {
            return StatsStatus::kInvalidArgument;
        }
        if (!cens.empty() && cens[i] != 0.0) {
            has_cens = true;
        }
        if (!freq.empty()) {
            if (!(freq[i] >= 0.0)) {
                return StatsStatus::kInvalidArgument;
            }
            if (freq[i] != 1.0) {
                freq_trivial = false;
            }
        }
    }

    f64 a = 1.0;
    f64 b = 1.0;
    if (!has_cens && freq_trivial) {
        f64 sum = 0.0;
        f64 sum_log = 0.0;
        for (usize i = 0; i < n; ++i) {
            sum += x[i];
            sum_log += log(x[i]);
        }
        const f64 mu = sum / as<f64>(n);
        const f64 mean_log = sum_log / as<f64>(n);
        const f64 s = log(mu) - mean_log;
        if (!(s > 0.0)) {
            a = Limits<f64>::infinity();
            b = mu;
        } else {
            a = (3.0 - s + sqrt((s - 3.0) * (s - 3.0) + 24.0 * s)) / (12.0 * s);
            for (i32 it = 0; it < 50; ++it) {
                const f64 fa = log(a) - digamma_local(a) - s;
                const f64 fpa = 1.0 / a - trigamma_local(a);
                const f64 step = fa / fpa;
                const f64 new_a = a - step;
                a = (new_a > 0.0) ? new_a : (a * 0.5);
                if (abs(step) < 1e-10 * max(a, 1.0)) {
                    break;
                }
            }
            b = mu / a;
        }
    } else {
        f64 nuw = 0.0;
        f64 sx = 0.0;
        f64 slx = 0.0;
        for (usize i = 0; i < n; ++i) {
            const f64 f = freq.empty() ? 1.0 : freq[i];
            const f64 u = (cens.empty() || cens[i] == 0.0) ? 1.0 : 0.0;
            nuw += f * u;
            sx += f * u * x[i];
            slx += f * u * log(x[i]);
        }
        if (!(nuw >= 2.0)) {
            return StatsStatus::kInvalidArgument;
        }
        const f64 mean_uw = sx / nuw;
        const f64 s_init = log(mean_uw) - slx / nuw;
        if (s_init > 0.0) {
            a = (3.0 - s_init + sqrt((s_init - 3.0) * (s_init - 3.0) + 24.0 * s_init)) /
                (12.0 * s_init);
            b = mean_uw / a;
        } else {
            a = 1.0;
            b = mean_uw;
        }
        auto nll = [&](f64 aa, f64 bb) noexcept { return nll_gam_full(aa, bb, x, cens, freq); };
        f64 f_cur = nll(a, b);
        for (i32 it = 0; it < 100; ++it) {
            const f64 ha = max(abs(a), 1.0) * 1e-5;
            const f64 hb = max(abs(b), 1.0) * 1e-5;
            const f64 ga = (nll(a + ha, b) - nll(a - ha, b)) / (2.0 * ha);
            const f64 gb = (nll(a, b + hb) - nll(a, b - hb)) / (2.0 * hb);
            if (sqrt(ga * ga + gb * gb) < 1e-12) {
                break;
            }
            const f64 haa = (nll(a + ha, b) - 2.0 * f_cur + nll(a - ha, b)) / (ha * ha);
            const f64 hbb = (nll(a, b + hb) - 2.0 * f_cur + nll(a, b - hb)) / (hb * hb);
            const f64 hab = (nll(a + ha, b + hb) - nll(a + ha, b - hb) - nll(a - ha, b + hb) +
                             nll(a - ha, b - hb)) /
                            (4.0 * ha * hb);
            const f64 det = haa * hbb - hab * hab;
            if (abs(det) < 1e-300) {
                break;
            }
            const f64 da = (-ga * hbb + gb * hab) / det;
            const f64 db = (-gb * haa + ga * hab) / det;
            f64 sc = 1.0;
            bool ok = false;
            for (i32 bt = 0; bt < 30; ++bt) {
                const f64 an = a + sc * da;
                const f64 bn = b + sc * db;
                if (an > 0.0 && bn > 0.0) {
                    const f64 f_new = nll(an, bn);
                    if (is_finite(f_new) && f_new < f_cur - 1e-15) {
                        a = an;
                        b = bn;
                        f_cur = f_new;
                        ok = true;
                        break;
                    }
                }
                sc *= 0.5;
            }
            if (!ok) {
                break;
            }
            if (abs(sc * da) + abs(sc * db) < 1e-12 * (abs(a) + abs(b) + 1.0)) {
                break;
            }
        }
    }

    out = Fit2Result{a, b, 0.0, 0.0, 0.0, 0.0};
    if (compute_ci) {
        if (!has_cens && freq_trivial) {
            wald_ci_fd_2d([&](f64 aa, f64 bb) noexcept { return nll_gam_simple(aa, bb, x); }, a, b,
                          CiTransform::kLog, CiTransform::kLog, alpha, out);
        } else {
            wald_ci_fd_2d(
                [&](f64 aa, f64 bb) noexcept { return nll_gam_full(aa, bb, x, cens, freq); }, a, b,
                CiTransform::kLog, CiTransform::kLog, alpha, out);
        }
    }
    return StatsStatus::kOk;
}

StatsStatus wblfit(ArrayRef<f64> x, ArrayRef<f64> cens, ArrayRef<f64> freq, f64 alpha,
                   bool compute_ci, Fit2Result& out, Allocator* alloc) {
    const usize n = x.size();
    if (n < 2) {
        return StatsStatus::kInvalidArgument;
    }
    if ((!cens.empty() && cens.size() != n) || (!freq.empty() && freq.size() != n)) {
        return StatsStatus::kInvalidArgument;
    }
    bool has_cens = false;
    bool freq_trivial = true;
    for (usize i = 0; i < n; ++i) {
        if (!(x[i] > 0.0)) {
            return StatsStatus::kInvalidArgument;
        }
        if (!cens.empty() && cens[i] != 0.0) {
            has_cens = true;
        }
        if (!freq.empty()) {
            if (!(freq[i] >= 0.0)) {
                return StatsStatus::kInvalidArgument;
            }
            if (freq[i] != 1.0) {
                freq_trivial = false;
            }
        }
    }

    f64 a = 1.0;
    f64 b = 1.0;
    if (!has_cens && freq_trivial) {
        ArenaTemp scratch = get_scratch(alloc);
        Array<f64> lx(n, 0.0, scratch.allocator());
        f64 sum_log = 0.0;
        for (usize i = 0; i < n; ++i) {
            lx[i] = log(x[i]);
            sum_log += lx[i];
        }
        const f64 mean_log = sum_log / as<f64>(n);
        f64 v2 = 0.0;
        for (usize i = 0; i < n; ++i) {
            const f64 d = lx[i] - mean_log;
            v2 += d * d;
        }
        const f64 sd_log = sqrt(v2 / as<f64>(n));
        b = kPi / (sd_log * sqrt(6.0) + 1e-12);
        if (!(b > 0.1) || !is_finite(b)) {
            b = 1.0;
        }
        for (i32 it = 0; it < 100; ++it) {
            f64 s0 = 0.0;
            f64 s1 = 0.0;
            f64 s2 = 0.0;
            for (usize i = 0; i < n; ++i) {
                const f64 w = pow(x[i], b);
                s0 += w;
                s1 += w * lx[i];
                s2 += w * lx[i] * lx[i];
            }
            const f64 r_val = s1 / s0;
            const f64 g = 1.0 / b + mean_log - r_val;
            const f64 dr = (s2 * s0 - s1 * s1) / (s0 * s0);
            const f64 dg = -1.0 / (b * b) - dr;
            const f64 step = g / dg;
            const f64 new_b = b - step;
            b = (new_b > 0.0) ? new_b : (b * 0.5);
            if (abs(step) < 1e-10 * max(b, 1.0)) {
                break;
            }
        }
        f64 s_pow = 0.0;
        for (usize i = 0; i < n; ++i) {
            s_pow += pow(x[i], b);
        }
        a = pow(s_pow / as<f64>(n), 1.0 / b);
    } else {
        f64 nuw_init = 0.0;
        f64 sum_lx = 0.0;
        for (usize i = 0; i < n; ++i) {
            const f64 f = freq.empty() ? 1.0 : freq[i];
            const f64 u = (cens.empty() || cens[i] == 0.0) ? 1.0 : 0.0;
            nuw_init += f * u;
            sum_lx += f * u * log(x[i]);
        }
        const f64 mean_log = (nuw_init > 0.0) ? (sum_lx / nuw_init) : 0.0;
        f64 v2 = 0.0;
        for (usize i = 0; i < n; ++i) {
            const f64 f = freq.empty() ? 1.0 : freq[i];
            const f64 u = (cens.empty() || cens[i] == 0.0) ? 1.0 : 0.0;
            const f64 d = log(x[i]) - mean_log;
            v2 += f * u * d * d;
        }
        const f64 sd_log = (nuw_init > 0.0) ? sqrt(v2 / nuw_init) : 1.0;
        f64 bcur = kPi / (sd_log * sqrt(6.0) + 1e-12);
        if (!(bcur > 0.05) || !is_finite(bcur)) {
            bcur = 1.0;
        }
        auto profile_a = [&](f64 bv) noexcept -> f64 {
            f64 s = 0.0;
            f64 u_sum = 0.0;
            for (usize i = 0; i < n; ++i) {
                const f64 f = freq.empty() ? 1.0 : freq[i];
                const f64 u = (cens.empty() || cens[i] == 0.0) ? 1.0 : 0.0;
                s += f * pow(x[i], bv);
                u_sum += f * u;
            }
            if (!(u_sum > 0.0)) {
                return Limits<f64>::quiet_nan();
            }
            return pow(s / u_sum, 1.0 / bv);
        };
        auto nll_at_b = [&](f64 bv) noexcept -> f64 {
            return nll_wbl_full(profile_a(bv), bv, x, cens, freq);
        };
        f64 f_cur = nll_at_b(bcur);
        for (i32 it = 0; it < 100; ++it) {
            const f64 h = max(abs(bcur), 1.0) * 1e-6;
            const f64 g = (nll_at_b(bcur + h) - nll_at_b(bcur - h)) / (2.0 * h);
            const f64 hh = (nll_at_b(bcur + h) - 2.0 * f_cur + nll_at_b(bcur - h)) / (h * h);
            if (abs(hh) < 1e-300) {
                break;
            }
            const f64 step = g / hh;
            f64 sc = 1.0;
            bool ok = false;
            for (i32 bt = 0; bt < 30; ++bt) {
                const f64 bn = bcur - sc * step;
                if (bn > 0.0) {
                    const f64 f_new = nll_at_b(bn);
                    if (is_finite(f_new) && f_new < f_cur - 1e-15) {
                        bcur = bn;
                        f_cur = f_new;
                        ok = true;
                        break;
                    }
                }
                sc *= 0.5;
            }
            if (!ok) {
                break;
            }
            if (abs(sc * step) < 1e-12 * max(bcur, 1.0)) {
                break;
            }
        }
        a = profile_a(bcur);
        b = bcur;
    }

    out = Fit2Result{a, b, 0.0, 0.0, 0.0, 0.0};
    if (compute_ci) {
        if (!has_cens && freq_trivial) {
            wald_ci_fd_2d([&](f64 aa, f64 bb) noexcept { return nll_wbl_simple(aa, bb, x); }, a, b,
                          CiTransform::kLog, CiTransform::kLog, alpha, out);
        } else {
            wald_ci_fd_2d(
                [&](f64 aa, f64 bb) noexcept { return nll_wbl_full(aa, bb, x, cens, freq); }, a, b,
                CiTransform::kLog, CiTransform::kLog, alpha, out);
        }
    }
    return StatsStatus::kOk;
}

StatsStatus betafit(ArrayRef<f64> x, f64 alpha, bool compute_ci, Fit2Result& out) noexcept {
    const usize n = x.size();
    if (n < 2) {
        return StatsStatus::kInvalidArgument;
    }
    f64 sum_x = 0.0;
    f64 sum_log_x = 0.0;
    f64 sum_log_1mx = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 v = x[i];
        if (!(v > 0.0 && v < 1.0)) {
            return StatsStatus::kInvalidArgument;
        }
        sum_x += v;
        sum_log_x += log(v);
        sum_log_1mx += log(1.0 - v);
    }
    const f64 dn = as<f64>(n);
    const f64 mean_x = sum_x / dn;
    f64 v0 = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 d = x[i] - mean_x;
        v0 += d * d;
    }
    v0 /= dn;
    if (!(v0 > 0.0) || v0 >= mean_x * (1.0 - mean_x)) {
        v0 = mean_x * (1.0 - mean_x) * 0.5;
    }
    const f64 common = mean_x * (1.0 - mean_x) / v0 - 1.0;
    f64 a = mean_x * common;
    f64 b = (1.0 - mean_x) * common;
    if (!(a > 0.0)) a = 1.0;
    if (!(b > 0.0)) b = 1.0;

    const f64 s_log_x = sum_log_x / dn;
    const f64 s_log_1x = sum_log_1mx / dn;
    for (i32 it = 0; it < 100; ++it) {
        const f64 psi_ab = digamma_local(a + b);
        const f64 f1 = digamma_local(a) - psi_ab - s_log_x;
        const f64 f2 = digamma_local(b) - psi_ab - s_log_1x;
        const f64 tab = trigamma_local(a + b);
        const f64 j11 = trigamma_local(a) - tab;
        const f64 j22 = trigamma_local(b) - tab;
        const f64 j12 = -tab;
        const f64 det = j11 * j22 - j12 * j12;
        if (abs(det) < 1e-300) {
            break;
        }
        const f64 da = (j22 * f1 - j12 * f2) / det;
        const f64 db = (j11 * f2 - j12 * f1) / det;
        a = (a - da > 0.0) ? (a - da) : (a * 0.5);
        b = (b - db > 0.0) ? (b - db) : (b * 0.5);
        if (abs(da) + abs(db) < 1e-10 * (abs(a) + abs(b) + 1.0)) {
            break;
        }
    }

    out = Fit2Result{a, b, 0.0, 0.0, 0.0, 0.0};
    if (compute_ci) {
        const f64 tab = trigamma_local(a + b);
        const f64 h11 = dn * (trigamma_local(a) - tab);
        const f64 h22 = dn * (trigamma_local(b) - tab);
        const f64 h12 = -dn * tab;
        wald_ci_from_hessian_2d(a, b, h11, h22, h12, CiTransform::kLog, CiTransform::kLog, alpha,
                                out);
    }
    return StatsStatus::kOk;
}

StatsStatus nbinfit(ArrayRef<f64> x, f64 alpha, bool compute_ci, Fit2Result& out) noexcept {
    const usize n = x.size();
    if (n < 2) {
        return StatsStatus::kInvalidArgument;
    }
    f64 sum = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 v = x[i];
        if (v < 0.0 || abs(v - round(v)) > 1e-9) {
            return StatsStatus::kInvalidArgument;
        }
        sum += v;
    }
    const f64 dn = as<f64>(n);
    const f64 mean_x = sum / dn;
    f64 v2 = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 d = x[i] - mean_x;
        v2 += d * d;
    }
    const f64 var_x = v2 / as<f64>(n - 1);
    if (var_x <= mean_x) {
        return StatsStatus::kSingularMatrix;
    }
    f64 r = (mean_x * mean_x) / (var_x - mean_x);
    if (!(r > 0.0)) {
        r = 1.0;
    }

    for (i32 it = 0; it < 100; ++it) {
        const f64 frac = r / (r + mean_x);
        f64 s1 = 0.0;
        f64 s2 = 0.0;
        for (usize i = 0; i < n; ++i) {
            s1 += digamma_local(x[i] + r) - digamma_local(r);
            s2 += trigamma_local(x[i] + r) - trigamma_local(r);
        }
        const f64 f_val = s1 + dn * log(frac);
        const f64 fp = s2 + dn * mean_x / (r * (r + mean_x));
        if (abs(fp) < 1e-300) {
            break;
        }
        const f64 step = f_val / fp;
        r = (r - step > 0.0) ? (r - step) : (r * 0.5);
        if (abs(step) < 1e-10 * max(r, 1.0)) {
            break;
        }
    }
    const f64 p = r / (r + mean_x);

    out = Fit2Result{r, p, 0.0, 0.0, 0.0, 0.0};
    if (compute_ci) {
        f64 sxh = 0.0;
        for (usize i = 0; i < n; ++i) {
            sxh += trigamma_local(x[i] + r);
        }
        const f64 h11 = -sxh + dn * trigamma_local(r);
        const f64 h22 = dn * r / (p * p) + sum / ((1.0 - p) * (1.0 - p));
        const f64 h12 = -dn / p;
        wald_ci_from_hessian_2d(r, p, h11, h22, h12, CiTransform::kLog, CiTransform::kLogit, alpha,
                                out);
    }
    return StatsStatus::kOk;
}

StatsStatus evfit(ArrayRef<f64> x, ArrayRef<f64> cens, ArrayRef<f64> freq, f64 alpha,
                  bool compute_ci, Fit2Result& out) noexcept {
    const usize n = x.size();
    if (n < 2) {
        return StatsStatus::kInvalidArgument;
    }
    if ((!cens.empty() && cens.size() != n) || (!freq.empty() && freq.size() != n)) {
        return StatsStatus::kInvalidArgument;
    }
    bool freq_trivial = true;
    f64 sum_fw = 0.0;
    f64 sum_uw = 0.0;
    f64 sum_uwx = 0.0;
    f64 sum_uwxx = 0.0;
    for (usize i = 0; i < n; ++i) {
        if (!is_finite(x[i])) {
            return StatsStatus::kInvalidArgument;
        }
        const f64 f = freq.empty() ? 1.0 : freq[i];
        if (!(f >= 0.0)) {
            return StatsStatus::kInvalidArgument;
        }
        if (f != 1.0) {
            freq_trivial = false;
        }
        const f64 u = (cens.empty() || cens[i] == 0.0) ? 1.0 : 0.0;
        sum_fw += f;
        sum_uw += f * u;
        sum_uwx += f * u * x[i];
        sum_uwxx += f * u * x[i] * x[i];
    }
    if (sum_uw < 2.0) {
        return StatsStatus::kInvalidArgument;
    }

    const bool has_cens = (sum_uw < sum_fw);
    const f64 mean_uw = sum_uwx / sum_uw;
    f64 var_uw = sum_uwxx / sum_uw - mean_uw * mean_uw;
    if (!(var_uw > 0.0)) {
        if (!has_cens && freq_trivial) {
            return StatsStatus::kSingularMatrix;
        }
        var_uw = 1.0;
    }

    f64 sigma = sqrt(6.0 * var_uw) / kPi;
    if (!(sigma > 0.0) || !is_finite(sigma)) {
        sigma = 1.0;
    }
    f64 mu = mean_uw + kEulerGamma * sigma;

    if (!has_cens && freq_trivial) {
        f64 sum = 0.0;
        f64 x_max = x[0];
        for (usize i = 0; i < n; ++i) {
            sum += x[i];
            if (x[i] > x_max) {
                x_max = x[i];
            }
        }
        const f64 mean_v = sum / as<f64>(n);
        for (i32 it = 0; it < 100; ++it) {
            f64 t_sum = 0.0;
            f64 u_sum = 0.0;
            f64 v_sum = 0.0;
            for (usize i = 0; i < n; ++i) {
                const f64 e = exp((x[i] - x_max) / sigma);
                t_sum += e;
                u_sum += x[i] * e;
                v_sum += x[i] * x[i] * e;
            }
            const f64 mean_w = u_sum / t_sum;
            const f64 var_w = v_sum / t_sum - mean_w * mean_w;
            const f64 f_val = mean_w - mean_v - sigma;
            const f64 fp = -var_w / (sigma * sigma) - 1.0;
            if (!(abs(fp) > 1e-300)) {
                break;
            }
            const f64 step = f_val / fp;
            sigma = (sigma - step > 0.0) ? (sigma - step) : (sigma * 0.5);
            if (abs(step) < 1e-12 * max(sigma, 1.0)) {
                break;
            }
        }
        f64 t_sum = 0.0;
        for (usize i = 0; i < n; ++i) {
            t_sum += exp((x[i] - x_max) / sigma);
        }
        mu = x_max + sigma * (log(t_sum) - log(as<f64>(n)));
    } else {
        auto grad = [&](f64 m, f64 s, f64& gm, f64& gs) noexcept {
            const f64 hm = max(abs(m), 1.0) * 1e-6;
            const f64 hs = max(abs(s), 1.0) * 1e-6;
            gm = (nll_ev_full(m + hm, s, x, cens, freq) - nll_ev_full(m - hm, s, x, cens, freq)) /
                 (2.0 * hm);
            gs = (nll_ev_full(m, s + hs, x, cens, freq) - nll_ev_full(m, s - hs, x, cens, freq)) /
                 (2.0 * hs);
        };
        f64 f_cur = nll_ev_full(mu, sigma, x, cens, freq);
        for (i32 it = 0; it < 100; ++it) {
            f64 gm = 0.0, gs = 0.0;
            grad(mu, sigma, gm, gs);
            if (sqrt(gm * gm + gs * gs) < 1e-12) {
                break;
            }
            const f64 hm = max(abs(mu), 1.0) * 1e-5;
            const f64 hs = max(abs(sigma), 1.0) * 1e-5;
            f64 gm_mp = 0.0, gs_mp = 0.0, gm_mm = 0.0, gs_mm = 0.0;
            f64 gm_sp = 0.0, gs_sp = 0.0, gm_sm = 0.0, gs_sm = 0.0;
            grad(mu + hm, sigma, gm_mp, gs_mp);
            grad(mu - hm, sigma, gm_mm, gs_mm);
            grad(mu, sigma + hs, gm_sp, gs_sp);
            grad(mu, sigma - hs, gm_sm, gs_sm);
            const f64 hmm = (gm_mp - gm_mm) / (2.0 * hm);
            const f64 hss = (gs_sp - gs_sm) / (2.0 * hs);
            const f64 hms = 0.5 * ((gm_sp - gm_sm) / (2.0 * hs) + (gs_mp - gs_mm) / (2.0 * hm));
            const f64 det = hmm * hss - hms * hms;
            if (abs(det) < 1e-300) {
                break;
            }
            const f64 dm = (-gm * hss + gs * hms) / det;
            const f64 ds = (-gs * hmm + gm * hms) / det;
            f64 step = 1.0;
            bool ok = false;
            for (i32 bt = 0; bt < 30; ++bt) {
                const f64 mu_new = mu + step * dm;
                const f64 s_new = sigma + step * ds;
                if (s_new > 0.0) {
                    const f64 f_new = nll_ev_full(mu_new, s_new, x, cens, freq);
                    if (is_finite(f_new) && f_new < f_cur - 1e-15) {
                        mu = mu_new;
                        sigma = s_new;
                        f_cur = f_new;
                        ok = true;
                        break;
                    }
                }
                step *= 0.5;
            }
            if (!ok) {
                break;
            }
            if (abs(step * dm) + abs(step * ds) < 1e-12 * (abs(mu) + abs(sigma) + 1.0)) {
                break;
            }
        }
    }

    out = Fit2Result{mu, sigma, 0.0, 0.0, 0.0, 0.0};
    if (compute_ci) {
        wald_ci_fd_2d([&](f64 m, f64 s) noexcept { return nll_ev_full(m, s, x, cens, freq); }, mu,
                      sigma, CiTransform::kLinear, CiTransform::kLog, alpha, out);
    }
    return StatsStatus::kOk;
}

StatsStatus gpfit(ArrayRef<f64> x, f64 alpha, bool compute_ci, Fit2Result& out, Allocator* alloc) {
    const usize n = x.size();
    if (n < 2) {
        return StatsStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n; ++i) {
        if (!is_finite(x[i]) || x[i] < 0.0) {
            return StatsStatus::kInvalidArgument;
        }
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> xs(x, scratch.allocator());
    sort(xs.as_mut_ref());
    f64 sum_x = 0.0;
    for (usize i = 0; i < n; ++i) {
        sum_x += xs[i];
    }
    const f64 dn = as<f64>(n);
    const f64 beta0 = sum_x / dn;
    f64 beta1 = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 f_pos = (as<f64>(i + 1) - 0.35) / dn;
        beta1 += (1.0 - f_pos) * xs[i];
    }
    beta1 /= dn;

    const f64 denom = beta0 - 2.0 * beta1;
    f64 k_hat = 0.0;
    f64 sigma_hat = beta0;
    if (abs(denom) >= 1e-300 && is_finite(denom)) {
        k_hat = 2.0 - beta0 / denom;
        sigma_hat = 2.0 * beta0 * beta1 / denom;
        if (!(sigma_hat > 0.0) || !is_finite(sigma_hat)) {
            k_hat = 0.0;
            sigma_hat = beta0;
        }
    }

    const f64 x_max = xs[n - 1];
    f64 f0 = nll_gp(k_hat, sigma_hat, x);
    if (!is_finite(f0)) {
        sigma_hat = beta0;
        k_hat = 0.0;
        f0 = nll_gp(k_hat, sigma_hat, x);
    }

    for (i32 it = 0; it < 100; ++it) {
        f64 gk = 0.0, gs = 0.0;
        gp_nll_grad(k_hat, sigma_hat, x, gk, gs);
        if (sqrt(gk * gk + gs * gs) < 1e-12) {
            break;
        }
        const f64 hk = max(abs(k_hat), 1.0) * 1e-6;
        const f64 hs = max(abs(sigma_hat), 1.0) * 1e-6;
        f64 gk_kp = 0.0, gs_kp = 0.0, gk_km = 0.0, gs_km = 0.0;
        f64 gk_sp = 0.0, gs_sp = 0.0, gk_sm = 0.0, gs_sm = 0.0;
        gp_nll_grad(k_hat + hk, sigma_hat, x, gk_kp, gs_kp);
        gp_nll_grad(k_hat - hk, sigma_hat, x, gk_km, gs_km);
        gp_nll_grad(k_hat, sigma_hat + hs, x, gk_sp, gs_sp);
        gp_nll_grad(k_hat, sigma_hat - hs, x, gk_sm, gs_sm);
        const f64 hkk = (gk_kp - gk_km) / (2.0 * hk);
        const f64 hss = (gs_sp - gs_sm) / (2.0 * hs);
        const f64 hks = 0.5 * ((gk_sp - gk_sm) / (2.0 * hs) + (gs_kp - gs_km) / (2.0 * hk));
        const f64 det = hkk * hss - hks * hks;
        if (abs(det) < 1e-300) {
            break;
        }
        const f64 dk = (-gk * hss + gs * hks) / det;
        const f64 ds = (-gs * hkk + gk * hks) / det;
        f64 step = 1.0;
        bool improved = false;
        for (i32 bt = 0; bt < 30; ++bt) {
            const f64 k_new = k_hat + step * dk;
            const f64 s_new = sigma_hat + step * ds;
            if (s_new > 0.0 && (k_new >= 0.0 || -k_new * x_max / s_new < 1.0 - 1e-12)) {
                const f64 f_new = nll_gp(k_new, s_new, x);
                if (is_finite(f_new) && f_new < f0 - 1e-15) {
                    k_hat = k_new;
                    sigma_hat = s_new;
                    f0 = f_new;
                    improved = true;
                    break;
                }
            }
            step *= 0.5;
        }
        if (!improved) {
            break;
        }
        if (abs(step * dk) + abs(step * ds) < 1e-12 * (abs(k_hat) + abs(sigma_hat) + 1.0)) {
            break;
        }
    }

    out = Fit2Result{k_hat, sigma_hat, 0.0, 0.0, 0.0, 0.0};
    if (compute_ci) {
        wald_ci_fd_2d([&](f64 k, f64 s) noexcept { return nll_gp(k, s, x); }, k_hat, sigma_hat,
                      CiTransform::kLinear, CiTransform::kLog, alpha, out);
    }
    return StatsStatus::kOk;
}

StatsStatus gevfit(ArrayRef<f64> x, f64 alpha, bool compute_ci, Fit3Result& out, Allocator* alloc) {
    const usize n = x.size();
    if (n < 3) {
        return StatsStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n; ++i) {
        if (!is_finite(x[i])) {
            return StatsStatus::kInvalidArgument;
        }
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> xs(x, scratch.allocator());
    sort(xs.as_mut_ref());
    f64 b0 = 0.0;
    f64 b1 = 0.0;
    f64 b2 = 0.0;
    const f64 dn = as<f64>(n);
    for (usize i = 0; i < n; ++i) {
        const f64 fhat = (as<f64>(i + 1) - 0.35) / dn;
        b0 += xs[i];
        b1 += xs[i] * fhat;
        b2 += xs[i] * fhat * fhat;
    }
    b0 /= dn;
    b1 /= dn;
    b2 /= dn;

    constexpr f64 kLn2Ln3 = 0.6309297535714574;
    const f64 denom = 3.0 * b2 - b0;
    f64 k_init = 0.0;
    if (abs(denom) > 1e-300) {
        const f64 c = (2.0 * b1 - b0) / denom - kLn2Ln3;
        k_init = 7.8590 * c + 2.9554 * c * c;
    }
    f64 sigma_init = 1.0;
    if (abs(k_init) < 1e-6) {
        sigma_init = (2.0 * b1 - b0) / log(2.0);
    } else {
        const f64 gam_1pk = specfun::gamma(1.0 + k_init);
        const f64 one_m_2nk = 1.0 - pow(2.0, -k_init);
        sigma_init = k_init * (2.0 * b1 - b0) / (gam_1pk * one_m_2nk);
    }
    if (!(sigma_init > 0.0) || !is_finite(sigma_init)) {
        sigma_init = max(b0 * 0.5, 1.0);
    }
    f64 mu_init = 0.0;
    if (abs(k_init) < 1e-6) {
        mu_init = b0 - sigma_init * kEulerGamma;
    } else {
        mu_init = b0 - sigma_init * (1.0 - specfun::gamma(1.0 + k_init)) / k_init;
    }

    auto feasible = [&](f64 kv, f64 sv, f64 mv) noexcept -> bool {
        if (!(sv > 0.0)) {
            return false;
        }
        if (abs(kv) < 1e-10) {
            return true;
        }
        for (usize i = 0; i < n; ++i) {
            if (1.0 + kv * (x[i] - mv) / sv <= 0.0) {
                return false;
            }
        }
        return true;
    };

    if (!feasible(k_init, sigma_init, mu_init)) {
        k_init = 0.0;
        sigma_init = sqrt(6.0) / kPi;
        mu_init = b0 - kEulerGamma * sigma_init;
    }

    f64 k = k_init;
    f64 sigma = sigma_init;
    f64 mu = mu_init;
    f64 f_cur = nll_gev(k, sigma, mu, x);
    if (!is_finite(f_cur)) {
        k = 0.0;
        sigma = 1.0;
        mu = 0.0;
        f_cur = nll_gev(k, sigma, mu, x);
    }

    auto nll = [&](f64 kv, f64 sv, f64 mv) noexcept { return nll_gev(kv, sv, mv, x); };
    for (i32 it = 0; it < 200; ++it) {
        const f64 hk = max(abs(k), 0.5) * 1e-5;
        const f64 hs = max(abs(sigma), 1.0) * 1e-5;
        const f64 hm = max(abs(mu), 1.0) * 1e-5;
        const f64 gk = (nll(k + hk, sigma, mu) - nll(k - hk, sigma, mu)) / (2.0 * hk);
        const f64 gs = (nll(k, sigma + hs, mu) - nll(k, sigma - hs, mu)) / (2.0 * hs);
        const f64 gm = (nll(k, sigma, mu + hm) - nll(k, sigma, mu - hm)) / (2.0 * hm);
        if (sqrt(gk * gk + gs * gs + gm * gm) < 1e-12) {
            break;
        }
        const f64 hkk = (nll(k + hk, sigma, mu) - 2.0 * f_cur + nll(k - hk, sigma, mu)) / (hk * hk);
        const f64 hss = (nll(k, sigma + hs, mu) - 2.0 * f_cur + nll(k, sigma - hs, mu)) / (hs * hs);
        const f64 hmm = (nll(k, sigma, mu + hm) - 2.0 * f_cur + nll(k, sigma, mu - hm)) / (hm * hm);
        const f64 hks = (nll(k + hk, sigma + hs, mu) - nll(k + hk, sigma - hs, mu) -
                         nll(k - hk, sigma + hs, mu) + nll(k - hk, sigma - hs, mu)) /
                        (4.0 * hk * hs);
        const f64 hkm = (nll(k + hk, sigma, mu + hm) - nll(k + hk, sigma, mu - hm) -
                         nll(k - hk, sigma, mu + hm) + nll(k - hk, sigma, mu - hm)) /
                        (4.0 * hk * hm);
        const f64 hsm = (nll(k, sigma + hs, mu + hm) - nll(k, sigma + hs, mu - hm) -
                         nll(k, sigma - hs, mu + hm) + nll(k, sigma - hs, mu - hm)) /
                        (4.0 * hs * hm);
        const f64 det = hkk * (hss * hmm - hsm * hsm) - hks * (hks * hmm - hsm * hkm) +
                        hkm * (hks * hsm - hss * hkm);
        if (abs(det) < 1e-300) {
            break;
        }
        const f64 b1n = -gk;
        const f64 b2n = -gs;
        const f64 b3n = -gm;
        const f64 dk = (b1n * (hss * hmm - hsm * hsm) - hks * (b2n * hmm - hsm * b3n) +
                        hkm * (b2n * hsm - hss * b3n)) /
                       det;
        const f64 ds = (hkk * (b2n * hmm - hsm * b3n) - b1n * (hks * hmm - hsm * hkm) +
                        hkm * (hks * b3n - b2n * hkm)) /
                       det;
        const f64 dm = (hkk * (hss * b3n - b2n * hsm) - hks * (hks * b3n - b2n * hkm) +
                        b1n * (hks * hsm - hss * hkm)) /
                       det;
        f64 sc = 1.0;
        bool ok = false;
        for (i32 bt = 0; bt < 30; ++bt) {
            const f64 kn = k + sc * dk;
            const f64 sn = sigma + sc * ds;
            const f64 mn = mu + sc * dm;
            if (sn > 0.0 && feasible(kn, sn, mn)) {
                const f64 f_new = nll(kn, sn, mn);
                if (is_finite(f_new) && f_new < f_cur - 1e-15) {
                    k = kn;
                    sigma = sn;
                    mu = mn;
                    f_cur = f_new;
                    ok = true;
                    break;
                }
            }
            sc *= 0.5;
        }
        if (!ok) {
            break;
        }
        if (abs(sc * dk) + abs(sc * ds) + abs(sc * dm) <
            1e-12 * (abs(k) + abs(sigma) + abs(mu) + 1.0)) {
            break;
        }
    }

    out = Fit3Result{k, sigma, mu, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (compute_ci) {
        const f64 qnan = Limits<f64>::quiet_nan();
        const f64 hk = max(abs(k), 0.5) * 1e-5;
        const f64 hs = max(abs(sigma), 1.0) * 1e-5;
        const f64 hm = max(abs(mu), 1.0) * 1e-5;
        const f64 f00 = nll(k, sigma, mu);
        const f64 hkk = (nll(k + hk, sigma, mu) - 2.0 * f00 + nll(k - hk, sigma, mu)) / (hk * hk);
        const f64 hss = (nll(k, sigma + hs, mu) - 2.0 * f00 + nll(k, sigma - hs, mu)) / (hs * hs);
        const f64 hmm = (nll(k, sigma, mu + hm) - 2.0 * f00 + nll(k, sigma, mu - hm)) / (hm * hm);
        const f64 hks = (nll(k + hk, sigma + hs, mu) - nll(k + hk, sigma - hs, mu) -
                         nll(k - hk, sigma + hs, mu) + nll(k - hk, sigma - hs, mu)) /
                        (4.0 * hk * hs);
        const f64 hkm = (nll(k + hk, sigma, mu + hm) - nll(k + hk, sigma, mu - hm) -
                         nll(k - hk, sigma, mu + hm) + nll(k - hk, sigma, mu - hm)) /
                        (4.0 * hk * hm);
        const f64 hsm = (nll(k, sigma + hs, mu + hm) - nll(k, sigma + hs, mu - hm) -
                         nll(k, sigma - hs, mu + hm) + nll(k, sigma - hs, mu - hm)) /
                        (4.0 * hs * hm);
        const f64 det = hkk * (hss * hmm - hsm * hsm) - hks * (hks * hmm - hsm * hkm) +
                        hkm * (hks * hsm - hss * hkm);
        if (abs(det) < 1e-300) {
            out.ci1_lo = out.ci1_hi = out.ci2_lo = out.ci2_hi = out.ci3_lo = out.ci3_hi = qnan;
            return StatsStatus::kOk;
        }
        const f64 se_k = sqrt(max((hss * hmm - hsm * hsm) / det, 0.0));
        const f64 se_s = sqrt(max((hkk * hmm - hkm * hkm) / det, 0.0));
        const f64 se_m = sqrt(max((hkk * hss - hks * hks) / det, 0.0));
        const f64 z = norm_inv(1.0 - 0.5 * alpha, 0.0, 1.0);
        const f64 se_log_s = (sigma > 0.0) ? (se_s / sigma) : 0.0;
        out.ci1_lo = k - z * se_k;
        out.ci1_hi = k + z * se_k;
        out.ci2_lo = sigma * exp(-z * se_log_s);
        out.ci2_hi = sigma * exp(z * se_log_s);
        out.ci3_lo = mu - z * se_m;
        out.ci3_hi = mu + z * se_m;
    }
    return StatsStatus::kOk;
}

} // namespace keel::stats
