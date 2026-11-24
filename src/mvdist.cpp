// keelcpp/stats/src/mvdist.cpp
// Pure zero-STL closed-form multivariate densities: mvnpdf, mvtpdf, mnpdf.

#include "keel/stats.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/linalg.h"
#include "keel/math.h"
#include "keel/random.h"
#include "keel/specfun.h"

namespace keel::stats {

namespace {

constexpr f64 kPi = 3.14159265358979323846;
constexpr f64 kLog2Pi = 1.8378770664093454835606594728112352;

bool psd_quad_forms(ArrayRef<f64> x0_col_major, usize n, usize d, ArrayRef<f64> sig_col_major,
                    Array<f64>& quad, f64& log_half_det, bool& degenerate, Allocator* alloc) {
    Array<f64> sym(d * d, 0.0, alloc);
    for (usize j = 0; j < d; ++j) {
        for (usize i = 0; i < d; ++i) {
            sym[i + j * d] = 0.5 * (sig_col_major[i + j * d] + sig_col_major[j + i * d]);
        }
    }
    Array<f64> evec(d * d, 0.0, alloc);
    Array<f64> eval(d, 0.0, alloc);
    linalg::eig_sym(sym.data(), d, evec.data(), eval.data(), alloc);

    f64 max_abs = 0.0;
    for (usize i = 0; i < d; ++i) {
        max_abs = max(max_abs, abs(eval[i]));
    }
    const f64 tol = as<f64>(d) * Limits<f64>::kEpsilon * max(max_abs, 1.0);
    for (usize i = 0; i < d; ++i) {
        if (eval[i] < -tol) {
            return false;
        }
    }

    log_half_det = 0.0;
    degenerate = false;
    for (usize i = 0; i < d; ++i) {
        if (eval[i] <= tol) {
            degenerate = true;
        }
        log_half_det += 0.5 * log(max(eval[i], 0.0));
    }

    quad.assign(n, 0.0);
    for (usize i = 0; i < d; ++i) {
        if (eval[i] <= tol) {
            continue;
        }
        for (usize r = 0; r < n; ++r) {
            f64 s = 0.0;
            for (usize j = 0; j < d; ++j) {
                s += evec[j + i * d] * x0_col_major[r + j * n];
            }
            quad[r] += s * s / eval[i];
        }
    }
    return true;
}

} // namespace

StatsStatus mvnpdf_full(ArrayRef<f64> x0_col_major, usize n, usize d, ArrayRef<f64> sigma_col_major,
                        MutArrayRef<f64> out, Allocator* alloc) {
    if (d == 0 || x0_col_major.size() < n * d || sigma_col_major.size() < d * d || out.size() < n) {
        return StatsStatus::kInvalidArgument;
    }
    if (n == 0) {
        return StatsStatus::kOk;
    }
    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> quad(scratch.allocator());
    f64 log_half_det = 0.0;
    bool degenerate = false;
    if (!psd_quad_forms(x0_col_major, n, d, sigma_col_major, quad, log_half_det, degenerate,
                        scratch.allocator())) {
        return StatsStatus::kSingularMatrix;
    }
    if (degenerate) {
        for (usize r = 0; r < n; ++r) {
            out[r] = 0.0;
        }
        return StatsStatus::kOk;
    }
    const f64 base = -0.5 * as<f64>(d) * kLog2Pi;
    for (usize r = 0; r < n; ++r) {
        out[r] = exp(base - log_half_det - 0.5 * quad[r]);
    }
    return StatsStatus::kOk;
}

StatsStatus mvnpdf_diag(ArrayRef<f64> x0_col_major, usize n, usize d, ArrayRef<f64> diag_var,
                        MutArrayRef<f64> out) noexcept {
    if (d == 0 || x0_col_major.size() < n * d || diag_var.size() < d || out.size() < n) {
        return StatsStatus::kInvalidArgument;
    }
    f64 log_half_det = 0.0;
    for (usize j = 0; j < d; ++j) {
        if (!(diag_var[j] > 0.0)) {
            return StatsStatus::kInvalidArgument;
        }
        log_half_det += 0.5 * log(diag_var[j]);
    }
    const f64 base = -0.5 * as<f64>(d) * kLog2Pi;
    for (usize r = 0; r < n; ++r) {
        f64 q = 0.0;
        for (usize j = 0; j < d; ++j) {
            const f64 dx = x0_col_major[r + j * n];
            q += dx * dx / diag_var[j];
        }
        out[r] = exp(base - log_half_det - 0.5 * q);
    }
    return StatsStatus::kOk;
}

StatsStatus mvtpdf_full(ArrayRef<f64> x_col_major, usize n, usize d, ArrayRef<f64> corr_col_major,
                        ArrayRef<f64> df_per_row, MutArrayRef<f64> out, Allocator* alloc) {
    if (d == 0 || x_col_major.size() < n * d || corr_col_major.size() < d * d ||
        df_per_row.size() < n || out.size() < n) {
        return StatsStatus::kInvalidArgument;
    }
    for (usize r = 0; r < n; ++r) {
        if (!(df_per_row[r] > 0.0)) {
            return StatsStatus::kInvalidArgument;
        }
    }
    if (n == 0) {
        return StatsStatus::kOk;
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> cm(ArrayRef<f64>(corr_col_major.data(), d * d), scratch.allocator());
    Array<f64> sq(d, 0.0, scratch.allocator());
    bool unit_diag = true;
    for (usize i = 0; i < d; ++i) {
        const f64 diag = cm[i + i * d];
        if (!(diag > 0.0)) {
            return StatsStatus::kSingularMatrix;
        }
        sq[i] = sqrt(diag);
        if (diag != 1.0) {
            unit_diag = false;
        }
    }
    if (!unit_diag) {
        for (usize j = 0; j < d; ++j) {
            for (usize i = 0; i < d; ++i) {
                cm[i + j * d] /= sq[i] * sq[j];
            }
        }
    }

    Array<f64> quad(scratch.allocator());
    f64 log_half_det = 0.0;
    bool degenerate = false;
    if (!psd_quad_forms(x_col_major, n, d, cm.as_ref(), quad, log_half_det, degenerate,
                        scratch.allocator())) {
        return StatsStatus::kSingularMatrix;
    }

    const f64 dd = as<f64>(d);
    for (usize r = 0; r < n; ++r) {
        const f64 nu = df_per_row[r];
        const f64 logpdf = specfun::gammaln(0.5 * (nu + dd)) - specfun::gammaln(0.5 * nu) -
                           0.5 * dd * log(nu * kPi) - log_half_det -
                           0.5 * (nu + dd) * log1p(quad[r] / nu);
        out[r] = exp(logpdf);
    }
    return StatsStatus::kOk;
}

StatsStatus mnpdf_rows(ArrayRef<f64> x_col_major, usize m, usize k, ArrayRef<f64> p_col_major,
                       usize mm, MutArrayRef<f64> out) noexcept {
    if (k == 0 || (mm != 1 && mm != m) || x_col_major.size() < m * k ||
        p_col_major.size() < mm * k || out.size() < m) {
        return StatsStatus::kInvalidArgument;
    }
    const f64 eps_d = Limits<f64>::kEpsilon;
    const f64 qnan = Limits<f64>::quiet_nan();

    for (usize r = 0; r < m; ++r) {
        const usize pr = (mm == 1) ? 0 : r;
        f64 n_tot = 0.0;
        bool x_bad = false;
        for (usize j = 0; j < k; ++j) {
            const f64 v = x_col_major[r + j * m];
            n_tot += v;
            if (v < 0.0 || v != round(v)) {
                x_bad = true;
            }
        }
        f64 p_sum = 0.0;
        bool p_bad = false;
        for (usize j = 0; j < k; ++j) {
            const f64 v = p_col_major[pr + j * mm];
            if (v < 0.0 || v > 1.0) {
                p_bad = true;
            }
            p_sum += v;
        }
        if (abs(p_sum - 1.0) > as<f64>(k) * eps_d) {
            p_bad = true;
        }
        const bool n_bad = (n_tot < 0.0) || (round(n_tot) != n_tot);
        if (p_bad || n_bad) {
            out[r] = qnan;
            continue;
        }
        if (x_bad) {
            out[r] = 0.0;
            continue;
        }
        f64 lp = specfun::gammaln(n_tot + 1.0);
        for (usize j = 0; j < k; ++j) {
            const f64 xv = x_col_major[r + j * m];
            lp -= specfun::gammaln(xv + 1.0);
            if (xv > 0.0) {
                lp += xv * log(p_col_major[pr + j * mm]);
            }
        }
        out[r] = exp(lp);
    }
    return StatsStatus::kOk;
}

// ============================================================================
// Covariance Factorization & Multivariate CDFs (cholcov, corrcov, mvncdf, mvtcdf)
// ============================================================================

StatsStatus cholcov(ArrayRef<f64> sigma_col_major, usize n, CholCovResult& out, Allocator* alloc) {
    if (sigma_col_major.size() < n * n) {
        return StatsStatus::kInvalidArgument;
    }
    out.t_col_major.clear();
    out.rows = 0;
    out.cols = n;
    out.non_pd_count = 0;
    if (n == 0) {
        return StatsStatus::kOk;
    }

    // 1. Try positive-definite upper Cholesky R'*R = sigma.
    Array<f64> chol_buf(n * n, 0.0, alloc);
    const usize fail_idx = linalg::cholesky_upper(sigma_col_major.data(), chol_buf.data(), n);
    if (fail_idx == 0) {
        out.t_col_major = move(chol_buf);
        out.rows = n;
        out.cols = n;
        out.non_pd_count = 0;
        return StatsStatus::kOk;
    }

    // 2. Fall back to symmetric eigendecomposition.
    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> sym(n * n, 0.0, scratch.allocator());
    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            sym[i + j * n] = 0.5 * (sigma_col_major[i + j * n] + sigma_col_major[j + i * n]);
        }
    }
    Array<f64> evec(n * n, 0.0, scratch.allocator());
    Array<f64> eval(n, 0.0, scratch.allocator());
    linalg::eig_sym(sym.data(), n, evec.data(), eval.data(), scratch.allocator());

    f64 max_abs_d = 0.0;
    for (usize i = 0; i < n; ++i) {
        max_abs_d = max(max_abs_d, abs(eval[i]));
    }
    const f64 tol = pow(Limits<f64>::kEpsilon, 0.75) * max_abs_d;

    usize neg_cnt = 0;
    for (usize i = 0; i < n; ++i) {
        if (eval[i] <= -tol) {
            ++neg_cnt;
        }
    }
    if (neg_cnt > 0) {
        out.non_pd_count = neg_cnt;
        out.rows = 0;
        out.cols = 0;
        return StatsStatus::kOk;
    }

    Array<usize> keep(scratch.allocator());
    keep.reserve(n);
    for (usize i = 0; i < n; ++i) {
        if (eval[i] > tol) {
            keep.push_back(i);
        }
    }
    const usize r = keep.size();
    out.t_col_major = Array<f64>(r * n, 0.0, alloc);
    for (usize k = 0; k < r; ++k) {
        const usize col = keep[k];
        const f64 s = sqrt(eval[col]);
        for (usize j = 0; j < n; ++j) {
            out.t_col_major[k + j * r] = s * evec[j + col * n];
        }
    }
    out.rows = r;
    out.cols = n;
    out.non_pd_count = 0;
    return StatsStatus::kOk;
}

StatsStatus corrcov(ArrayRef<f64> c_col_major, usize n, MutArrayRef<f64> out_r_col_major,
                    MutArrayRef<f64> out_sigma) noexcept {
    if (n == 0 || c_col_major.size() < n * n || out_r_col_major.size() < n * n ||
        out_sigma.size() < n) {
        return StatsStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n; ++i) {
        const f64 v = c_col_major[i + i * n];
        if (v < 0.0) {
            return StatsStatus::kInvalidArgument;
        }
        out_sigma[i] = sqrt(v);
    }
    const f64 nanv = Limits<f64>::quiet_nan();
    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            const f64 denom = out_sigma[i] * out_sigma[j];
            out_r_col_major[i + j * n] = (denom == 0.0) ? nanv : (c_col_major[i + j * n] / denom);
        }
    }
    return StatsStatus::kOk;
}

f64 genz_sov_mvncdf(ArrayRef<f64> l_row_major, usize d, ArrayRef<f64> a, ArrayRef<f64> b,
                    usize max_evals, f64 tol, Allocator* alloc) {
    if (d == 0 || l_row_major.size() < d * d || a.size() < d || b.size() < d) {
        return Limits<f64>::quiet_nan();
    }
    auto eval_phi = [](f64 x) noexcept -> f64 {
        if (is_inf(x)) {
            return (x < 0.0) ? 0.0 : 1.0;
        }
        return 0.5 * erfc(-x * kInvSqrt2);
    };
    const f64 d0 = eval_phi(a[0] / l_row_major[0]);
    const f64 e0 = eval_phi(b[0] / l_row_major[0]);
    const f64 f0 = max(0.0, e0 - d0);
    if (f0 == 0.0 || d <= 1) {
        return f0;
    }

    static constexpr i32 kPrimes[32] = {2,   3,   5,   7,   11,  13,  17,  19,  23,  29,  31,
                                        37,  41,  43,  47,  53,  59,  61,  67,  71,  73,  79,
                                        83,  89,  97,  101, 103, 107, 109, 113, 127, 131};
    ArenaTemp scratch = get_scratch(alloc);
    const usize m = d - 1;
    Array<f64> alpha(m, 0.0, scratch.allocator());
    for (usize k = 0; k < m; ++k) {
        const f64 pk = (k < 32) ? as<f64>(kPrimes[k]) : as<f64>(137 + 2 * k);
        const f64 rt = sqrt(pk);
        alpha[k] = rt - floor(rt);
    }

    Array<f64> y(m, 0.0, scratch.allocator());
    Array<f64> w(m, 0.0, scratch.allocator());
    auto eval_integrand = [&](const Array<f64>& uvec) noexcept -> f64 {
        f64 f = f0;
        f64 dp = d0;
        f64 ep = e0;
        for (usize i = 1; i < d; ++i) {
            f64 u = dp + uvec[i - 1] * (ep - dp);
            if (u < 1e-16) {
                u = 1e-16;
            }
            if (u > 1.0 - 1e-16) {
                u = 1.0 - 1e-16;
            }
            y[i - 1] = norm_inv(u, 0.0, 1.0);
            f64 s = 0.0;
            for (usize k = 0; k < i; ++k) {
                s += l_row_major[i * d + k] * y[k];
            }
            const f64 di = eval_phi((a[i] - s) / l_row_major[i * d + i]);
            const f64 ei = eval_phi((b[i] - s) / l_row_major[i * d + i]);
            const f64 diff = max(0.0, ei - di);
            f *= diff;
            if (f == 0.0) {
                return 0.0;
            }
            dp = di;
            ep = ei;
        }
        return f;
    };

    const usize n_batches = 8;
    const usize pairs_per_batch = max<usize>(64, max_evals / (2 * n_batches));
    random::Mt19937 shift_gen(5489u);
    Array<f64> shift(m, 0.0, scratch.allocator());

    f64 mean_est = 0.0;
    f64 m2 = 0.0;
    usize completed_batches = 0;
    for (usize batch = 0; batch < n_batches; ++batch) {
        for (usize k = 0; k < m; ++k) {
            shift[k] = random::uniform_f64(shift_gen);
        }
        f64 batch_sum = 0.0;
        for (usize p = 1; p <= pairs_per_batch; ++p) {
            const f64 pd = as<f64>(p);
            for (usize k = 0; k < m; ++k) {
                const f64 v = pd * alpha[k] + shift[k];
                w[k] = v - floor(v);
            }
            const f64 f1 = eval_integrand(w);
            for (usize k = 0; k < m; ++k) {
                w[k] = 1.0 - w[k];
            }
            const f64 f2 = eval_integrand(w);
            batch_sum += 0.5 * (f1 + f2);
        }
        const f64 batch_mean = batch_sum / as<f64>(pairs_per_batch);
        ++completed_batches;
        const f64 delta = batch_mean - mean_est;
        mean_est += delta / as<f64>(completed_batches);
        m2 += delta * (batch_mean - mean_est);
        if (completed_batches >= 4) {
            const f64 se = sqrt(m2 / as<f64>(completed_batches * (completed_batches - 1)));
            if (se <= tol) {
                break;
            }
        }
    }
    return min(1.0, max(0.0, mean_est));
}

f64 mvtcdf_mc(ArrayRef<f64> l_row_major, usize d, f64 df, ArrayRef<f64> xl, ArrayRef<f64> xu,
              i32 n_samples, u64 seed, Allocator* alloc) {
    if (d == 0 || l_row_major.size() < d * d || xl.size() < d || xu.size() < d || !(df > 0.0)) {
        return Limits<f64>::quiet_nan();
    }
    if (d == 1) {
        const f64 fu = is_inf(xu[0]) ? (xu[0] > 0.0 ? 1.0 : 0.0) : t_cdf(xu[0], df);
        const f64 fl = is_inf(xl[0]) ? (xl[0] > 0.0 ? 1.0 : 0.0) : t_cdf(xl[0], df);
        return max(0.0, fu - fl);
    }
    if (n_samples < 2) {
        n_samples = 2;
    }
    if (n_samples % 2 != 0) {
        ++n_samples;
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> z(d, 0.0, scratch.allocator());
    Array<f64> y(d, 0.0, scratch.allocator());
    random::Xoshiro256PlusPlus gen(seed);

    i32 hits = 0;
    const i32 half_n = n_samples / 2;
    for (i32 k = 0; k < half_n; ++k) {
        for (usize j = 0; j < d; ++j) {
            z[j] = random::normal_f64(gen);
        }
        const f64 scale = sqrt(df / random::chi_squared_f64(gen, df));
        // Pass 1: +z
        bool inside = true;
        for (usize j = 0; j < d; ++j) {
            f64 s = 0.0;
            for (usize k2 = 0; k2 <= j; ++k2) {
                s += l_row_major[j * d + k2] * z[k2];
            }
            y[j] = s * scale;
            if (y[j] < xl[j] || y[j] > xu[j]) {
                inside = false;
            }
        }
        if (inside) {
            ++hits;
        }
        // Pass 2: -z (antithetic)
        inside = true;
        for (usize j = 0; j < d; ++j) {
            const f64 yj = -y[j];
            if (yj < xl[j] || yj > xu[j]) {
                inside = false;
                break;
            }
        }
        if (inside) {
            ++hits;
        }
    }
    return as<f64>(hits) / as<f64>(n_samples);
}

} // namespace keel::stats
