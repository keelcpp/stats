// keelcpp/stats/src/mvdist.cpp
// Pure zero-STL closed-form multivariate densities: mvnpdf, mvtpdf, mnpdf.

#include "keel/stats.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/linalg.h"
#include "keel/math.h"
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

} // namespace keel::stats
