#include "keel/stats.h"

#include "keel/algo.h"
#include "keel/linalg.h"
#include "keel/math.h"

namespace keel::stats {

namespace {

constexpr f64 kLog2Pi = 1.8378770664093454835606594728112352;

inline bool mul_overflows(usize a, usize b) noexcept {
    return a != 0u && b > Limits<usize>::kMax / a;
}

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

f64 frob_norm(ArrayRef<f64> m) noexcept {
    f64 s = 0.0;
    for (usize i = 0; i < m.size(); ++i) s += m[i] * m[i];
    return sqrt(s);
}

void proj_psd(ArrayRef<f64> M, usize n, MutArrayRef<f64> X, Allocator* alloc) {
    Array<f64> m_sym(n * n, 0.0, alloc);
    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            m_sym[i + j * n] = 0.5 * (M[i + j * n] + M[j + i * n]);
        }
    }
    Array<f64> V(n * n, 0.0, alloc);
    Array<f64> D(n, 0.0, alloc);
    linalg::eig_sym(m_sym.data(), n, V.data(), D.data(), alloc);

    Array<f64> vs(n * n, 0.0, alloc);
    for (usize j = 0; j < n; ++j) {
        const f64 sd = sqrt(max(D[j], 0.0));
        for (usize i = 0; i < n; ++i) vs[i + j * n] = V[i + j * n] * sd;
    }
    for (usize i = 0; i < n; ++i) {
        for (usize j = 0; j < n; ++j) {
            f64 s = 0.0;
            for (usize k = 0; k < n; ++k) s += vs[i + k * n] * vs[j + k * n];
            X[i + j * n] = s;
        }
    }
}

} // namespace

StatsStatus pcacov(ArrayRef<f64> c_col_major, usize d, MutArrayRef<f64> out_coeff_col_major,
                   MutArrayRef<f64> out_latent, MutArrayRef<f64> out_explained, Allocator* alloc) {
    if (d == 0 || mul_overflows(d, d) || c_col_major.size() != d * d ||
        out_coeff_col_major.size() != d * d || out_latent.size() != d) {
        return StatsStatus::kInvalidArgument;
    }
    if (!out_explained.empty() && out_explained.size() != d) {
        return StatsStatus::kInvalidArgument;
    }
    for (usize i = 0; i < d * d; ++i) {
        if (!is_finite(c_col_major[i])) return StatsStatus::kInvalidArgument;
    }

    Array<f64> sym(d * d, 0.0, alloc);
    for (usize j = 0; j < d; ++j) {
        for (usize i = 0; i < d; ++i) {
            sym[i + j * d] = 0.5 * (c_col_major[i + j * d] + c_col_major[j + i * d]);
        }
    }
    Array<f64> V(d * d, 0.0, alloc);
    Array<f64> evals(d, 0.0, alloc);
    linalg::eig_sym(sym.data(), d, V.data(), evals.data(), alloc);

    Array<usize> order(d, usize{0}, alloc);
    for (usize i = 0; i < d; ++i) order[i] = i;
    keel::sort(order, [&](usize a, usize b) noexcept { return evals[a] > evals[b]; });

    f64 total = 0.0;
    for (usize col = 0; col < d; ++col) {
        const usize src = order[col];
        out_latent[col] = evals[src];
        total += evals[src];
        // Enforce deterministic sign convention: largest-magnitude entry in each column is positive.
        f64 max_abs = -1.0;
        f64 sign = 1.0;
        for (usize r = 0; r < d; ++r) {
            const f64 v = V[r + src * d];
            const f64 av = abs(v);
            if (av > max_abs) {
                max_abs = av;
                sign = (v < 0.0) ? -1.0 : 1.0;
            }
        }
        for (usize r = 0; r < d; ++r) {
            out_coeff_col_major[r + col * d] = sign * V[r + src * d];
        }
    }
    if (!out_explained.empty()) {
        if (total > 0.0) {
            for (usize i = 0; i < d; ++i) out_explained[i] = 100.0 * out_latent[i] / total;
        } else {
            for (usize i = 0; i < d; ++i) out_explained[i] = 0.0;
        }
    }
    return StatsStatus::kOk;
}

StatsStatus pca(ArrayRef<f64> x_col_major, usize n, usize d, MutArrayRef<f64> out_coeff_col_major,
                MutArrayRef<f64> out_score_col_major, MutArrayRef<f64> out_latent,
                MutArrayRef<f64> out_tsq, MutArrayRef<f64> out_explained, MutArrayRef<f64> out_mu,
                Allocator* alloc) {
    if (n < 2 || d == 0 || mul_overflows(n, d) || mul_overflows(d, d) ||
        x_col_major.size() != n * d || out_coeff_col_major.size() != d * d ||
        out_latent.size() != d) {
        return StatsStatus::kInvalidArgument;
    }
    if (!out_score_col_major.empty() && out_score_col_major.size() != n * d) {
        return StatsStatus::kInvalidArgument;
    }
    if (!out_tsq.empty() && out_tsq.size() != n) return StatsStatus::kInvalidArgument;
    if (!out_explained.empty() && out_explained.size() != d) return StatsStatus::kInvalidArgument;
    if (!out_mu.empty() && out_mu.size() != d) return StatsStatus::kInvalidArgument;
    for (usize i = 0; i < n * d; ++i) {
        if (!is_finite(x_col_major[i])) return StatsStatus::kInvalidArgument;
    }

    Array<f64> mu(d, 0.0, alloc);
    for (usize c = 0; c < d; ++c) {
        f64 s = 0.0;
        for (usize r = 0; r < n; ++r) s += x_col_major[r + c * n];
        mu[c] = s / as<f64>(n);
        if (!out_mu.empty()) out_mu[c] = mu[c];
    }

    Array<f64> xc(n * d, 0.0, alloc);
    for (usize c = 0; c < d; ++c) {
        for (usize r = 0; r < n; ++r) xc[r + c * n] = x_col_major[r + c * n] - mu[c];
    }

    Array<f64> cov(d * d, 0.0, alloc);
    const f64 inv_n = 1.0 / as<f64>(n - 1);
    for (usize b = 0; b < d; ++b) {
        for (usize a = b; a < d; ++a) {
            f64 s = 0.0;
            for (usize r = 0; r < n; ++r) s += xc[r + a * n] * xc[r + b * n];
            const f64 v = s * inv_n;
            cov[a + b * d] = v;
            cov[b + a * d] = v;
        }
    }

    const StatsStatus st =
        pcacov(cov.as_ref(), d, out_coeff_col_major, out_latent, out_explained, alloc);
    if (st != StatsStatus::kOk) return st;

    // Sample covariance is PSD; clamp tiny numerical negative eigenvalues to 0.
    f64 total = 0.0;
    for (usize k = 0; k < d; ++k) {
        if (out_latent[k] < 0.0) out_latent[k] = 0.0;
        total += out_latent[k];
    }
    if (!out_explained.empty()) {
        if (total > 0.0) {
            for (usize k = 0; k < d; ++k) out_explained[k] = 100.0 * out_latent[k] / total;
        } else {
            for (usize k = 0; k < d; ++k) out_explained[k] = 0.0;
        }
    }

    const bool need_score = !out_score_col_major.empty() || !out_tsq.empty();
    if (need_score) {
        Array<f64> score_tmp(alloc);
        MutArrayRef<f64> score = out_score_col_major;
        if (score.empty()) {
            score_tmp.resize(n * d, 0.0);
            score = score_tmp.as_mut_ref();
        }
        for (usize k = 0; k < d; ++k) {
            for (usize r = 0; r < n; ++r) {
                f64 s = 0.0;
                for (usize j = 0; j < d; ++j) s += xc[r + j * n] * out_coeff_col_major[j + k * d];
                score[r + k * n] = s;
            }
        }
        if (!out_tsq.empty()) {
            const f64 lat_tol =
                max(1e-15, out_latent[0] * as<f64>(max(n, d)) * Limits<f64>::kEpsilon);
            for (usize r = 0; r < n; ++r) {
                f64 s = 0.0;
                for (usize k = 0; k < d; ++k) {
                    if (out_latent[k] > lat_tol) {
                        const f64 sc = score[r + k * n];
                        s += (sc * sc) / out_latent[k];
                    }
                }
                out_tsq[r] = s;
            }
        }
    }
    return StatsStatus::kOk;
}

StatsStatus pcares(ArrayRef<f64> x_col_major, usize n, usize d, usize ndim,
                   MutArrayRef<f64> out_res_col_major, MutArrayRef<f64> out_recon_col_major,
                   Allocator* alloc) {
    if (n < 2 || d == 0 || ndim > d || mul_overflows(n, d) || x_col_major.size() != n * d ||
        out_res_col_major.size() != n * d || out_recon_col_major.size() != n * d) {
        return StatsStatus::kInvalidArgument;
    }

    Array<f64> coeff(d * d, 0.0, alloc);
    Array<f64> score(n * d, 0.0, alloc);
    Array<f64> latent(d, 0.0, alloc);
    Array<f64> mu(d, 0.0, alloc);
    const StatsStatus st = pca(x_col_major, n, d, coeff.as_mut_ref(), score.as_mut_ref(),
                               latent.as_mut_ref(), {}, {}, mu.as_mut_ref(), alloc);
    if (st != StatsStatus::kOk) return st;

    for (usize j = 0; j < d; ++j) {
        for (usize i = 0; i < n; ++i) {
            f64 r = mu[j];
            for (usize k = 0; k < ndim; ++k) {
                r += score[i + k * n] * coeff[j + k * d];
            }
            out_recon_col_major[i + j * n] = r;
            out_res_col_major[i + j * n] = x_col_major[i + j * n] - r;
        }
    }
    return StatsStatus::kOk;
}

StatsStatus lda_classify(ArrayRef<f64> sample_col_major, usize n_samp,
                         ArrayRef<f64> training_col_major, usize n_tr, usize d,
                         ArrayRef<f64> group_labels, DiscrimType type, LdaResult& out,
                         Allocator* alloc) {
    if (n_tr == 0 || d == 0 || mul_overflows(n_tr, d) || mul_overflows(n_samp, d) ||
        mul_overflows(d, d) || training_col_major.size() != n_tr * d ||
        sample_col_major.size() != n_samp * d || group_labels.size() != n_tr) {
        return StatsStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n_tr; ++i) {
        if (!is_finite(group_labels[i])) return StatsStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n_tr * d; ++i) {
        if (!is_finite(training_col_major[i])) return StatsStatus::kInvalidArgument;
    }

    const bool quadratic =
        (type == DiscrimType::kQuadratic || type == DiscrimType::kDiagQuadratic);
    const bool diagonal =
        (type == DiscrimType::kDiagLinear || type == DiscrimType::kDiagQuadratic);

    Array<f64> uniq(n_tr, 0.0, alloc);
    for (usize i = 0; i < n_tr; ++i) uniq[i] = group_labels[i];
    keel::sort(uniq);
    usize K = 0;
    for (usize i = 0; i < n_tr; ++i) {
        if (K == 0 || uniq[i] != uniq[K - 1]) uniq[K++] = uniq[i];
    }
    uniq.resize(K);

    auto find_class = [&](f64 lbl) noexcept -> usize {
        for (usize k = 0; k < K; ++k) {
            if (uniq[k] == lbl) return k;
        }
        return 0;
    };

    Array<usize> n_k(K, usize{0}, alloc);
    Array<f64> mu(K * d, 0.0, alloc);
    for (usize i = 0; i < n_tr; ++i) {
        const usize k = find_class(group_labels[i]);
        ++n_k[k];
        for (usize j = 0; j < d; ++j) mu[j + k * d] += training_col_major[i + j * n_tr];
    }
    for (usize k = 0; k < K; ++k) {
        for (usize j = 0; j < d; ++j) mu[j + k * d] /= as<f64>(n_k[k]);
    }

    const usize blocks = quadratic ? K : 1;
    Array<f64> sigma(blocks * d * d, 0.0, alloc);
    auto sig_block = [&](usize k) noexcept -> f64* {
        return sigma.data() + (quadratic ? k * d * d : 0);
    };

    Array<f64> diff(d, 0.0, alloc);
    for (usize i = 0; i < n_tr; ++i) {
        const usize k = find_class(group_labels[i]);
        for (usize j = 0; j < d; ++j) {
            diff[j] = training_col_major[i + j * n_tr] - mu[j + k * d];
        }
        f64* S = sig_block(k);
        for (usize a = 0; a < d; ++a) {
            for (usize b = 0; b < d; ++b) S[a + b * d] += diff[a] * diff[b];
        }
    }
    if (quadratic) {
        for (usize k = 0; k < K; ++k) {
            f64* S = sig_block(k);
            const f64 denom = (n_k[k] > 1) ? as<f64>(n_k[k] - 1) : 1.0;
            for (usize i = 0; i < d * d; ++i) S[i] /= denom;
        }
    } else {
        const f64 denom = (n_tr > K) ? as<f64>(n_tr - K) : 1.0;
        for (usize i = 0; i < d * d; ++i) sigma[i] /= denom;
    }

    if (diagonal) {
        for (usize k = 0; k < blocks; ++k) {
            f64* S = sig_block(k);
            for (usize a = 0; a < d; ++a) {
                for (usize b = 0; b < d; ++b) {
                    if (a != b) S[a + b * d] = 0.0;
                }
            }
        }
    }

    Array<f64> l_blocks(blocks * d * d, 0.0, alloc);
    Array<f64> sum_log_diag(blocks, 0.0, alloc);
    for (usize k = 0; k < blocks; ++k) {
        f64* L = l_blocks.data() + k * d * d;
        if (!cholesky_col(sig_block(k), L, d)) return StatsStatus::kSingularMatrix;
        f64 s = 0.0;
        for (usize i = 0; i < d; ++i) s += log(L[i + i * d]);
        sum_log_diag[k] = s;
    }

    Array<f64> log_prior(K, 0.0, alloc);
    for (usize k = 0; k < K; ++k) log_prior[k] = log(as<f64>(n_k[k]) / as<f64>(n_tr));

    out.num_classes = K;
    out.pred.resize(n_samp);
    out.posterior.resize(n_samp * K);
    out.logp.resize(n_samp);

    Array<f64> lp(K, 0.0, alloc);
    Array<f64> z(d, 0.0, alloc);
    Array<f64> x(d, 0.0, alloc);
    for (usize r = 0; r < n_samp; ++r) {
        for (usize j = 0; j < d; ++j) x[j] = sample_col_major[r + j * n_samp];
        f64 max_lp = -Limits<f64>::infinity();
        usize k_best = 0;
        for (usize k = 0; k < K; ++k) {
            const f64* L = l_blocks.data() + (quadratic ? k * d * d : 0);
            const f64 sld = quadratic ? sum_log_diag[k] : sum_log_diag[0];
            for (usize j = 0; j < d; ++j) diff[j] = x[j] - mu[j + k * d];
            fwd_solve_col(L, z.data(), diff.data(), d);
            f64 q = 0.0;
            for (usize j = 0; j < d; ++j) q += z[j] * z[j];
            const f64 log_fk = -0.5 * as<f64>(d) * kLog2Pi - sld - 0.5 * q;
            lp[k] = log_fk + log_prior[k];
            if (lp[k] > max_lp) {
                max_lp = lp[k];
                k_best = k;
            }
        }
        out.pred[r] = uniq[k_best];
        f64 sum_exp = 0.0;
        for (usize k = 0; k < K; ++k) sum_exp += exp(lp[k] - max_lp);
        const f64 log_sum = max_lp + log(sum_exp);
        for (usize k = 0; k < K; ++k) out.posterior[r + k * n_samp] = exp(lp[k] - log_sum);
        out.logp[r] = log_sum;
    }

    f64 mis = 0.0;
    for (usize i = 0; i < n_tr; ++i) {
        for (usize j = 0; j < d; ++j) x[j] = training_col_major[i + j * n_tr];
        f64 max_lp = -Limits<f64>::infinity();
        usize k_best = 0;
        for (usize k = 0; k < K; ++k) {
            const f64* L = l_blocks.data() + (quadratic ? k * d * d : 0);
            const f64 sld = quadratic ? sum_log_diag[k] : sum_log_diag[0];
            for (usize j = 0; j < d; ++j) diff[j] = x[j] - mu[j + k * d];
            fwd_solve_col(L, z.data(), diff.data(), d);
            f64 q = 0.0;
            for (usize j = 0; j < d; ++j) q += z[j] * z[j];
            const f64 log_fk = -0.5 * as<f64>(d) * kLog2Pi - sld - 0.5 * q;
            const f64 l_val = log_fk + log_prior[k];
            if (l_val > max_lp) {
                max_lp = l_val;
                k_best = k;
            }
        }
        if (uniq[k_best] != group_labels[i]) mis += 1.0;
    }
    out.train_error = mis / as<f64>(n_tr);
    return StatsStatus::kOk;
}

StatsStatus nearcorr(ArrayRef<f64> a_col_major, usize n, ArrayRef<f64> w_scale_col_major, f64 tol,
                     i64 max_iter, MutArrayRef<f64> out_col_major, Allocator* alloc) {
    if (n == 0 || mul_overflows(n, n) || a_col_major.size() != n * n ||
        out_col_major.size() != n * n || !(tol > 0.0) || max_iter < 0) {
        return StatsStatus::kInvalidArgument;
    }
    const usize nn = n * n;
    for (usize k = 0; k < nn; ++k) {
        if (!is_finite(a_col_major[k])) return StatsStatus::kInvalidArgument;
    }
    const bool has_weights = !w_scale_col_major.empty();
    if (has_weights) {
        if (w_scale_col_major.size() != nn) return StatsStatus::kInvalidArgument;
        for (usize k = 0; k < nn; ++k) {
            if (!(w_scale_col_major[k] > 0.0) || !is_finite(w_scale_col_major[k])) {
                return StatsStatus::kInvalidArgument;
            }
        }
    }
    if (n == 1) {
        out_col_major[0] = 1.0;
        return StatsStatus::kOk;
    }

    Array<f64> y_old(nn, 0.0, alloc);
    Array<f64> x_old(nn, 0.0, alloc);
    Array<f64> dS(nn, 0.0, alloc);
    Array<f64> Y(nn, 0.0, alloc);
    Array<f64> R(nn, 0.0, alloc);
    Array<f64> X(nn, 0.0, alloc);
    Array<f64> rw(has_weights ? nn : 0, 0.0, alloc);
    Array<f64> xw(has_weights ? nn : 0, 0.0, alloc);
    Array<f64> d1(nn, 0.0, alloc);
    Array<f64> d2(nn, 0.0, alloc);
    Array<f64> d3(nn, 0.0, alloc);

    for (usize k = 0; k < nn; ++k) {
        y_old[k] = a_col_major[k];
        x_old[k] = a_col_major[k];
        Y[k] = a_col_major[k];
    }

    for (i64 it = 0; it <= max_iter; ++it) {
        for (usize k = 0; k < nn; ++k) R[k] = y_old[k] - dS[k];
        if (has_weights) {
            for (usize k = 0; k < nn; ++k) rw[k] = R[k] * w_scale_col_major[k];
            proj_psd(rw.as_ref(), n, xw.as_mut_ref(), alloc);
            for (usize k = 0; k < nn; ++k) X[k] = xw[k] / w_scale_col_major[k];
        } else {
            proj_psd(R.as_ref(), n, X.as_mut_ref(), alloc);
        }
        for (usize k = 0; k < nn; ++k) {
            dS[k] = X[k] - R[k];
            Y[k] = X[k];
        }
        for (usize i = 0; i < n; ++i) Y[i + i * n] = 1.0;

        const f64 norm_y = max(frob_norm(Y.as_ref()), 1e-300);
        const f64 norm_x = max(frob_norm(X.as_ref()), 1e-300);
        for (usize k = 0; k < nn; ++k) {
            d1[k] = Y[k] - y_old[k];
            d2[k] = X[k] - x_old[k];
            d3[k] = Y[k] - X[k];
        }
        const f64 cond1 = frob_norm(d1.as_ref()) / norm_y;
        const f64 cond2 = frob_norm(d2.as_ref()) / norm_x;
        const f64 cond3 = frob_norm(d3.as_ref()) / norm_y;
        for (usize k = 0; k < nn; ++k) x_old[k] = X[k];
        if (max(cond1, max(cond2, cond3)) <= tol) break;
        for (usize k = 0; k < nn; ++k) y_old[k] = Y[k];
    }

    Array<f64> dd(n, 0.0, alloc);
    for (usize i = 0; i < n; ++i) dd[i] = max(x_old[i + i * n], 0.0);
    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            if (i == j) {
                out_col_major[i + j * n] = 1.0;
            } else {
                const f64 denom = sqrt(dd[i] * dd[j]);
                out_col_major[i + j * n] =
                    (denom > 0.0) ? clamp(0.5 * (x_old[i + j * n] + x_old[j + i * n]) / denom,
                                          -1.0, 1.0)
                                  : 0.0;
            }
        }
    }
    return StatsStatus::kOk;
}

} // namespace keel::stats
