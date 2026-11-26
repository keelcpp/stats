#include "keel/stats.h"

#include "keel/algo.h"
#include "keel/linalg.h"
#include "keel/math.h"

namespace keel::stats {

namespace {

inline bool mul_overflows(usize a, usize b) noexcept {
    return a != 0u && b > Limits<usize>::kMax / a;
}

f64 ad_pvalue_estimated_params(f64 a2star) noexcept {
    f64 p = 0.0;
    if (a2star < 0.200) {
        p = 1.0 - exp(-13.436 + 101.14 * a2star - 223.73 * a2star * a2star);
    } else if (a2star < 0.340) {
        p = 1.0 - exp(-8.318 + 42.796 * a2star - 59.938 * a2star * a2star);
    } else if (a2star < 0.600) {
        p = exp(0.9177 - 4.279 * a2star - 1.38 * a2star * a2star);
    } else if (a2star < 13.0) {
        p = exp(1.2937 - 5.709 * a2star + 0.0186 * a2star * a2star);
    } else {
        p = 0.0;
    }
    return clamp(p, 0.0, 1.0);
}

// Invert monotonically decreasing ad_pvalue_estimated_params(a2star) == alpha via bisection.
f64 ad_critical_value(f64 alpha) noexcept {
    f64 lo = 0.0;
    f64 hi = 13.0;
    for (i32 it = 0; it < 80; ++it) {
        const f64 mid = 0.5 * (lo + hi);
        if (ad_pvalue_estimated_params(mid) > alpha) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return 0.5 * (lo + hi);
}

bool invert_small_row(Array<f64>& A, usize d, Allocator* alloc) {
    if (d == 0) return true;
    if (mul_overflows(d, 2 * d)) return false;
    Array<f64> aug(d * 2 * d, 0.0, alloc);
    for (usize r = 0; r < d; ++r) {
        for (usize c = 0; c < d; ++c) aug[r * 2 * d + c] = A[r * d + c];
        aug[r * 2 * d + d + r] = 1.0;
    }
    for (usize kk = 0; kk < d; ++kk) {
        usize piv = kk;
        f64 pmax = abs(aug[kk * 2 * d + kk]);
        for (usize r = kk + 1; r < d; ++r) {
            const f64 v = abs(aug[r * 2 * d + kk]);
            if (v > pmax) {
                pmax = v;
                piv = r;
            }
        }
        if (!(pmax >= 1e-14)) return false;
        if (piv != kk) {
            for (usize c = 0; c < 2 * d; ++c) {
                keel::swap(aug[kk * 2 * d + c], aug[piv * 2 * d + c]);
            }
        }
        const f64 inv = 1.0 / aug[kk * 2 * d + kk];
        for (usize c = 0; c < 2 * d; ++c) aug[kk * 2 * d + c] *= inv;
        for (usize r = 0; r < d; ++r) {
            if (r == kk) continue;
            const f64 f = aug[r * 2 * d + kk];
            if (f == 0.0) continue;
            for (usize c = 0; c < 2 * d; ++c) aug[r * 2 * d + c] -= f * aug[kk * 2 * d + c];
        }
    }
    for (usize r = 0; r < d; ++r) {
        for (usize c = 0; c < d; ++c) A[r * d + c] = aug[r * 2 * d + d + c];
    }
    return true;
}

f64 imhof_integrand(f64 u, ArrayRef<f64> h) noexcept {
    if (u <= 0.0) {
        f64 s = 0.0;
        for (usize i = 0; i < h.size(); ++i) s += h[i];
        return 0.5 * s;
    }
    f64 theta = 0.0;
    f64 logrho = 0.0;
    for (usize i = 0; i < h.size(); ++i) {
        const f64 hu = h[i] * u;
        theta += atan(hu);
        logrho += 0.25 * log1p(hu * hu);
    }
    theta *= 0.5;
    if (logrho > 709.0) return 0.0;
    return sin(theta) / (u * exp(logrho));
}

f64 imhof_p_left(ArrayRef<f64> h) noexcept {
    const usize m = h.size();
    if (m == 0) return 0.5;
    if (m == 1) return (h[0] < 0.0) ? 1.0 : ((h[0] > 0.0) ? 0.0 : 0.5);

    constexpr i32 kPanels = 6000;
    auto ft = [&](f64 t) noexcept -> f64 {
        if (t >= 1.0) return 0.0;
        const f64 u = t / (1.0 - t);
        const f64 w = 1.0 / ((1.0 - t) * (1.0 - t));
        return imhof_integrand(u, h) * w;
    };
    f64 s = ft(0.0) + ft(1.0);
    const f64 step = 1.0 / as<f64>(kPanels);
    for (i32 i = 1; i < kPanels; ++i) {
        const f64 t = as<f64>(i) * step;
        s += ((i & 1) ? 4.0 : 2.0) * ft(t);
    }
    const f64 integral = s * step / 3.0;
    return clamp(0.5 - integral / kPi, 0.0, 1.0);
}

} // namespace

StatsStatus adtest(ArrayRef<f64> x, f64 alpha, AdTestResult& out, Allocator* alloc) {
    if (!(alpha > 0.0) || !(alpha < 1.0)) return StatsStatus::kInvalidArgument;

    Array<f64> v(alloc);
    v.reserve(x.size());
    for (usize i = 0; i < x.size(); ++i) {
        if (is_nan(x[i])) continue;
        if (!is_finite(x[i])) return StatsStatus::kInvalidArgument;
        v.push_back(x[i]);
    }
    const usize n = v.size();
    if (n < 4) return StatsStatus::kInvalidArgument;

    f64 m = 0.0;
    for (usize i = 0; i < n; ++i) m += v[i];
    m /= as<f64>(n);

    f64 s2 = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 d = v[i] - m;
        s2 += d * d;
    }
    s2 /= as<f64>(n - 1);
    const f64 sd = sqrt(s2);
    if (!(sd > 0.0)) return StatsStatus::kSingularMatrix;

    for (usize i = 0; i < n; ++i) v[i] = (v[i] - m) / sd;
    keel::sort(v);

    f64 sum = 0.0;
    for (usize i = 0; i < n; ++i) {
        // Use Phi(-z) == 1 - Phi(z) via erfc to avoid catastrophic tail cancellation.
        const f64 phi_lo = norm_cdf(v[i], 0.0, 1.0);
        const f64 surv_hi = norm_cdf(-v[n - 1 - i], 0.0, 1.0);
        const f64 ln_lo = log(max(phi_lo, 1e-300));
        const f64 ln_hi = log(max(surv_hi, 1e-300));
        sum += (2.0 * as<f64>(i) + 1.0) * (ln_lo + ln_hi);
    }
    const f64 nf = as<f64>(n);
    const f64 a2 = -nf - sum / nf;
    const f64 a2star = a2 * (1.0 + 0.75 / nf + 2.25 / (nf * nf));
    const f64 p = ad_pvalue_estimated_params(a2star);

    out.h = (p < alpha) ? 1 : 0;
    out.p = p;
    out.adstat = a2star;
    out.cv = ad_critical_value(alpha);
    return StatsStatus::kOk;
}

StatsStatus dwtest(ArrayRef<f64> r, ArrayRef<f64> x_col_major, usize n, usize k, bool exact,
                   DwTestResult& out, Allocator* alloc) {
    if (n < 3 || mul_overflows(n, k) || mul_overflows(n, n) || r.size() != n ||
        x_col_major.size() != n * k || k >= n) {
        return StatsStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n; ++i) {
        if (!is_finite(r[i])) return StatsStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n * k; ++i) {
        if (!is_finite(x_col_major[i])) return StatsStatus::kInvalidArgument;
    }

    f64 num = 0.0;
    f64 den = 0.0;
    for (usize i = 0; i < n; ++i) den += r[i] * r[i];
    for (usize i = 1; i < n; ++i) {
        const f64 d = r[i] - r[i - 1];
        num += d * d;
    }
    if (!(den > 0.0)) return StatsStatus::kInvalidArgument;

    const f64 dw = num / den;
    f64 p_left = 0.5;
    if (exact) {
        Array<f64> xtx(k * k, 0.0, alloc);
        for (usize a = 0; a < k; ++a) {
            for (usize b = 0; b < k; ++b) {
                f64 sab = 0.0;
                for (usize i = 0; i < n; ++i) {
                    sab += x_col_major[a * n + i] * x_col_major[b * n + i];
                }
                xtx[a * k + b] = sab;
            }
        }
        if (!invert_small_row(xtx, k, alloc)) return StatsStatus::kSingularMatrix;

        Array<f64> P(n * k, 0.0, alloc);
        for (usize i = 0; i < n; ++i) {
            for (usize a = 0; a < k; ++a) {
                f64 s = 0.0;
                for (usize b = 0; b < k; ++b) s += x_col_major[b * n + i] * xtx[b * k + a];
                P[i * k + a] = s;
            }
        }
        Array<f64> M(n * n, 0.0, alloc);
        for (usize i = 0; i < n; ++i) {
            for (usize j = 0; j < n; ++j) {
                f64 s = 0.0;
                for (usize a = 0; a < k; ++a) s += P[i * k + a] * x_col_major[a * n + j];
                M[i * n + j] = (i == j ? 1.0 : 0.0) - s;
            }
        }
        Array<f64> AM(n * n, 0.0, alloc);
        for (usize i = 0; i < n; ++i) {
            for (usize j = 0; j < n; ++j) {
                f64 v = (i == 0 || i + 1 == n ? 1.0 : 2.0) * M[i * n + j];
                if (i > 0) v -= M[(i - 1) * n + j];
                if (i + 1 < n) v -= M[(i + 1) * n + j];
                AM[i * n + j] = v;
            }
        }
        Array<f64> S(n * n, 0.0, alloc);
        for (usize i = 0; i < n; ++i) {
            for (usize j = 0; j < n; ++j) {
                f64 s = 0.0;
                for (usize l = 0; l < n; ++l) s += M[i * n + l] * AM[l * n + j];
                S[j * n + i] = s; // column-major
            }
        }
        for (usize j = 0; j < n; ++j) {
            for (usize i = j + 1; i < n; ++i) {
                const f64 sym = 0.5 * (S[i + j * n] + S[j + i * n]);
                S[i + j * n] = sym;
                S[j + i * n] = sym;
            }
        }
        Array<f64> V(n * n, 0.0, alloc);
        Array<f64> eig(n, 0.0, alloc);
        linalg::eig_sym(S.data(), n, V.data(), eig.data(), alloc);
        keel::sort(eig);

        Array<f64> h(alloc);
        h.reserve(n - k);
        for (usize i = k; i < n; ++i) h.push_back(eig[i] - dw);
        p_left = imhof_p_left(h.as_ref());
    } else {
        // Durbin-Watson normal approximation using exact mean and variance of r'Ar / r'r.
        Array<f64> xtx(k * k, 0.0, alloc);
        for (usize a = 0; a < k; ++a) {
            for (usize b = 0; b < k; ++b) {
                f64 sab = 0.0;
                for (usize i = 0; i < n; ++i) {
                    sab += x_col_major[a * n + i] * x_col_major[b * n + i];
                }
                xtx[a * k + b] = sab;
            }
        }
        if (!invert_small_row(xtx, k, alloc)) return StatsStatus::kSingularMatrix;

        // Compute A*X and X'*A*X and X'*A^2*X in O(n*k^2) without forming n x n matrices.
        Array<f64> ax(n * k, 0.0, alloc);
        for (usize a = 0; a < k; ++a) {
            for (usize i = 0; i < n; ++i) {
                f64 v = (i == 0 || i + 1 == n ? 1.0 : 2.0) * x_col_major[a * n + i];
                if (i > 0) v -= x_col_major[a * n + (i - 1)];
                if (i + 1 < n) v -= x_col_major[a * n + (i + 1)];
                ax[a * n + i] = v;
            }
        }
        // B = (X'X)^{-1} (X' A X)
        Array<f64> xax(k * k, 0.0, alloc);
        Array<f64> xa2x(k * k, 0.0, alloc);
        for (usize a = 0; a < k; ++a) {
            for (usize b = 0; b < k; ++b) {
                f64 s1 = 0.0;
                f64 s2 = 0.0;
                for (usize i = 0; i < n; ++i) {
                    s1 += x_col_major[a * n + i] * ax[b * n + i];
                    s2 += ax[a * n + i] * ax[b * n + i];
                }
                xax[a * k + b] = s1;
                xa2x[a * k + b] = s2;
            }
        }
        Array<f64> B(k * k, 0.0, alloc);
        f64 tr_b = 0.0;
        f64 tr_c = 0.0;
        for (usize a = 0; a < k; ++a) {
            for (usize b = 0; b < k; ++b) {
                f64 sb = 0.0;
                for (usize c = 0; c < k; ++c) sb += xtx[a * k + c] * xax[c * k + b];
                B[a * k + b] = sb;
                if (a == b) {
                    tr_b += sb;
                    f64 sc = 0.0;
                    for (usize c = 0; c < k; ++c) sc += xtx[a * k + c] * xa2x[c * k + b];
                    tr_c += sc;
                }
            }
        }
        f64 tr_b2 = 0.0;
        for (usize a = 0; a < k; ++a) {
            for (usize b = 0; b < k; ++b) tr_b2 += B[a * k + b] * B[b * k + a];
        }
        // tr(A) = 2*(n - 1), tr(A^2) = 2 + 6*(n - 2) + 2 = 6*n - 8 (for n >= 3).
        const f64 df = as<f64>(n - k);
        const f64 tr_ma = 2.0 * as<f64>(n - 1) - tr_b;
        const f64 tr_ma2 = (6.0 * as<f64>(n) - 8.0) - 2.0 * tr_c + tr_b2;
        const f64 mu = tr_ma / df;
        const f64 var = 2.0 * (tr_ma2 - mu * tr_ma) / (df * (df + 2.0));
        if (var > 0.0) {
            p_left = norm_cdf((dw - mu) / sqrt(var), 0.0, 1.0);
        } else {
            p_left = (dw < mu) ? 0.0 : ((dw > mu) ? 1.0 : 0.5);
        }
    }

    out.dw = dw;
    out.p_left = p_left;
    out.p_both = min(1.0, 2.0 * min(p_left, 1.0 - p_left));
    return StatsStatus::kOk;
}

} // namespace keel::stats
