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

inline bool safe_npairs(usize n, usize& out_pairs) noexcept {
    if (n < 2u) {
        out_pairs = 0u;
        return true;
    }
    const usize half = (n % 2u == 0u) ? (n / 2u) : ((n - 1u) / 2u);
    const usize other = (n % 2u == 0u) ? (n - 1u) : n;
    if (mul_overflows(half, other)) return false;
    out_pairs = half * other;
    return true;
}

inline void read_row(ArrayRef<f64> x_col, usize rows, usize cols, usize r,
                     MutArrayRef<f64> out) noexcept {
    for (usize c = 0; c < cols; ++c) out[c] = x_col[c * rows + r];
}

bool invert_small(MutArrayRef<f64> a_row_major, usize d, Allocator* alloc) {
    if (d == 0u) return true;
    if (mul_overflows(d, 2u * d)) return false;
    Array<f64> aug(d * 2u * d, 0.0, alloc);
    for (usize r = 0; r < d; ++r) {
        for (usize c = 0; c < d; ++c) {
            const f64 v = a_row_major[r * d + c];
            if (!is_finite(v)) return false;
            aug[r * 2u * d + c] = v;
        }
        aug[r * 2u * d + d + r] = 1.0;
    }
    for (usize k = 0; k < d; ++k) {
        usize piv = k;
        f64 pmax = keel::abs(aug[k * 2u * d + k]);
        for (usize r = k + 1u; r < d; ++r) {
            const f64 v = keel::abs(aug[r * 2u * d + k]);
            if (v > pmax) {
                pmax = v;
                piv = r;
            }
        }
        if (!(pmax >= 1e-14)) return false;
        if (piv != k) {
            for (usize c = 0; c < 2u * d; ++c) {
                keel::swap(aug[k * 2u * d + c], aug[piv * 2u * d + c]);
            }
        }
        const f64 inv_piv = 1.0 / aug[k * 2u * d + k];
        for (usize c = 0; c < 2u * d; ++c) aug[k * 2u * d + c] *= inv_piv;
        for (usize r = 0; r < d; ++r) {
            if (r == k) continue;
            const f64 factor = aug[r * 2u * d + k];
            if (factor == 0.0) continue;
            for (usize c = 0; c < 2u * d; ++c) {
                aug[r * 2u * d + c] -= factor * aug[k * 2u * d + c];
            }
        }
    }
    for (usize r = 0; r < d; ++r) {
        for (usize c = 0; c < d; ++c) {
            a_row_major[r * d + c] = aug[r * 2u * d + d + c];
        }
    }
    return true;
}

void data_cov(ArrayRef<f64> x_col, usize rows, usize cols, MutArrayRef<f64> cov_row_major,
              Allocator* alloc) {
    Array<f64> mu(cols, 0.0, alloc);
    for (usize c = 0; c < cols; ++c) {
        f64 s = 0.0;
        for (usize r = 0; r < rows; ++r) s += x_col[c * rows + r];
        mu[c] = s / as<f64>(rows);
    }
    for (usize i = 0; i < cols * cols; ++i) cov_row_major[i] = 0.0;
    Array<f64> dev(cols, 0.0, alloc);
    for (usize r = 0; r < rows; ++r) {
        for (usize c = 0; c < cols; ++c) dev[c] = x_col[c * rows + r] - mu[c];
        for (usize a = 0; a < cols; ++a) {
            for (usize b = 0; b < cols; ++b) {
                cov_row_major[a * cols + b] += dev[a] * dev[b];
            }
        }
    }
    const f64 inv_dof = (rows > 1u) ? (1.0 / as<f64>(rows - 1u)) : 1.0;
    for (usize i = 0; i < cols * cols; ++i) cov_row_major[i] *= inv_dof;
}

void col_std(ArrayRef<f64> x_col, usize rows, usize cols, MutArrayRef<f64> sd_out) noexcept {
    for (usize c = 0; c < cols; ++c) {
        f64 mu = 0.0;
        for (usize r = 0; r < rows; ++r) mu += x_col[c * rows + r];
        mu /= as<f64>(rows);
        f64 ss = 0.0;
        for (usize r = 0; r < rows; ++r) {
            const f64 d = x_col[c * rows + r] - mu;
            ss += d * d;
        }
        sd_out[c] = (rows > 1u) ? keel::sqrt(ss / as<f64>(rows - 1u)) : 0.0;
    }
}

void tiedrank_row(ArrayRef<f64> v, MutArrayRef<f64> r_out, Allocator* alloc) {
    const usize d = v.size();
    Array<usize> idx(alloc);
    idx.reserve(d);
    for (usize k = 0; k < d; ++k) {
        if (is_nan(v[k])) {
            r_out[k] = Limits<f64>::quiet_nan();
        } else {
            idx.push_back(k);
        }
    }
    const usize m = idx.size();
    keel::sort(idx.as_mut_ref(), [&](usize a, usize b) noexcept { return v[a] < v[b]; });
    usize i = 0u;
    while (i < m) {
        usize j = i;
        while (j + 1u < m && v[idx[j + 1u]] == v[idx[i]]) ++j;
        const f64 avg = 0.5 * (as<f64>(i + 1u) + as<f64>(j + 1u));
        for (usize t = i; t <= j; ++t) r_out[idx[t]] = avg;
        i = j + 1u;
    }
}

void tiedrank_rows(ArrayRef<f64> x_col, usize rows, usize cols, MutArrayRef<f64> r_row_major,
                   Allocator* alloc) {
    Array<f64> row(cols, 0.0, alloc);
    for (usize r = 0; r < rows; ++r) {
        read_row(x_col, rows, cols, r, row.as_mut_ref());
        tiedrank_row(row.as_ref(), MutArrayRef<f64>(r_row_major.data() + r * cols, cols), alloc);
    }
}

inline f64 mahal_distance(ArrayRef<f64> u, ArrayRef<f64> v, ArrayRef<f64> cinv,
                          MutArrayRef<f64> diff) noexcept {
    const usize d = u.size();
    for (usize k = 0; k < d; ++k) diff[k] = u[k] - v[k];
    f64 s = 0.0;
    for (usize a = 0; a < d; ++a) {
        f64 row = 0.0;
        for (usize b = 0; b < d; ++b) row += cinv[a * d + b] * diff[b];
        s += diff[a] * row;
    }
    if (is_nan(s)) return Limits<f64>::quiet_nan();
    return keel::sqrt(s > 0.0 ? s : 0.0);
}

inline f64 seuclidean_distance(ArrayRef<f64> x, ArrayRef<f64> y,
                               ArrayRef<f64> inv_scale) noexcept {
    f64 s = 0.0;
    for (usize k = 0; k < x.size(); ++k) {
        const f64 d = (x[k] - y[k]) * inv_scale[k];
        s += d * d;
    }
    return keel::sqrt(s);
}

inline f64 correlation_distance(ArrayRef<f64> x, ArrayRef<f64> y) noexcept {
    const usize d = x.size();
    if (d == 0u) return Limits<f64>::quiet_nan();
    f64 mx = 0.0;
    f64 my = 0.0;
    for (usize k = 0; k < d; ++k) {
        mx += x[k];
        my += y[k];
    }
    mx /= as<f64>(d);
    my /= as<f64>(d);
    f64 xy = 0.0, xx = 0.0, yy = 0.0;
    for (usize k = 0; k < d; ++k) {
        const f64 dx = x[k] - mx;
        const f64 dy = y[k] - my;
        xy += dx * dy;
        xx += dx * dx;
        yy += dy * dy;
    }
    const f64 denom = keel::sqrt(xx) * keel::sqrt(yy);
    if (!(denom > 0.0)) return Limits<f64>::quiet_nan();
    const f64 dist = 1.0 - xy / denom;
    return is_nan(dist) ? dist : clamp(dist, 0.0, 2.0);
}

inline f64 row_distance(ArrayRef<f64> x, ArrayRef<f64> y, DistanceMetric m, f64 p) noexcept {
    const usize d = x.size();
    switch (m) {
        case DistanceMetric::kEuclidean: {
            f64 scale = 0.0;
            for (usize k = 0; k < d; ++k) {
                const f64 ad = keel::abs(x[k] - y[k]);
                if (is_nan(ad)) return Limits<f64>::quiet_nan();
                if (ad > scale) scale = ad;
            }
            if (scale == 0.0) return 0.0;
            if (is_inf(scale)) return scale;
            const f64 inv_s = 1.0 / scale;
            f64 s = 0.0;
            for (usize k = 0; k < d; ++k) {
                const f64 diff = (x[k] - y[k]) * inv_s;
                s += diff * diff;
            }
            return scale * keel::sqrt(s);
        }
        case DistanceMetric::kSqEuclidean: {
            f64 s = 0.0;
            for (usize k = 0; k < d; ++k) {
                const f64 diff = x[k] - y[k];
                s += diff * diff;
            }
            return s;
        }
        case DistanceMetric::kCityblock: {
            f64 s = 0.0;
            for (usize k = 0; k < d; ++k) s += keel::abs(x[k] - y[k]);
            return s;
        }
        case DistanceMetric::kChebychev: {
            f64 mx = 0.0;
            for (usize k = 0; k < d; ++k) {
                const f64 diff = keel::abs(x[k] - y[k]);
                if (is_nan(diff)) return Limits<f64>::quiet_nan();
                if (diff > mx) mx = diff;
            }
            return mx;
        }
        case DistanceMetric::kMinkowski: {
            f64 scale = 0.0;
            for (usize k = 0; k < d; ++k) {
                const f64 ad = keel::abs(x[k] - y[k]);
                if (is_nan(ad)) return Limits<f64>::quiet_nan();
                if (ad > scale) scale = ad;
            }
            if (scale == 0.0 || is_inf(scale)) return scale;
            if (is_inf(p)) return scale;
            const f64 inv_s = 1.0 / scale;
            f64 s = 0.0;
            for (usize k = 0; k < d; ++k) s += keel::pow(keel::abs(x[k] - y[k]) * inv_s, p);
            return scale * keel::pow(s, 1.0 / p);
        }
        case DistanceMetric::kCosine: {
            f64 xy = 0.0, xx = 0.0, yy = 0.0;
            for (usize k = 0; k < d; ++k) {
                xy += x[k] * y[k];
                xx += x[k] * x[k];
                yy += y[k] * y[k];
            }
            const f64 denom = keel::sqrt(xx) * keel::sqrt(yy);
            if (!(denom > 0.0)) return Limits<f64>::quiet_nan();
            const f64 dist = 1.0 - xy / denom;
            return is_nan(dist) ? dist : clamp(dist, 0.0, 2.0);
        }
        case DistanceMetric::kCorrelation:
            return correlation_distance(x, y);
        case DistanceMetric::kHamming: {
            if (d == 0u) return 0.0;
            usize diff = 0u;
            for (usize k = 0; k < d; ++k) {
                if (x[k] != y[k]) ++diff;
            }
            return as<f64>(diff) / as<f64>(d);
        }
        case DistanceMetric::kJaccard: {
            usize diff = 0u;
            usize considered = 0u;
            for (usize k = 0; k < d; ++k) {
                if (is_nan(x[k]) || is_nan(y[k])) return Limits<f64>::quiet_nan();
                const bool nz = (x[k] != 0.0) || (y[k] != 0.0);
                if (nz) {
                    ++considered;
                    if (x[k] != y[k]) ++diff;
                }
            }
            return (considered > 0u) ? (as<f64>(diff) / as<f64>(considered))
                                     : Limits<f64>::quiet_nan();
        }
        case DistanceMetric::kMahalanobis:
        case DistanceMetric::kSeuclidean:
        case DistanceMetric::kSpearman:
            break;
    }
    return 0.0;
}

} // namespace

StatsStatus pdist(ArrayRef<f64> x_col_major, usize rows, usize cols, DistanceMetric metric,
                  f64 minkowski_p, ArrayRef<f64> extra_param, MutArrayRef<f64> out,
                  Allocator* alloc) {
    if (mul_overflows(rows, cols) || x_col_major.size() < rows * cols) {
        return StatsStatus::kInvalidArgument;
    }
    if (metric == DistanceMetric::kMinkowski && !(minkowski_p > 0.0)) {
        return StatsStatus::kInvalidArgument;
    }
    if (rows < 2u) return StatsStatus::kOk;
    usize npairs = 0u;
    if (!safe_npairs(rows, npairs) || out.size() < npairs) return StatsStatus::kInvalidArgument;

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> xi(cols, 0.0, scratch.allocator());
    Array<f64> xj(cols, 0.0, scratch.allocator());

    if (metric == DistanceMetric::kMahalanobis) {
        if (mul_overflows(cols, cols)) return StatsStatus::kInvalidArgument;
        Array<f64> cinv(cols * cols, 0.0, scratch.allocator());
        if (!extra_param.empty()) {
            if (extra_param.size() < cols * cols) return StatsStatus::kInvalidArgument;
            for (usize r = 0; r < cols; ++r) {
                for (usize c = 0; c < cols; ++c) {
                    cinv[r * cols + c] = extra_param[c * cols + r];
                }
            }
        } else {
            data_cov(x_col_major, rows, cols, cinv.as_mut_ref(), scratch.allocator());
        }
        if (!invert_small(cinv.as_mut_ref(), cols, scratch.allocator())) {
            return StatsStatus::kSingularMatrix;
        }
        Array<f64> diff(cols, 0.0, scratch.allocator());
        usize idx = 0u;
        for (usize i = 0; i < rows; ++i) {
            read_row(x_col_major, rows, cols, i, xi.as_mut_ref());
            for (usize j = i + 1u; j < rows; ++j) {
                read_row(x_col_major, rows, cols, j, xj.as_mut_ref());
                out[idx++] = mahal_distance(xi.as_ref(), xj.as_ref(), cinv.as_ref(), diff.as_mut_ref());
            }
        }
        return StatsStatus::kOk;
    }

    if (metric == DistanceMetric::kSeuclidean) {
        Array<f64> inv_scale(cols, 0.0, scratch.allocator());
        if (!extra_param.empty()) {
            if (extra_param.size() != cols) return StatsStatus::kInvalidArgument;
            for (usize k = 0; k < cols; ++k) {
                if (!(extra_param[k] > 0.0)) return StatsStatus::kInvalidArgument;
                inv_scale[k] = 1.0 / extra_param[k];
            }
        } else {
            Array<f64> sd(cols, 0.0, scratch.allocator());
            col_std(x_col_major, rows, cols, sd.as_mut_ref());
            for (usize k = 0; k < cols; ++k) {
                inv_scale[k] = (sd[k] > 0.0) ? (1.0 / sd[k]) : Limits<f64>::quiet_nan();
            }
        }
        usize idx = 0u;
        for (usize i = 0; i < rows; ++i) {
            read_row(x_col_major, rows, cols, i, xi.as_mut_ref());
            for (usize j = i + 1u; j < rows; ++j) {
                read_row(x_col_major, rows, cols, j, xj.as_mut_ref());
                out[idx++] = seuclidean_distance(xi.as_ref(), xj.as_ref(), inv_scale.as_ref());
            }
        }
        return StatsStatus::kOk;
    }

    if (metric == DistanceMetric::kSpearman) {
        Array<f64> r_mat(rows * cols, 0.0, scratch.allocator());
        tiedrank_rows(x_col_major, rows, cols, r_mat.as_mut_ref(), scratch.allocator());
        usize idx = 0u;
        for (usize i = 0; i < rows; ++i) {
            const ArrayRef<f64> ri(r_mat.data() + i * cols, cols);
            for (usize j = i + 1u; j < rows; ++j) {
                const ArrayRef<f64> rj(r_mat.data() + j * cols, cols);
                out[idx++] = correlation_distance(ri, rj);
            }
        }
        return StatsStatus::kOk;
    }

    if (metric == DistanceMetric::kCosine) {
        Array<f64> norm_rows(rows * cols, 0.0, scratch.allocator());
        Array<u8> valid(rows, u8{0}, scratch.allocator());
        for (usize i = 0; i < rows; ++i) {
            read_row(x_col_major, rows, cols, i, xi.as_mut_ref());
            f64 scale = 0.0;
            bool has_nan = false;
            for (usize k = 0; k < cols; ++k) {
                if (is_nan(xi[k])) {
                    has_nan = true;
                    break;
                }
                const f64 a = keel::abs(xi[k]);
                if (a > scale) scale = a;
            }
            if (has_nan || !(scale > 0.0) || !is_finite(scale)) continue;
            f64 ss = 0.0;
            const f64 inv_s = 1.0 / scale;
            for (usize k = 0; k < cols; ++k) {
                const f64 v = xi[k] * inv_s;
                ss += v * v;
            }
            const f64 nrm = keel::sqrt(ss);
            if (nrm > 0.0) {
                valid[i] = 1u;
                f64* dst = norm_rows.data() + i * cols;
                for (usize k = 0; k < cols; ++k) dst[k] = (xi[k] * inv_s) / nrm;
            }
        }
        usize idx = 0u;
        for (usize i = 0; i < rows; ++i) {
            const f64* ui = norm_rows.data() + i * cols;
            for (usize j = i + 1u; j < rows; ++j) {
                if (!valid[i] || !valid[j]) {
                    out[idx++] = Limits<f64>::quiet_nan();
                    continue;
                }
                const f64* uj = norm_rows.data() + j * cols;
                f64 dot = 0.0;
                for (usize k = 0; k < cols; ++k) dot += ui[k] * uj[k];
                const f64 dist = 1.0 - dot;
                out[idx++] = is_nan(dist) ? dist : clamp(dist, 0.0, 2.0);
            }
        }
        return StatsStatus::kOk;
    }

    usize idx = 0u;
    for (usize i = 0; i < rows; ++i) {
        read_row(x_col_major, rows, cols, i, xi.as_mut_ref());
        for (usize j = i + 1u; j < rows; ++j) {
            read_row(x_col_major, rows, cols, j, xj.as_mut_ref());
            out[idx++] = row_distance(xi.as_ref(), xj.as_ref(), metric, minkowski_p);
        }
    }
    return StatsStatus::kOk;
}

StatsStatus pdist2(ArrayRef<f64> x_col_major, usize mx, ArrayRef<f64> y_col_major, usize my,
                   usize cols, DistanceMetric metric, f64 minkowski_p, ArrayRef<f64> extra_param,
                   MutArrayRef<f64> out_col_major, Allocator* alloc) {
    if (mul_overflows(mx, cols) || mul_overflows(my, cols) || mul_overflows(mx, my) ||
        x_col_major.size() < mx * cols || y_col_major.size() < my * cols ||
        out_col_major.size() < mx * my) {
        return StatsStatus::kInvalidArgument;
    }
    if (metric == DistanceMetric::kMinkowski && !(minkowski_p > 0.0)) {
        return StatsStatus::kInvalidArgument;
    }
    if (mx == 0u || my == 0u) return StatsStatus::kOk;

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> xi(cols, 0.0, scratch.allocator());
    Array<f64> yj(cols, 0.0, scratch.allocator());

    if (metric == DistanceMetric::kMahalanobis) {
        if (mul_overflows(cols, cols)) return StatsStatus::kInvalidArgument;
        Array<f64> cinv(cols * cols, 0.0, scratch.allocator());
        if (!extra_param.empty()) {
            if (extra_param.size() < cols * cols) return StatsStatus::kInvalidArgument;
            for (usize r = 0; r < cols; ++r) {
                for (usize c = 0; c < cols; ++c) {
                    cinv[r * cols + c] = extra_param[c * cols + r];
                }
            }
        } else {
            data_cov(x_col_major, mx, cols, cinv.as_mut_ref(), scratch.allocator());
        }
        if (!invert_small(cinv.as_mut_ref(), cols, scratch.allocator())) {
            return StatsStatus::kSingularMatrix;
        }
        Array<f64> diff(cols, 0.0, scratch.allocator());
        for (usize j = 0; j < my; ++j) {
            read_row(y_col_major, my, cols, j, yj.as_mut_ref());
            for (usize i = 0; i < mx; ++i) {
                read_row(x_col_major, mx, cols, i, xi.as_mut_ref());
                out_col_major[j * mx + i] =
                    mahal_distance(xi.as_ref(), yj.as_ref(), cinv.as_ref(), diff.as_mut_ref());
            }
        }
        return StatsStatus::kOk;
    }

    if (metric == DistanceMetric::kSeuclidean) {
        Array<f64> inv_scale(cols, 0.0, scratch.allocator());
        if (!extra_param.empty()) {
            if (extra_param.size() != cols) return StatsStatus::kInvalidArgument;
            for (usize k = 0; k < cols; ++k) {
                if (!(extra_param[k] > 0.0)) return StatsStatus::kInvalidArgument;
                inv_scale[k] = 1.0 / extra_param[k];
            }
        } else {
            Array<f64> sd(cols, 0.0, scratch.allocator());
            col_std(x_col_major, mx, cols, sd.as_mut_ref());
            for (usize k = 0; k < cols; ++k) {
                inv_scale[k] = (sd[k] > 0.0) ? (1.0 / sd[k]) : Limits<f64>::quiet_nan();
            }
        }
        for (usize j = 0; j < my; ++j) {
            read_row(y_col_major, my, cols, j, yj.as_mut_ref());
            for (usize i = 0; i < mx; ++i) {
                read_row(x_col_major, mx, cols, i, xi.as_mut_ref());
                out_col_major[j * mx + i] =
                    seuclidean_distance(xi.as_ref(), yj.as_ref(), inv_scale.as_ref());
            }
        }
        return StatsStatus::kOk;
    }

    if (metric == DistanceMetric::kSpearman) {
        Array<f64> rx(mx * cols, 0.0, scratch.allocator());
        Array<f64> ry(my * cols, 0.0, scratch.allocator());
        tiedrank_rows(x_col_major, mx, cols, rx.as_mut_ref(), scratch.allocator());
        tiedrank_rows(y_col_major, my, cols, ry.as_mut_ref(), scratch.allocator());
        for (usize j = 0; j < my; ++j) {
            const ArrayRef<f64> rj(ry.data() + j * cols, cols);
            for (usize i = 0; i < mx; ++i) {
                const ArrayRef<f64> ri(rx.data() + i * cols, cols);
                out_col_major[j * mx + i] = correlation_distance(ri, rj);
            }
        }
        return StatsStatus::kOk;
    }

    if (metric == DistanceMetric::kCosine) {
        Array<f64> norm_x(mx * cols, 0.0, scratch.allocator());
        Array<u8> valid_x(mx, u8{0}, scratch.allocator());
        for (usize i = 0; i < mx; ++i) {
            read_row(x_col_major, mx, cols, i, xi.as_mut_ref());
            f64 scale = 0.0;
            bool has_nan = false;
            for (usize k = 0; k < cols; ++k) {
                if (is_nan(xi[k])) {
                    has_nan = true;
                    break;
                }
                const f64 a = keel::abs(xi[k]);
                if (a > scale) scale = a;
            }
            if (has_nan || !(scale > 0.0) || !is_finite(scale)) continue;
            f64 ss = 0.0;
            const f64 inv_s = 1.0 / scale;
            for (usize k = 0; k < cols; ++k) {
                const f64 v = xi[k] * inv_s;
                ss += v * v;
            }
            const f64 nrm = keel::sqrt(ss);
            if (nrm > 0.0) {
                valid_x[i] = 1u;
                f64* dst = norm_x.data() + i * cols;
                for (usize k = 0; k < cols; ++k) dst[k] = (xi[k] * inv_s) / nrm;
            }
        }
        Array<f64> norm_y(my * cols, 0.0, scratch.allocator());
        Array<u8> valid_y(my, u8{0}, scratch.allocator());
        for (usize j = 0; j < my; ++j) {
            read_row(y_col_major, my, cols, j, yj.as_mut_ref());
            f64 scale = 0.0;
            bool has_nan = false;
            for (usize k = 0; k < cols; ++k) {
                if (is_nan(yj[k])) {
                    has_nan = true;
                    break;
                }
                const f64 a = keel::abs(yj[k]);
                if (a > scale) scale = a;
            }
            if (has_nan || !(scale > 0.0) || !is_finite(scale)) continue;
            f64 ss = 0.0;
            const f64 inv_s = 1.0 / scale;
            for (usize k = 0; k < cols; ++k) {
                const f64 v = yj[k] * inv_s;
                ss += v * v;
            }
            const f64 nrm = keel::sqrt(ss);
            if (nrm > 0.0) {
                valid_y[j] = 1u;
                f64* dst = norm_y.data() + j * cols;
                for (usize k = 0; k < cols; ++k) dst[k] = (yj[k] * inv_s) / nrm;
            }
        }
        for (usize j = 0; j < my; ++j) {
            const f64* uj = norm_y.data() + j * cols;
            for (usize i = 0; i < mx; ++i) {
                if (!valid_x[i] || !valid_y[j]) {
                    out_col_major[j * mx + i] = Limits<f64>::quiet_nan();
                    continue;
                }
                const f64* ui = norm_x.data() + i * cols;
                f64 dot = 0.0;
                for (usize k = 0; k < cols; ++k) dot += ui[k] * uj[k];
                const f64 dist = 1.0 - dot;
                out_col_major[j * mx + i] = is_nan(dist) ? dist : clamp(dist, 0.0, 2.0);
            }
        }
        return StatsStatus::kOk;
    }

    for (usize j = 0; j < my; ++j) {
        read_row(y_col_major, my, cols, j, yj.as_mut_ref());
        for (usize i = 0; i < mx; ++i) {
            read_row(x_col_major, mx, cols, i, xi.as_mut_ref());
            out_col_major[j * mx + i] = row_distance(xi.as_ref(), yj.as_ref(), metric, minkowski_p);
        }
    }
    return StatsStatus::kOk;
}

StatsStatus squareform_to_matrix(ArrayRef<f64> condensed, usize n,
                                 MutArrayRef<f64> out_col_major) noexcept {
    usize npairs = 0u;
    if (!safe_npairs(n, npairs) || mul_overflows(n, n) || condensed.size() != npairs ||
        out_col_major.size() < n * n) {
        return StatsStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n * n; ++i) out_col_major[i] = 0.0;
    usize k = 0u;
    for (usize i = 0; i < n; ++i) {
        for (usize j = i + 1u; j < n; ++j) {
            const f64 v = condensed[k++];
            out_col_major[j * n + i] = v;
            out_col_major[i * n + j] = v;
        }
    }
    return StatsStatus::kOk;
}

StatsStatus squareform_to_condensed(ArrayRef<f64> mat_col_major, usize n,
                                    MutArrayRef<f64> out_condensed) noexcept {
    usize npairs = 0u;
    if (!safe_npairs(n, npairs) || mul_overflows(n, n) || mat_col_major.size() < n * n ||
        out_condensed.size() < npairs) {
        return StatsStatus::kInvalidArgument;
    }
    usize k = 0u;
    for (usize i = 0; i < n; ++i) {
        for (usize j = i + 1u; j < n; ++j) {
            out_condensed[k++] = mat_col_major[j * n + i];
        }
    }
    return StatsStatus::kOk;
}

StatsStatus mahal_qr(ArrayRef<f64> y_col_major, usize my, ArrayRef<f64> x_col_major, usize mx,
                     usize cols, MutArrayRef<f64> out_d2, Allocator* alloc) {
    if (mx < 2u || mul_overflows(mx, cols) || mul_overflows(my, cols) || mul_overflows(cols, cols) ||
        x_col_major.size() < mx * cols || y_col_major.size() < my * cols || out_d2.size() < my) {
        return StatsStatus::kInvalidArgument;
    }
    if (my == 0u) return StatsStatus::kOk;
    if (cols == 0u) {
        for (usize r = 0; r < my; ++r) out_d2[r] = 0.0;
        return StatsStatus::kOk;
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> mu(cols, 0.0, scratch.allocator());
    for (usize c = 0; c < cols; ++c) {
        f64 s = 0.0;
        for (usize r = 0; r < mx; ++r) s += x_col_major[c * mx + r];
        mu[c] = s / as<f64>(mx);
    }

    Array<f64> xc(mx * cols, 0.0, scratch.allocator());
    for (usize c = 0; c < cols; ++c) {
        for (usize r = 0; r < mx; ++r) {
            xc[r + c * mx] = x_col_major[c * mx + r] - mu[c];
        }
    }

    Array<f64> r_mat(cols * cols, 0.0, scratch.allocator());
    Array<f64> v(mx, 0.0, scratch.allocator());
    for (usize k = 0; k < cols; ++k) {
        if (k >= mx) break;
        f64 tail_sq = 0.0;
        for (usize i = k + 1u; i < mx; ++i) tail_sq += xc[i + k * mx] * xc[i + k * mx];
        if (tail_sq == 0.0) {
            r_mat[k + k * cols] = xc[k + k * mx];
            for (usize j = k + 1u; j < cols; ++j) r_mat[k + j * cols] = xc[k + j * mx];
            continue;
        }
        const f64 xk = xc[k + k * mx];
        const f64 norm = keel::sqrt(tail_sq + xk * xk);
        const f64 alpha = (xk >= 0.0) ? -norm : norm;
        v[0] = xk - alpha;
        for (usize i = k + 1u; i < mx; ++i) v[i - k] = xc[i + k * mx];
        f64 v_sq = 0.0;
        for (usize i = 0; i < mx - k; ++i) v_sq += v[i] * v[i];
        const f64 tau = 2.0 / v_sq;
        for (usize j = k; j < cols; ++j) {
            f64 dot = 0.0;
            for (usize i = k; i < mx; ++i) dot += v[i - k] * xc[i + j * mx];
            const f64 s = tau * dot;
            for (usize i = k; i < mx; ++i) xc[i + j * mx] -= s * v[i - k];
        }
        r_mat[k + k * cols] = alpha;
        for (usize j = k + 1u; j < cols; ++j) r_mat[k + j * cols] = xc[k + j * mx];
    }

    constexpr f64 kDiagTol = 1e-12;
    Array<f64> dy(cols, 0.0, scratch.allocator());
    Array<f64> z(cols, 0.0, scratch.allocator());
    for (usize r = 0; r < my; ++r) {
        for (usize k = 0; k < cols; ++k) dy[k] = y_col_major[k * my + r] - mu[k];
        for (usize k = 0; k < cols; ++k) z[k] = 0.0;
        for (usize i = 0; i < cols; ++i) {
            const f64 rii = r_mat[i + i * cols];
            if (!(keel::abs(rii) >= kDiagTol)) continue;
            f64 s = dy[i];
            for (usize k = 0; k < i; ++k) s -= r_mat[k + i * cols] * z[k];
            z[i] = s / rii;
        }
        f64 m2 = 0.0;
        for (usize k = 0; k < cols; ++k) m2 += z[k] * z[k];
        out_d2[r] = m2 * as<f64>(mx - 1u);
    }
    return StatsStatus::kOk;
}

} // namespace keel::stats
