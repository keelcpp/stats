#include "keel/stats.h"

#include "keel/algo.h"
#include "keel/math.h"

namespace keel::stats {

namespace {

inline f64 dist_pair_row(const f64* a, const f64* b, usize d, DistanceMetric m, f64 p) noexcept {
    switch (m) {
        case DistanceMetric::kEuclidean: {
            f64 s = 0.0;
            for (usize k = 0; k < d; ++k) {
                const f64 diff = a[k] - b[k];
                s += diff * diff;
            }
            return sqrt(s);
        }
        case DistanceMetric::kSqEuclidean: {
            f64 s = 0.0;
            for (usize k = 0; k < d; ++k) {
                const f64 diff = a[k] - b[k];
                s += diff * diff;
            }
            return s;
        }
        case DistanceMetric::kCityblock: {
            f64 s = 0.0;
            for (usize k = 0; k < d; ++k) s += abs(a[k] - b[k]);
            return s;
        }
        case DistanceMetric::kChebychev: {
            f64 mv = 0.0;
            for (usize k = 0; k < d; ++k) {
                const f64 diff = abs(a[k] - b[k]);
                if (diff > mv) mv = diff;
            }
            return mv;
        }
        case DistanceMetric::kMinkowski: {
            f64 s = 0.0;
            for (usize k = 0; k < d; ++k) s += pow(abs(a[k] - b[k]), p);
            return pow(s, 1.0 / p);
        }
        case DistanceMetric::kCosine: {
            f64 xy = 0.0;
            f64 xx = 0.0;
            f64 yy = 0.0;
            for (usize k = 0; k < d; ++k) {
                xy += a[k] * b[k];
                xx += a[k] * a[k];
                yy += b[k] * b[k];
            }
            const f64 denom = sqrt(xx) * sqrt(yy);
            return (denom > 0.0) ? (1.0 - xy / denom) : 1.0;
        }
        case DistanceMetric::kHamming: {
            i32 diff = 0;
            for (usize k = 0; k < d; ++k) {
                if (a[k] != b[k]) ++diff;
            }
            return (d > 0) ? (as<f64>(diff) / as<f64>(d)) : 0.0;
        }
        case DistanceMetric::kJaccard: {
            i32 diff = 0;
            i32 considered = 0;
            for (usize k = 0; k < d; ++k) {
                const bool xi = a[k] != 0.0;
                const bool yi = b[k] != 0.0;
                if (xi || yi) {
                    ++considered;
                    if (xi != yi) ++diff;
                }
            }
            return (considered > 0) ? (as<f64>(diff) / as<f64>(considered)) : 0.0;
        }
        default: {
            f64 s = 0.0;
            for (usize k = 0; k < d; ++k) {
                const f64 diff = a[k] - b[k];
                s += diff * diff;
            }
            return sqrt(s);
        }
    }
}

void col_to_row_major(ArrayRef<f64> x_col, usize n, usize d, Array<f64>& out_row) {
    out_row.resize(n * d);
    for (usize r = 0; r < n; ++r) {
        for (usize c = 0; c < d; ++c) {
            out_row[r * d + c] = x_col[c * n + r];
        }
    }
}

} // namespace

StatsStatus kmedoids_pam_step(ArrayRef<f64> x_col_major, usize n, usize d,
                              ArrayRef<i32> init_medoids, i32 max_iter, DistanceMetric metric,
                              f64 minkowski_p, MutArrayRef<i32> out_idx,
                              MutArrayRef<i32> out_medoids, MutArrayRef<f64> out_sumd,
                              f64& out_total, i32& out_iters, Allocator* alloc) {
    const usize k_cnt = init_medoids.size();
    if (n == 0 || d == 0 || k_cnt == 0 || k_cnt > n || x_col_major.size() != n * d ||
        out_idx.size() != n || out_medoids.size() != k_cnt || out_sumd.size() != k_cnt) {
        return StatsStatus::kInvalidArgument;
    }
    if (max_iter <= 0) max_iter = 100;

    Array<f64> xv(alloc);
    col_to_row_major(x_col_major, n, d, xv);

    for (usize k = 0; k < k_cnt; ++k) {
        if (init_medoids[k] < 0 || as<usize>(init_medoids[k]) >= n) {
            return StatsStatus::kInvalidArgument;
        }
        out_medoids[k] = init_medoids[k];
    }

    Array<i32> members(alloc);
    members.reserve(n);
    i32 iters = 0;

    for (i32 iter = 0; iter < max_iter; ++iter) {
        ++iters;
        for (usize k = 0; k < k_cnt; ++k) out_sumd[k] = 0.0;
        for (usize i = 0; i < n; ++i) {
            i32 best = 0;
            f64 bd = Limits<f64>::infinity();
            for (usize k = 0; k < k_cnt; ++k) {
                const f64 dist = dist_pair_row(&xv[i * d], &xv[as<usize>(out_medoids[k]) * d], d,
                                               metric, minkowski_p);
                if (dist < bd) {
                    bd = dist;
                    best = as<i32>(k);
                }
            }
            out_idx[i] = best;
            out_sumd[as<usize>(best)] += bd;
        }

        bool changed = false;
        for (usize k = 0; k < k_cnt; ++k) {
            members.clear();
            for (usize i = 0; i < n; ++i) {
                if (out_idx[i] == as<i32>(k)) members.push_back(as<i32>(i));
            }
            if (members.empty()) continue;

            i32 new_med = out_medoids[k];
            f64 best_sum = Limits<f64>::infinity();
            for (usize ci = 0; ci < members.size(); ++ci) {
                const i32 cand = members[ci];
                f64 s = 0.0;
                for (usize pi = 0; pi < members.size(); ++pi) {
                    const i32 pt = members[pi];
                    s += dist_pair_row(&xv[as<usize>(cand) * d], &xv[as<usize>(pt) * d], d, metric,
                                       minkowski_p);
                }
                if (s < best_sum) {
                    best_sum = s;
                    new_med = cand;
                }
            }
            if (new_med != out_medoids[k]) {
                out_medoids[k] = new_med;
                changed = true;
            }
        }
        if (!changed) break;
    }

    f64 total = 0.0;
    for (usize k = 0; k < k_cnt; ++k) out_sumd[k] = 0.0;
    for (usize i = 0; i < n; ++i) {
        const usize k = as<usize>(out_idx[i]);
        const f64 dist = dist_pair_row(&xv[i * d], &xv[as<usize>(out_medoids[k]) * d], d, metric,
                                       minkowski_p);
        out_sumd[k] += dist;
        total += dist;
    }
    out_total = total;
    out_iters = iters;
    return StatsStatus::kOk;
}

StatsStatus dbscan(ArrayRef<f64> x_col_major, usize n, usize d, f64 eps, i32 minpts,
                   DistanceMetric metric, f64 minkowski_p, bool precomputed,
                   MutArrayRef<i32> out_labels, MutArrayRef<u8> out_core, Allocator* alloc) {
    if (!(eps > 0.0) || out_labels.size() != n || out_core.size() != n) {
        return StatsStatus::kInvalidArgument;
    }
    if (minpts <= 0) minpts = 1;
    if (precomputed) {
        if (x_col_major.size() != n * n) return StatsStatus::kInvalidArgument;
    } else {
        if (x_col_major.size() != n * d) return StatsStatus::kInvalidArgument;
    }
    if (n == 0) return StatsStatus::kOk;

    Array<f64> xv(alloc);
    if (!precomputed) col_to_row_major(x_col_major, n, d, xv);

    auto neighbours = [&](usize i, Array<i32>& out) {
        out.clear();
        for (usize j = 0; j < n; ++j) {
            if (j == i) {
                out.push_back(as<i32>(j));
                continue;
            }
            const f64 dist = precomputed
                                 ? x_col_major[j * n + i]
                                 : dist_pair_row(&xv[i * d], &xv[j * d], d, metric, minkowski_p);
            if (dist <= eps) out.push_back(as<i32>(j));
        }
    };

    for (usize i = 0; i < n; ++i) {
        out_labels[i] = 0;
        out_core[i] = 0;
    }

    i32 cluster = 0;
    Array<i32> nbrs(alloc);
    Array<i32> qn(alloc);
    Array<i32> seeds(alloc);
    nbrs.reserve(64);
    qn.reserve(64);
    seeds.reserve(n);

    for (usize i = 0; i < n; ++i) {
        if (out_labels[i] != 0) continue;
        neighbours(i, nbrs);
        if (as<i32>(nbrs.size()) < minpts) {
            out_labels[i] = -1;
            continue;
        }
        ++cluster;
        out_labels[i] = cluster;
        out_core[i] = 1;

        seeds.clear();
        for (usize m = 0; m < nbrs.size(); ++m) seeds.push_back(nbrs[m]);
        for (usize s = 0; s < seeds.size(); ++s) {
            const i32 q = seeds[s];
            if (as<usize>(q) == i) continue;
            if (out_labels[as<usize>(q)] == -1) out_labels[as<usize>(q)] = cluster;
            if (out_labels[as<usize>(q)] != 0) continue;
            out_labels[as<usize>(q)] = cluster;
            neighbours(as<usize>(q), qn);
            if (as<i32>(qn.size()) >= minpts) {
                out_core[as<usize>(q)] = 1;
                for (usize m = 0; m < qn.size(); ++m) {
                    const i32 nb = qn[m];
                    if (out_labels[as<usize>(nb)] == 0 || out_labels[as<usize>(nb)] == -1) {
                        seeds.push_back(nb);
                    }
                }
            }
        }
    }
    return StatsStatus::kOk;
}

StatsStatus silhouette(ArrayRef<f64> x_col_major, usize n, usize d, ArrayRef<i32> labels,
                       DistanceMetric metric, f64 minkowski_p, MutArrayRef<f64> out_s,
                       Allocator* alloc) {
    if (x_col_major.size() != n * d || labels.size() != n || out_s.size() != n) {
        return StatsStatus::kInvalidArgument;
    }
    if (n == 0) return StatsStatus::kOk;

    Array<f64> dist_mat(n * n, 0.0, alloc);
    const StatsStatus st = pdist2(x_col_major, n, x_col_major, n, d, metric, minkowski_p, {},
                                  dist_mat.as_mut_ref(), alloc);
    if (st != StatsStatus::kOk) return st;

    // Group indices by sorted unique labels (CSR offsets).
    Array<i32> uniq(n, i32{0}, alloc);
    for (usize i = 0; i < n; ++i) uniq[i] = labels[i];
    keel::sort(uniq);
    usize K = 0;
    for (usize i = 0; i < n; ++i) {
        if (K == 0 || uniq[i] != uniq[K - 1]) uniq[K++] = uniq[i];
    }
    uniq.resize(K);

    Array<usize> counts(K, usize{0}, alloc);
    Array<usize> lbl_cid(n, usize{0}, alloc);
    for (usize i = 0; i < n; ++i) {
        const usize cid = keel::lower_bound(uniq.as_ref(), labels[i]);
        lbl_cid[i] = cid;
        counts[cid]++;
    }
    Array<usize> offsets(K + 1, usize{0}, alloc);
    for (usize c = 0; c < K; ++c) offsets[c + 1] = offsets[c] + counts[c];
    Array<usize> cursor(K, usize{0}, alloc);
    for (usize c = 0; c < K; ++c) cursor[c] = offsets[c];
    Array<usize> members(n, usize{0}, alloc);
    for (usize i = 0; i < n; ++i) {
        const usize cid = lbl_cid[i];
        members[cursor[cid]++] = i;
    }

    for (usize i = 0; i < n; ++i) {
        const usize my_c = lbl_cid[i];
        const usize my_sz = counts[my_c];
        if (my_sz <= 1) {
            out_s[i] = 0.0;
            continue;
        }
        f64 a = 0.0;
        for (usize pos = offsets[my_c]; pos < offsets[my_c + 1]; ++pos) {
            const usize j = members[pos];
            if (j != i) a += dist_mat[i + j * n];
        }
        a /= as<f64>(my_sz - 1);

        f64 b = Limits<f64>::infinity();
        for (usize cid = 0; cid < K; ++cid) {
            if (cid == my_c) continue;
            f64 m = 0.0;
            for (usize pos = offsets[cid]; pos < offsets[cid + 1]; ++pos) {
                m += dist_mat[i + members[pos] * n];
            }
            m /= as<f64>(counts[cid]);
            if (m < b) b = m;
        }
        if (!is_finite(b)) {
            out_s[i] = Limits<f64>::quiet_nan();
            continue;
        }
        const f64 denom = max(a, b);
        out_s[i] = (denom > 0.0) ? ((b - a) / denom) : 0.0;
    }
    return StatsStatus::kOk;
}

} // namespace keel::stats
