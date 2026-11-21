#include "keel/stats.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"

namespace keel::stats {

namespace {

inline usize pdist_index(usize i, usize j, usize n) noexcept {
    if (i > j) keel::swap(i, j);
    return i * n - (i * (i + 1u)) / 2u + (j - i - 1u);
}

inline f64 lance_williams(LinkageMethod m, f64 d_ak, f64 d_bk, f64 d_ab, i32 n_a, i32 n_b,
                          i32 n_k) noexcept {
    switch (m) {
        case LinkageMethod::kSingle:
            return keel::min(d_ak, d_bk);
        case LinkageMethod::kComplete:
            return keel::max(d_ak, d_bk);
        case LinkageMethod::kAverage:
            return (as<f64>(n_a) * d_ak + as<f64>(n_b) * d_bk) / as<f64>(n_a + n_b);
        case LinkageMethod::kWeighted:
            return 0.5 * (d_ak + d_bk);
        case LinkageMethod::kCentroid: {
            const f64 na = as<f64>(n_a);
            const f64 nb = as<f64>(n_b);
            const f64 v = (na * d_ak * d_ak + nb * d_bk * d_bk) / (na + nb) -
                          na * nb * d_ab * d_ab / ((na + nb) * (na + nb));
            return keel::sqrt(v > 0.0 ? v : 0.0);
        }
        case LinkageMethod::kMedian: {
            const f64 v = 0.5 * d_ak * d_ak + 0.5 * d_bk * d_bk - 0.25 * d_ab * d_ab;
            return keel::sqrt(v > 0.0 ? v : 0.0);
        }
        case LinkageMethod::kWard: {
            const f64 na = as<f64>(n_a);
            const f64 nb = as<f64>(n_b);
            const f64 nk = as<f64>(n_k);
            const f64 v =
                ((na + nk) * d_ak * d_ak + (nb + nk) * d_bk * d_bk - nk * d_ab * d_ab) /
                (na + nb + nk);
            return keel::sqrt(v > 0.0 ? v : 0.0);
        }
    }
    return 0.0;
}

void compact_labels(ArrayRef<i32> raw_labels, MutArrayRef<i32> out_labels, Allocator* alloc) {
    const usize n = raw_labels.size();
    Array<i32> remap(n + 1u, 0, alloc);
    i32 next = 0;
    for (usize i = 0; i < n; ++i) {
        const usize lbl = as<usize>(raw_labels[i]);
        if (remap[lbl] == 0) remap[lbl] = ++next;
        out_labels[i] = remap[lbl];
    }
}

void assign_leaves(i32 id, usize n, ArrayRef<i32> a, ArrayRef<i32> b, i32 lab,
                   MutArrayRef<i32> labels) noexcept {
    if (as<usize>(id) < n) {
        labels[as<usize>(id)] = lab;
        return;
    }
    const usize idx = as<usize>(id) - n;
    assign_leaves(a[idx], n, a, b, lab, labels);
    assign_leaves(b[idx], n, a, b, lab, labels);
}

void walk_cutoff(i32 id, usize n, ArrayRef<i32> a, ArrayRef<i32> b, ArrayRef<f64> score,
                 f64 cutoff, i32& next_label, MutArrayRef<i32> labels) noexcept {
    if (as<usize>(id) < n) {
        labels[as<usize>(id)] = ++next_label;
        return;
    }
    const usize idx = as<usize>(id) - n;
    if (score[idx] <= cutoff) {
        ++next_label;
        assign_leaves(id, n, a, b, next_label, labels);
    } else {
        walk_cutoff(a[idx], n, a, b, score, cutoff, next_label, labels);
        walk_cutoff(b[idx], n, a, b, score, cutoff, next_label, labels);
    }
}

void collect_depths(i32 id, usize n, ArrayRef<i32> a, ArrayRef<i32> b, ArrayRef<f64> d, i32 rem,
                    Array<f64>& acc) {
    if (as<usize>(id) < n || rem == 0) return;
    const usize idx = as<usize>(id) - n;
    acc.push_back(d[idx]);
    collect_depths(a[idx], n, a, b, d, rem - 1, acc);
    collect_depths(b[idx], n, a, b, d, rem - 1, acc);
}

} // namespace

StatsStatus linkage(ArrayRef<f64> pdist_condensed, usize n, LinkageMethod method,
                    MutArrayRef<f64> out_z_col_major, Allocator* alloc) {
    if (n < 2u) return StatsStatus::kOk;
    const usize expected = n * (n - 1u) / 2u;
    const usize rows = n - 1u;
    if (pdist_condensed.size() != expected || out_z_col_major.size() < rows * 3u) {
        return StatsStatus::kInvalidArgument;
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> dm(n * n, 0.0, scratch.allocator());
    for (usize i = 0; i < n; ++i) {
        for (usize j = i + 1u; j < n; ++j) {
            const f64 d = pdist_condensed[pdist_index(i, j, n)];
            dm[i * n + j] = d;
            dm[j * n + i] = d;
        }
    }

    Array<i32> ids(n, 0, scratch.allocator());
    Array<i32> sizes(n, 1, scratch.allocator());
    for (usize i = 0; i < n; ++i) ids[i] = as<i32>(i);
    Array<f64> newrow(n, 0.0, scratch.allocator());

    i32 next_id = as<i32>(n);
    for (usize step = 0; step < rows; ++step) {
        const usize m = ids.size();
        usize bi = 0u;
        usize bj = 1u;
        f64 bd = Limits<f64>::infinity();
        for (usize i = 0; i < m; ++i) {
            for (usize j = i + 1u; j < m; ++j) {
                if (dm[i * n + j] <= bd) {
                    bd = dm[i * n + j];
                    bi = i;
                    bj = j;
                }
            }
        }

        i32 a_id = ids[bi];
        i32 b_id = ids[bj];
        if (a_id > b_id) keel::swap(a_id, b_id);
        out_z_col_major[0u * rows + step] = as<f64>(a_id + 1);
        out_z_col_major[1u * rows + step] = as<f64>(b_id + 1);
        out_z_col_major[2u * rows + step] = bd;

        const i32 n_a = sizes[bi];
        const i32 n_b = sizes[bj];
        for (usize k = 0; k < m; ++k) {
            if (k == bi || k == bj) {
                newrow[k] = 0.0;
                continue;
            }
            newrow[k] = lance_williams(method, dm[bi * n + k], dm[bj * n + k], bd, n_a, n_b, sizes[k]);
        }

        ids[bi] = next_id++;
        sizes[bi] = n_a + n_b;
        for (usize k = 0; k < m; ++k) {
            dm[bi * n + k] = newrow[k];
            dm[k * n + bi] = newrow[k];
        }
        ids.erase(bj);
        sizes.erase(bj);
        for (usize r = 0; r < m; ++r) {
            for (usize c = bj; c + 1u < m; ++c) dm[r * n + c] = dm[r * n + c + 1u];
        }
        for (usize r = bj; r + 1u < m; ++r) {
            for (usize c = 0; c + 1u < m; ++c) dm[r * n + c] = dm[(r + 1u) * n + c];
        }
    }
    return StatsStatus::kOk;
}

StatsStatus inconsistent(ArrayRef<f64> z_col_major, usize num_merges, i32 depth,
                         MutArrayRef<f64> out_col_major, Allocator* alloc) {
    if (z_col_major.size() < num_merges * 3u || out_col_major.size() < num_merges * 4u) {
        return StatsStatus::kInvalidArgument;
    }
    if (depth <= 0) depth = 2;
    const usize n = num_merges + 1u;

    ArenaTemp scratch = get_scratch(alloc);
    Array<i32> a(num_merges, 0, scratch.allocator());
    Array<i32> b(num_merges, 0, scratch.allocator());
    Array<f64> d(num_merges, 0.0, scratch.allocator());
    for (usize i = 0; i < num_merges; ++i) {
        a[i] = as<i32>(z_col_major[0u * num_merges + i]) - 1;
        b[i] = as<i32>(z_col_major[1u * num_merges + i]) - 1;
        d[i] = z_col_major[2u * num_merges + i];
    }

    Array<f64> ds(scratch.allocator());
    for (usize i = 0; i < num_merges; ++i) {
        ds.clear();
        collect_depths(as<i32>(n + i), n, a.as_ref(), b.as_ref(), d.as_ref(), depth, ds);
        const f64 cnt = as<f64>(ds.size());
        f64 mu = 0.0;
        for (usize k = 0; k < ds.size(); ++k) mu += ds[k];
        mu /= cnt;
        f64 var_sum = 0.0;
        for (usize k = 0; k < ds.size(); ++k) {
            const f64 diff = ds[k] - mu;
            var_sum += diff * diff;
        }
        const f64 sd = (cnt > 1.0) ? keel::sqrt(var_sum / (cnt - 1.0)) : 0.0;
        const f64 inc = (sd > 0.0) ? ((d[i] - mu) / sd) : 0.0;
        out_col_major[0u * num_merges + i] = mu;
        out_col_major[1u * num_merges + i] = sd;
        out_col_major[2u * num_merges + i] = cnt;
        out_col_major[3u * num_merges + i] = inc;
    }
    return StatsStatus::kOk;
}

StatsStatus cluster_maxclust(ArrayRef<f64> z_col_major, usize num_merges, i32 max_clust,
                             MutArrayRef<i32> out_labels, Allocator* alloc) {
    const usize n = num_merges + 1u;
    if (z_col_major.size() < num_merges * 3u || out_labels.size() < n) {
        return StatsStatus::kInvalidArgument;
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<i32> labels(n, 0, scratch.allocator());

    if (max_clust >= 1 && as<usize>(max_clust) <= n) {
        const usize merges = n - as<usize>(max_clust);
        Array<i32> parent(n + num_merges, 0, scratch.allocator());
        for (usize i = 0; i < parent.size(); ++i) parent[i] = as<i32>(i);
        auto find_root = [&](i32 x) -> i32 {
            while (parent[as<usize>(x)] != x) {
                parent[as<usize>(x)] = parent[as<usize>(parent[as<usize>(x)])];
                x = parent[as<usize>(x)];
            }
            return x;
        };
        for (usize i = 0; i < merges; ++i) {
            const i32 a_i = as<i32>(z_col_major[0u * num_merges + i]) - 1;
            const i32 b_i = as<i32>(z_col_major[1u * num_merges + i]) - 1;
            const i32 newid = as<i32>(n + i);
            i32 rx = find_root(a_i);
            i32 ry = find_root(newid);
            if (rx != ry) parent[as<usize>(rx)] = ry;
            rx = find_root(b_i);
            ry = find_root(newid);
            if (rx != ry) parent[as<usize>(rx)] = ry;
        }
        Array<i32> root2lab(scratch.allocator());
        for (usize i = 0; i < n; ++i) {
            const i32 r = find_root(as<i32>(i));
            i32 lab = -1;
            for (usize j = 0; j < root2lab.size(); ++j) {
                if (root2lab[j] == r) {
                    lab = as<i32>(j);
                    break;
                }
            }
            if (lab < 0) {
                lab = as<i32>(root2lab.size());
                root2lab.push_back(r);
            }
            labels[i] = lab + 1;
        }
    } else {
        for (usize i = 0; i < n; ++i) labels[i] = as<i32>(i + 1u);
    }

    compact_labels(labels.as_ref(), out_labels, scratch.allocator());
    return StatsStatus::kOk;
}

StatsStatus cluster_cutoff(ArrayRef<f64> z_col_major, usize num_merges, f64 cutoff,
                           bool by_distance, i32 depth, MutArrayRef<i32> out_labels,
                           Allocator* alloc) {
    const usize n = num_merges + 1u;
    if (z_col_major.size() < num_merges * 3u || out_labels.size() < n) {
        return StatsStatus::kInvalidArgument;
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<i32> labels(n, 0, scratch.allocator());
    if (!(cutoff > 0.0) || num_merges == 0u) {
        for (usize i = 0; i < n; ++i) out_labels[i] = as<i32>(i + 1u);
        return StatsStatus::kOk;
    }

    Array<i32> a(num_merges, 0, scratch.allocator());
    Array<i32> b(num_merges, 0, scratch.allocator());
    Array<f64> score(num_merges, 0.0, scratch.allocator());
    for (usize i = 0; i < num_merges; ++i) {
        a[i] = as<i32>(z_col_major[0u * num_merges + i]) - 1;
        b[i] = as<i32>(z_col_major[1u * num_merges + i]) - 1;
        score[i] = z_col_major[2u * num_merges + i];
    }

    if (!by_distance) {
        Array<f64> inc_mat(num_merges * 4u, 0.0, scratch.allocator());
        const StatsStatus st =
            inconsistent(z_col_major, num_merges, depth, inc_mat.as_mut_ref(), scratch.allocator());
        if (st != StatsStatus::kOk) return st;
        for (usize i = 0; i < num_merges; ++i) score[i] = inc_mat[3u * num_merges + i];
    }

    i32 next_label = 0;
    walk_cutoff(as<i32>(n + num_merges - 1u), n, a.as_ref(), b.as_ref(), score.as_ref(), cutoff,
                next_label, labels.as_mut_ref());
    compact_labels(labels.as_ref(), out_labels, scratch.allocator());
    return StatsStatus::kOk;
}

StatsStatus cophenet(ArrayRef<f64> z_col_major, usize num_merges, ArrayRef<f64> pdist_condensed,
                     MutArrayRef<f64> out_dcoph, f64& out_corr, Allocator* alloc) {
    const usize n = num_merges + 1u;
    const usize yn = n * (n - 1u) / 2u;
    if (z_col_major.size() < num_merges * 3u || pdist_condensed.size() != yn ||
        (!out_dcoph.empty() && out_dcoph.size() < yn)) {
        return StatsStatus::kInvalidArgument;
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<Array<i32>> members(scratch.allocator());
    members.reserve(n + num_merges);
    for (usize i = 0; i < n + num_merges; ++i) members.push_back(Array<i32>(scratch.allocator()));
    for (usize i = 0; i < n; ++i) members[i].push_back(as<i32>(i));

    Array<f64> dcoph(yn, 0.0, scratch.allocator());
    for (usize i = 0; i < num_merges; ++i) {
        const usize a_i = as<usize>(as<i32>(z_col_major[0u * num_merges + i]) - 1);
        const usize b_i = as<usize>(as<i32>(z_col_major[1u * num_merges + i]) - 1);
        const f64 dist = z_col_major[2u * num_merges + i];
        const usize newid = n + i;
        const Array<i32>& lhs = members[a_i];
        const Array<i32>& rhs = members[b_i];
        for (usize li = 0; li < lhs.size(); ++li) {
            for (usize ri = 0; ri < rhs.size(); ++ri) {
                dcoph[pdist_index(as<usize>(lhs[li]), as<usize>(rhs[ri]), n)] = dist;
            }
        }
        members[newid].append(lhs.as_ref());
        members[newid].append(rhs.as_ref());
    }

    f64 mx = 0.0;
    f64 my = 0.0;
    for (usize i = 0; i < yn; ++i) {
        mx += pdist_condensed[i];
        my += dcoph[i];
    }
    mx /= as<f64>(yn);
    my /= as<f64>(yn);
    f64 sxy = 0.0, sx2 = 0.0, sy2 = 0.0;
    for (usize i = 0; i < yn; ++i) {
        const f64 dx = pdist_condensed[i] - mx;
        const f64 dy = dcoph[i] - my;
        sxy += dx * dy;
        sx2 += dx * dx;
        sy2 += dy * dy;
    }
    const f64 denom = keel::sqrt(sx2) * keel::sqrt(sy2);
    out_corr = (denom > 0.0) ? (sxy / denom) : 0.0;

    if (!out_dcoph.empty()) {
        for (usize i = 0; i < yn; ++i) out_dcoph[i] = dcoph[i];
    }
    return StatsStatus::kOk;
}

} // namespace keel::stats
