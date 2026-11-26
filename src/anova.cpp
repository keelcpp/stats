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

} // namespace

StatsStatus anova1_grouped(ArrayRef<f64> values, ArrayRef<usize> group_offsets,
                           Anova1Result& out) noexcept {
    const f64 nan = Limits<f64>::quiet_nan();
    out = {nan, nan, 0.0, 0.0, nan, nan};
    if (group_offsets.size() < 3u) return StatsStatus::kOk;
    const usize k_groups = group_offsets.size() - 1u;
    if (group_offsets[k_groups] > values.size()) return StatsStatus::kInvalidArgument;

    usize n_total = 0u;
    usize nonempty_groups = 0u;
    f64 grand_sum = 0.0;
    for (usize g = 0; g < k_groups; ++g) {
        const usize beg = group_offsets[g];
        const usize end = group_offsets[g + 1u];
        if (end < beg || end > values.size()) return StatsStatus::kInvalidArgument;
        usize ng_valid = 0u;
        for (usize i = beg; i < end; ++i) {
            if (!is_nan(values[i])) {
                grand_sum += values[i];
                ++ng_valid;
            }
        }
        if (ng_valid > 0u) {
            ++nonempty_groups;
            n_total += ng_valid;
        }
    }
    if (n_total == 0u || nonempty_groups < 2u) return StatsStatus::kOk;
    const f64 grand_mean = grand_sum / as<f64>(n_total);

    f64 ss_b = 0.0;
    f64 ss_w = 0.0;
    for (usize g = 0; g < k_groups; ++g) {
        const usize beg = group_offsets[g];
        const usize end = group_offsets[g + 1u];
        usize ng_valid = 0u;
        f64 g_sum = 0.0;
        for (usize i = beg; i < end; ++i) {
            if (!is_nan(values[i])) {
                g_sum += values[i];
                ++ng_valid;
            }
        }
        if (ng_valid == 0u) continue;
        const f64 g_mean = g_sum / as<f64>(ng_valid);
        const f64 db = g_mean - grand_mean;
        ss_b += as<f64>(ng_valid) * db * db;
        for (usize i = beg; i < end; ++i) {
            if (!is_nan(values[i])) {
                const f64 dw = values[i] - g_mean;
                ss_w += dw * dw;
            }
        }
    }

    const f64 df_b = as<f64>(nonempty_groups - 1u);
    const f64 df_w = (n_total > nonempty_groups) ? as<f64>(n_total - nonempty_groups) : 0.0;
    out.df_between = df_b;
    out.df_within = df_w;
    out.ss_between = ss_b;
    out.ss_within = ss_w;
    if (df_w <= 0.0) return StatsStatus::kOk;

    const f64 ms_b = ss_b / df_b;
    const f64 ms_w = ss_w / df_w;
    if (ms_w <= 0.0) {
        if (ms_b > 0.0) {
            out.p = 0.0;
            out.f = Limits<f64>::infinity();
        } else {
            out.p = nan;
            out.f = nan;
        }
        return StatsStatus::kOk;
    }
    out.f = ms_b / ms_w;
    out.p = keel::max(0.0, 1.0 - f_cdf(out.f, df_b, df_w));
    return StatsStatus::kOk;
}

StatsStatus anova2(ArrayRef<f64> y_col_major, usize rows, usize cols, Anova2Result& out,
                   Allocator* alloc) {
    if (rows < 2u || cols < 2u || mul_overflows(rows, cols) || y_col_major.size() < rows * cols) {
        return StatsStatus::kInvalidArgument;
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> row_mean(rows, 0.0, scratch.allocator());
    Array<f64> col_mean(cols, 0.0, scratch.allocator());
    f64 grand_sum = 0.0;
    for (usize j = 0; j < cols; ++j) {
        for (usize i = 0; i < rows; ++i) {
            const f64 v = y_col_major[i + j * rows];
            row_mean[i] += v;
            col_mean[j] += v;
            grand_sum += v;
        }
    }
    const f64 n_tot = as<f64>(rows * cols);
    const f64 grand_mean = grand_sum / n_tot;
    for (usize i = 0; i < rows; ++i) row_mean[i] /= as<f64>(cols);
    for (usize j = 0; j < cols; ++j) col_mean[j] /= as<f64>(rows);

    f64 ss_rows = 0.0;
    for (usize i = 0; i < rows; ++i) {
        const f64 d = row_mean[i] - grand_mean;
        ss_rows += as<f64>(cols) * d * d;
    }
    f64 ss_cols = 0.0;
    for (usize j = 0; j < cols; ++j) {
        const f64 d = col_mean[j] - grand_mean;
        ss_cols += as<f64>(rows) * d * d;
    }
    // Direct residual sum of squares avoids catastrophic cancellation (never negative).
    f64 ss_err = 0.0;
    for (usize j = 0; j < cols; ++j) {
        for (usize i = 0; i < rows; ++i) {
            const f64 res = y_col_major[i + j * rows] - row_mean[i] - col_mean[j] + grand_mean;
            ss_err += res * res;
        }
    }

    const f64 df_rows = as<f64>(rows - 1u);
    const f64 df_cols = as<f64>(cols - 1u);
    const f64 df_err = as<f64>((rows - 1u) * (cols - 1u));

    out.df_cols = df_cols;
    out.df_rows = df_rows;
    out.df_err = df_err;
    out.ss_cols = ss_cols;
    out.ss_rows = ss_rows;
    out.ss_err = ss_err;

    const f64 ms_err = ss_err / df_err;
    const f64 nan = Limits<f64>::quiet_nan();
    if (!(ms_err > 0.0)) {
        if (is_nan(ms_err)) {
            out.f_cols = nan;
            out.f_rows = nan;
            out.p_cols = nan;
            out.p_rows = nan;
            return StatsStatus::kOk;
        }
        out.f_cols = (ss_cols > 0.0) ? Limits<f64>::infinity() : nan;
        out.p_cols = (ss_cols > 0.0) ? 0.0 : nan;
        out.f_rows = (ss_rows > 0.0) ? Limits<f64>::infinity() : nan;
        out.p_rows = (ss_rows > 0.0) ? 0.0 : nan;
        return StatsStatus::kOk;
    }

    out.f_cols = (ss_cols / df_cols) / ms_err;
    out.f_rows = (ss_rows / df_rows) / ms_err;
    out.p_cols = keel::max(0.0, 1.0 - f_cdf(out.f_cols, df_cols, df_err));
    out.p_rows = keel::max(0.0, 1.0 - f_cdf(out.f_rows, df_rows, df_err));
    return StatsStatus::kOk;
}

StatsStatus kruskal_wallis(ArrayRef<f64> values, ArrayRef<usize> group_offsets, KruskalResult& out,
                           Allocator* alloc) {
    const f64 nan = Limits<f64>::quiet_nan();
    out = {nan, nan, 0.0, 0.0};
    if (group_offsets.size() < 3u) return StatsStatus::kOk;
    const usize k_groups = group_offsets.size() - 1u;
    if (group_offsets[k_groups] > values.size()) return StatsStatus::kInvalidArgument;

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> valid_vals(scratch.allocator());
    Array<usize> grp_of(scratch.allocator());
    valid_vals.reserve(group_offsets[k_groups]);
    grp_of.reserve(group_offsets[k_groups]);

    usize nonempty_groups = 0u;
    for (usize g = 0; g < k_groups; ++g) {
        const usize beg = group_offsets[g];
        const usize end = group_offsets[g + 1u];
        if (end < beg || end > values.size()) return StatsStatus::kInvalidArgument;
        usize g_cnt = 0u;
        for (usize i = beg; i < end; ++i) {
            if (!is_nan(values[i])) {
                valid_vals.push_back(values[i]);
                grp_of.push_back(g);
                ++g_cnt;
            }
        }
        if (g_cnt > 0u) ++nonempty_groups;
    }
    const usize n = valid_vals.size();
    if (n == 0u || nonempty_groups < 2u) return StatsStatus::kOk;

    Array<usize> ord(n, 0u, scratch.allocator());
    for (usize i = 0; i < n; ++i) ord[i] = i;
    keel::sort(ord.as_mut_ref(), [&](usize a, usize b) noexcept {
        return valid_vals[a] < valid_vals[b];
    });

    Array<f64> ranks(n, 0.0, scratch.allocator());
    f64 tie_sum = 0.0;
    usize i = 0u;
    while (i < n) {
        usize j = i + 1u;
        while (j < n && valid_vals[ord[j]] == valid_vals[ord[i]]) ++j;
        const f64 avg = 0.5 * as<f64>(i + j + 1u);
        for (usize k = i; k < j; ++k) ranks[ord[k]] = avg;
        const usize t = j - i;
        if (t > 1u) {
            const f64 td = as<f64>(t);
            tie_sum += td * td * td - td;
        }
        i = j;
    }

    Array<f64> r_sum(k_groups, 0.0, scratch.allocator());
    Array<usize> ng(k_groups, 0u, scratch.allocator());
    for (usize k = 0; k < n; ++k) {
        r_sum[grp_of[k]] += ranks[k];
        ++ng[grp_of[k]];
    }

    f64 sum_r2_n = 0.0;
    for (usize g = 0; g < k_groups; ++g) {
        if (ng[g] > 0u) sum_r2_n += r_sum[g] * r_sum[g] / as<f64>(ng[g]);
    }

    const f64 nd = as<f64>(n);
    const f64 df = as<f64>(nonempty_groups - 1u);
    out.df = df;
    out.sum_r2_n = sum_r2_n;
    f64 h = (12.0 / (nd * (nd + 1.0))) * sum_r2_n - 3.0 * (nd + 1.0);
    if (tie_sum > 0.0 && nd > 1.0) {
        const f64 tie_corr = 1.0 - tie_sum / (nd * nd * nd - nd);
        if (!(tie_corr > 0.0)) {
            out.h = nan;
            out.p = nan;
            return StatsStatus::kOk;
        }
        h /= tie_corr;
    }
    out.h = max(0.0, h);
    out.p = keel::max(0.0, 1.0 - chi2_cdf(out.h, df));
    return StatsStatus::kOk;
}

StatsStatus friedman(ArrayRef<f64> x_col_major, usize rows, usize cols, i32 reps,
                     FriedmanResult& out, Allocator* alloc) {
    const f64 nan = Limits<f64>::quiet_nan();
    out = {nan, nan, 0.0};
    if (reps < 1) return StatsStatus::kInvalidArgument;
    if (mul_overflows(rows, cols) || x_col_major.size() < rows * cols) {
        return StatsStatus::kInvalidArgument;
    }

    const usize r_cnt = as<usize>(reps);
    if (cols < 2u || rows < 2u * r_cnt || (rows % r_cnt) != 0u) return StatsStatus::kOk;

    const usize n_blocks = rows / r_cnt;
    if (mul_overflows(r_cnt, cols)) return StatsStatus::kInvalidArgument;
    const usize m = r_cnt * cols;

    for (usize i = 0; i < rows * cols; ++i) {
        if (is_nan(x_col_major[i])) {
            out.df = as<f64>(cols - 1u);
            return StatsStatus::kOk;
        }
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> rj(cols, 0.0, scratch.allocator());
    Array<f64> blk_vals(m, 0.0, scratch.allocator());
    Array<f64> rank(m, 0.0, scratch.allocator());
    Array<usize> blk_cols(m, 0u, scratch.allocator());
    Array<usize> ord(m, 0u, scratch.allocator());

    f64 tie_term = 0.0;
    for (usize b = 0; b < n_blocks; ++b) {
        usize idx = 0u;
        for (usize c = 0; c < cols; ++c) {
            for (usize r = 0; r < r_cnt; ++r) {
                blk_vals[idx] = x_col_major[c * rows + (b * r_cnt + r)];
                blk_cols[idx] = c;
                ord[idx] = idx;
                ++idx;
            }
        }
        keel::sort(ord.as_mut_ref(), [&](usize a, usize bb) noexcept {
            return blk_vals[a] < blk_vals[bb];
        });
        usize i = 0u;
        while (i < m) {
            usize j = i + 1u;
            while (j < m && blk_vals[ord[j]] == blk_vals[ord[i]]) ++j;
            const f64 avg = 0.5 * as<f64>(i + j + 1u);
            for (usize t = i; t < j; ++t) rank[ord[t]] = avg;
            const f64 tg = as<f64>(j - i);
            if (tg > 1.0) tie_term += tg * tg * tg - tg;
            i = j;
        }
        for (usize t = 0; t < m; ++t) rj[blk_cols[t]] += rank[t];
    }

    const f64 nd = as<f64>(n_blocks);
    const f64 kd = as<f64>(cols);
    const f64 rd = as<f64>(r_cnt);
    const f64 md = rd * kd;
    f64 sum_rj2 = 0.0;
    for (usize c = 0; c < cols; ++c) sum_rj2 += rj[c] * rj[c];

    const f64 df = kd - 1.0;
    out.df = df;
    f64 q = (12.0 / (nd * kd * rd * rd * (md + 1.0))) * sum_rj2 - 3.0 * nd * (md + 1.0);
    const f64 c_tie = 1.0 - tie_term / (nd * md * (md * md - 1.0));
    if (!(c_tie > 0.0)) {
        out.q = nan;
        out.p = nan;
        return StatsStatus::kOk;
    }
    q /= c_tie;
    out.q = max(0.0, q);
    out.p = keel::max(0.0, 1.0 - chi2_cdf(out.q, df));
    return StatsStatus::kOk;
}

} // namespace keel::stats
