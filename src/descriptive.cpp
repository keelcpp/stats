#include "keel/stats.h"

namespace keel::stats {

namespace {

[[nodiscard]] constexpr f64 qnan() noexcept { return Limits<f64>::quiet_nan(); }

void sort_f64(f64* data, usize n) noexcept {
    if (n < 2) return;
    auto sift_down = [data](usize start, usize end) {
        usize root = start;
        while (2 * root + 1 < end) {
            usize child = 2 * root + 1;
            if (child + 1 < end && data[child] < data[child + 1]) ++child;
            if (data[root] < data[child]) {
                swap(data[root], data[child]);
                root = child;
            } else {
                break;
            }
        }
    };
    for (usize i = n / 2; i > 0; --i) sift_down(i - 1, n);
    for (usize end = n - 1; end > 0; --end) {
        swap(data[0], data[end]);
        sift_down(0, end);
    }
}

} // namespace

f64 mean(ArrayRef<f64> x) noexcept {
    const usize n = x.size();
    if (n == 0) return qnan();
    f64 sum = 0.0;
    for (usize i = 0; i < n; ++i) sum += x[i];
    return sum / static_cast<f64>(n);
}

f64 var(ArrayRef<f64> x, bool population) noexcept {
    const usize n = x.size();
    if (n == 0) return qnan();
    if (n == 1) return population ? 0.0 : qnan();
    const f64 m = mean(x);
    f64 ss = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 d = x[i] - m;
        ss += d * d;
    }
    const f64 denom = population ? static_cast<f64>(n) : static_cast<f64>(n - 1);
    return ss / denom;
}

f64 stddev(ArrayRef<f64> x, bool population) noexcept {
    return sqrt(var(x, population));
}

f64 skewness(ArrayRef<f64> x, bool bias_corrected) noexcept {
    const usize n = x.size();
    if (n == 0) return qnan();
    const f64 m = mean(x);
    f64 m2 = 0.0;
    f64 m3 = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 d = x[i] - m;
        const f64 d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
    }
    const f64 nf = static_cast<f64>(n);
    m2 /= nf;
    m3 /= nf;
    if (m2 == 0.0) return qnan();
    const f64 g1 = m3 / pow(m2, 1.5);
    if (!bias_corrected) return g1;
    if (n < 3) return qnan();
    return g1 * sqrt(nf * (nf - 1.0)) / (nf - 2.0);
}

f64 kurtosis(ArrayRef<f64> x, bool bias_corrected) noexcept {
    const usize n = x.size();
    if (n == 0) return qnan();
    const f64 m = mean(x);
    f64 m2 = 0.0;
    f64 m4 = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 d = x[i] - m;
        const f64 d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    const f64 nf = static_cast<f64>(n);
    m2 /= nf;
    m4 /= nf;
    if (m2 == 0.0) return qnan();
    const f64 k = m4 / (m2 * m2);
    if (!bias_corrected) return k;
    if (n < 4) return qnan();
    return ((nf - 1.0) / ((nf - 2.0) * (nf - 3.0))) * ((nf + 1.0) * k - 3.0 * (nf - 1.0)) + 3.0;
}

f64 cov(ArrayRef<f64> x, ArrayRef<f64> y, bool population) noexcept {
    const usize n = x.size();
    if (n == 0 || y.size() != n) return qnan();
    if (n == 1) return population ? 0.0 : qnan();
    const f64 mx = mean(x);
    const f64 my = mean(y);
    f64 s = 0.0;
    for (usize i = 0; i < n; ++i) s += (x[i] - mx) * (y[i] - my);
    return s / static_cast<f64>(population ? n : (n - 1));
}

f64 corrcoef(ArrayRef<f64> x, ArrayRef<f64> y) noexcept {
    const usize n = x.size();
    if (n < 2 || y.size() != n) return qnan();
    const f64 mx = mean(x);
    const f64 my = mean(y);
    f64 sxx = 0.0;
    f64 syy = 0.0;
    f64 sxy = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 dx = x[i] - mx;
        const f64 dy = y[i] - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    if (sxx <= 0.0 || syy <= 0.0) return qnan();
    return sxy / sqrt(sxx * syy);
}

f64 quantile(ArrayRef<f64> x, f64 p, Allocator* alloc) {
    const usize n = x.size();
    if (n == 0 || is_nan(p) || p < 0.0 || p > 1.0) return qnan();
    Array<f64> sorted(alloc);
    sorted.resize(n);
    for (usize i = 0; i < n; ++i) {
        if (is_nan(x[i])) return qnan();
        sorted[i] = x[i];
    }
    sort_f64(sorted.data(), n);
    if (n == 1) return sorted[0];

    const f64 pos = p * static_cast<f64>(n) - 0.5;
    if (pos <= 0.0) return sorted[0];
    if (pos >= static_cast<f64>(n - 1)) return sorted[n - 1];
    const usize lo = static_cast<usize>(floor(pos));
    const f64 frac = pos - static_cast<f64>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[lo + 1] * frac;
}

f64 median(ArrayRef<f64> x, Allocator* alloc) {
    return quantile(x, 0.5, alloc);
}

f64 iqr(ArrayRef<f64> x, Allocator* alloc) {
    return quantile(x, 0.75, alloc) - quantile(x, 0.25, alloc);
}

void zscore(ArrayRef<f64> x, MutArrayRef<f64> out, bool population) noexcept {
    const usize n = min(x.size(), out.size());
    if (n == 0) return;
    const f64 m = mean(ArrayRef<f64>(x.data(), n));
    const f64 s = stddev(ArrayRef<f64>(x.data(), n), population);
    const f64 denom = (s == 0.0) ? 1.0 : s;
    for (usize i = 0; i < n; ++i) {
        out[i] = (x[i] - m) / denom;
    }
}

} // namespace keel::stats
