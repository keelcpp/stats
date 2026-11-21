#include <keel/stats.h>

#include <cmath>
#include <cstdio>

namespace {

int g_failures = 0;

void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++g_failures;
    }
}

void check_near(keel::f64 got, keel::f64 expected, keel::f64 tol, const char* msg) {
    if (std::fabs(got - expected) > tol || std::isnan(got)) {
        std::fprintf(stderr, "FAIL: %s (got %.15g, expected %.15g, tol %.3g)\n", msg, got, expected, tol);
        ++g_failures;
    }
}

void test_continuous_distributions() {
    // Normal
    check_near(keel::stats::norm_cdf(0.0), 0.5, 1e-15, "norm_cdf(0)");
    check_near(keel::stats::norm_inv(keel::stats::norm_cdf(1.96, 2.0, 3.0), 2.0, 3.0), 1.96, 1e-12, "norm roundtrip");
    auto ns = keel::stats::norm_stat(2.0, 3.0);
    check_near(ns.mean, 2.0, 1e-15, "norm mean");
    check_near(ns.var, 9.0, 1e-15, "norm var");

    // Student's t
    check_near(keel::stats::t_cdf(0.0, 5.0), 0.5, 1e-15, "t_cdf(0)");
    check_near(keel::stats::t_inv(keel::stats::t_cdf(1.5, 7.0), 7.0), 1.5, 1e-10, "t roundtrip");
    check_near(keel::stats::t_inv(0.975, keel::Limits<keel::f64>::infinity()), 1.959963984540054, 1e-12, "t_inv inf");

    // Chi2, F, Gamma, Beta
    check_near(keel::stats::chi2_inv(keel::stats::chi2_cdf(6.5, 4.0), 4.0), 6.5, 1e-10, "chi2 roundtrip");
    check_near(keel::stats::f_inv(keel::stats::f_cdf(2.5, 5.0, 10.0), 5.0, 10.0), 2.5, 1e-10, "f roundtrip");
    check_near(keel::stats::gam_inv(keel::stats::gam_cdf(3.2, 2.5, 1.5), 2.5, 1.5), 3.2, 1e-10, "gam roundtrip");
    check_near(keel::stats::beta_inv(keel::stats::beta_cdf(0.4, 2.0, 5.0), 2.0, 5.0), 0.4, 1e-10, "beta roundtrip");

    // Exp, Weibull, Lognormal, Uniform, Rayleigh, GEV
    check_near(keel::stats::exp_inv(keel::stats::exp_cdf(2.0, 3.0), 3.0), 2.0, 1e-13, "exp roundtrip");
    check_near(keel::stats::wbl_inv(keel::stats::wbl_cdf(1.5, 2.0, 3.0), 2.0, 3.0), 1.5, 1e-13, "wbl roundtrip");
    check_near(keel::stats::logn_inv(keel::stats::logn_cdf(2.5, 0.5, 0.8), 0.5, 0.8), 2.5, 1e-12, "logn roundtrip");
    check_near(keel::stats::unif_inv(keel::stats::unif_cdf(3.5, 1.0, 5.0), 1.0, 5.0), 3.5, 1e-14, "unif roundtrip");
    check_near(keel::stats::rayl_inv(keel::stats::rayl_cdf(2.0, 1.5), 1.5), 2.0, 1e-13, "rayl roundtrip");
    check_near(keel::stats::gev_inv(keel::stats::gev_cdf(1.2, 0.2, 1.5, 0.5), 0.2, 1.5, 0.5), 1.2, 1e-12, "gev roundtrip");
}

void test_discrete_and_descriptive() {
    // Poisson
    const keel::f64 pc = keel::stats::poiss_cdf(3.0, 4.0);
    check_near(keel::stats::poiss_inv(pc, 4.0), 3.0, 1e-14, "poiss roundtrip");

    // Binomial
    const keel::f64 bc = keel::stats::bino_cdf(4.0, 10.0, 0.4);
    check_near(keel::stats::bino_inv(bc, 10.0, 0.4), 4.0, 1e-14, "bino roundtrip");
    check_near(keel::stats::bino_pdf(0.0, 5.0, 0.0), 1.0, 1e-15, "bino_pdf p=0");

    // Descriptive
    const keel::f64 xs[5] = {2.0, 4.0, 4.0, 4.0, 6.0};
    check_near(keel::stats::mean(xs), 4.0, 1e-15, "mean");
    check_near(keel::stats::var(xs, true), 1.6, 1e-15, "pop var");
    check_near(keel::stats::var(xs, false), 2.0, 1e-15, "sample var");
    check_near(keel::stats::median(xs), 4.0, 1e-15, "median");
    check_near(keel::stats::corrcoef(xs, xs), 1.0, 1e-14, "corrcoef self");
    check(std::isnan(keel::stats::mean({})), "empty mean is NaN");
}

void test_distance_and_linkage() {
    using namespace keel;
    namespace st = keel::stats;

    // 4 points in 2D (col-major 4x2): (0,0), (0,1), (10,0), (10,1)
    const f64 x[8] = {0.0, 0.0, 10.0, 10.0, 0.0, 1.0, 0.0, 1.0};
    f64 d[6]{};
    check(st::pdist(x, 4, 2, st::DistanceMetric::kEuclidean, 2.0, {}, d) == st::StatsStatus::kOk,
          "pdist ok");
    check_near(d[0], 1.0, 1e-14, "pdist(0,1)");
    check_near(d[5], 1.0, 1e-14, "pdist(2,3)");

    f64 sq[16]{}, back[6]{};
    check(st::squareform_to_matrix(d, 4, sq) == st::StatsStatus::kOk, "squareform to mat");
    check(st::squareform_to_condensed(sq, 4, back) == st::StatsStatus::kOk, "squareform back");
    for (usize i = 0; i < 6; ++i) check_near(back[i], d[i], 1e-14, "squareform roundtrip");

    // Linkage -> 2 clusters should group {0,1} and {2,3}
    f64 z[3 * 3]{};
    check(st::linkage(d, 4, st::LinkageMethod::kSingle, z) == st::StatsStatus::kOk, "linkage ok");
    i32 labels[4]{};
    check(st::cluster_maxclust(z, 3, 2, labels) == st::StatsStatus::kOk, "cluster_maxclust ok");
    check(labels[0] == labels[1] && labels[2] == labels[3] && labels[0] != labels[2],
          "cluster separation");

    f64 dcoph[6]{}, corr = 0.0;
    check(st::cophenet(z, 3, d, dcoph, corr) == st::StatsStatus::kOk, "cophenet ok");
    check(corr > 0.99, "cophenet high corr");

    f64 inc[3 * 4]{};
    check(st::inconsistent(z, 3, 2, inc) == st::StatsStatus::kOk, "inconsistent ok");

    // QR Mahalanobis distance on well-conditioned 4x2 points
    const f64 ref_x[8] = {0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0};
    const f64 qry[2] = {0.75, 1.0}; // mean of ref_x
    f64 md2[1]{};
    check(st::mahal_qr(qry, 1, ref_x, 4, 2, md2) == st::StatsStatus::kOk, "mahal_qr ok");
    check_near(md2[0], 0.0, 1e-12, "mahal_qr at mean == 0");
}

void test_lasso_and_anova() {
    using namespace keel;
    namespace st = keel::stats;

    // LASSO recovers y = 2 + 3*x1 + 0*x2 with small lambda
    const f64 x[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 1.0, 4.0, 3.0, 5.0};
    const f64 y[5] = {5.0, 8.0, 11.0, 14.0, 17.0};
    const f64 lam[2] = {1e-5, 100.0};
    f64 b[4]{}, interc[2]{};
    check(st::lasso(x, 5, 2, y, lam, 1.0, b, interc) == st::StatsStatus::kOk, "lasso ok");
    check_near(b[0], 3.0, 1e-3, "lasso coef 1");
    check_near(b[1], 0.0, 1e-3, "lasso coef 2");
    check_near(interc[0], 2.0, 1e-2, "lasso intercept");
    check_near(b[2], 0.0, 1e-12, "lasso large lambda shrinks to 0");
    check_near(b[3], 0.0, 1e-12, "lasso large lambda shrinks to 0");

    // One-way ANOVA + Kruskal-Wallis
    const f64 vals[9] = {1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0};
    const usize offs[4] = {0, 3, 6, 9};
    st::Anova1Result a1;
    check(st::anova1_grouped(vals, offs, a1) == st::StatsStatus::kOk, "anova1 ok");
    check(a1.p < 1e-4 && a1.f > 100.0, "anova1 detects separated groups");

    st::KruskalResult kw;
    check(st::kruskal_wallis(vals, offs, kw) == st::StatsStatus::kOk, "kruskal_wallis ok");
    check(kw.p < 0.05 && kw.df == 2.0, "kruskal_wallis detects separated groups");

    // Two-way ANOVA & Friedman
    const f64 m2[9] = {1.0, 2.0, 3.0, 11.0, 12.0, 13.0, 21.0, 22.0, 23.0};
    st::Anova2Result a2;
    check(st::anova2(m2, 3, 3, a2) == st::StatsStatus::kOk, "anova2 ok");
    check_near(a2.ss_err, 0.0, 1e-12, "anova2 additive exact");

    st::FriedmanResult fr;
    check(st::friedman(m2, 3, 3, 1, fr) == st::StatsStatus::kOk, "friedman ok");
    check_near(fr.q, 6.0, 1e-12, "friedman Q");
}

} // namespace

int main() {
    test_continuous_distributions();
    test_discrete_and_descriptive();
    test_distance_and_linkage();
    test_lasso_and_anova();
    if (g_failures == 0) {
        std::printf("ALL keel::stats TESTS PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "%d FAILURE(S)\n", g_failures);
    return 1;
}
