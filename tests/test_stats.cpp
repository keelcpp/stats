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

} // namespace

int main() {
    test_continuous_distributions();
    test_discrete_and_descriptive();
    if (g_failures == 0) {
        std::printf("ALL keel::stats TESTS PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "%d FAILURE(S)\n", g_failures);
    return 1;
}
