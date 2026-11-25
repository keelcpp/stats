#include <keel/stats.h>
#include <keel/format.h>
#include <keel/math.h>

namespace {

keel::i32 g_failures = 0;

void check(bool cond, const char* msg) {
    if (!cond) {
        keel::eprintln("FAIL: {}", msg);
        ++g_failures;
    }
}

void check_near(keel::f64 got, keel::f64 expected, keel::f64 tol, const char* msg) {
    if (keel::abs(got - expected) > tol || keel::is_nan(got)) {
        keel::eprintln("FAIL: {} (got {}, expected {}, tol {})", msg, got, expected, tol);
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
    check(keel::is_nan(keel::stats::mean({})), "empty mean is NaN");
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

void test_regress_pca_lda_cluster_tests() {
    using namespace keel;
    namespace st = keel::stats;

    // 1. regress / lscov / ridge / glmfit / glmval
    const f64 x_ols[10] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    const f64 y_ols[5] = {3.1, 4.9, 7.1, 8.9, 11.0};
    f64 beta[2]{}, bint[4]{}, r[5]{}, rint[10]{};
    st::RegressStats rs{};
    check(st::regress(y_ols, x_ols, 5, 2, 0.05, beta, bint, r, rint, rs) == st::StatsStatus::kOk,
          "regress ok");
    check_near(beta[0], 1.0, 0.1, "regress intercept");
    check_near(beta[1], 2.0, 0.05, "regress slope");
    check(rs.r2 > 0.99, "regress r2");

    f64 lx[2]{}, lstd[2]{}, lmse = 0.0, lS[4]{};
    check(st::lscov(x_ols, y_ols, 5, 2, {}, false, lx, lstd, lmse, lS) == st::StatsStatus::kOk,
          "lscov ok");
    check_near(lx[1], beta[1], 1e-12, "lscov matches regress");

    const f64 x1[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    const f64 kv[1] = {0.0};
    f64 rb[2]{};
    check(st::ridge(y_ols, x1, 5, 1, kv, false, rb) == st::StatsStatus::kOk, "ridge ok");
    check_near(rb[1], beta[1], 1e-10, "ridge k=0 matches OLS");

    st::GlmFitOut gfit;
    check(st::glmfit(x1, y_ols, 5, 1, st::GlmDist::kNormal, st::GlmLink::kIdentity, gfit) ==
              st::StatsStatus::kOk,
          "glmfit normal ok");
    check_near(gfit.beta[1], beta[1], 1e-8, "glmfit normal slope");
    f64 gpred[5]{};
    check(st::glmval(gfit.beta.as_ref(), x1, 5, 1, st::GlmLink::kIdentity, gpred) ==
              st::StatsStatus::kOk,
          "glmval ok");
    check_near(gpred[2], 7.0, 0.1, "glmval pred");

    // 2. pca / pcacov / pcares / lda_classify / nearcorr
    const f64 x_pca[8] = {1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0};
    f64 coeff[4]{}, score[8]{}, latent[2]{}, tsq[4]{}, expl[2]{}, mu[2]{};
    check(st::pca(x_pca, 4, 2, coeff, score, latent, tsq, expl, mu) == st::StatsStatus::kOk,
          "pca ok");
    check_near(expl[0], 100.0, 1e-10, "pca collinear 100% explained");
    check_near(latent[1], 0.0, 1e-12, "pca second latent 0");

    f64 res[8]{}, recon[8]{};
    check(st::pcares(x_pca, 4, 2, 1, res, recon) == st::StatsStatus::kOk, "pcares ok");
    check_near(res[0], 0.0, 1e-12, "pcares 1 component exact");

    // LDA on two well-separated 2D classes
    const f64 tr[16] = {0.0, 0.2, -0.2, 0.1, 10.0, 10.2, 9.8, 10.1,
                        0.1, -0.1, 0.2, -0.2, 10.1, 9.9, 10.2, 9.8};
    const f64 grp[8] = {1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0};
    const f64 smp[4] = {0.05, 9.95, 0.0, 10.0};
    st::LdaResult lda;
    check(st::lda_classify(smp, 2, tr, 8, 2, grp, st::DiscrimType::kLinear, lda) ==
              st::StatsStatus::kOk,
          "lda ok");
    check_near(lda.pred[0], 1.0, 1e-12, "lda class 1");
    check_near(lda.pred[1], 2.0, 1e-12, "lda class 2");
    check_near(lda.train_error, 0.0, 1e-12, "lda 0 train error");

    // nearcorr on non-PSD symmetric unit-diagonal matrix
    const f64 bad_corr[9] = {1.0, 0.9, 0.9, 0.9, 1.0, -0.9, 0.9, -0.9, 1.0};
    f64 nc[9]{};
    check(st::nearcorr(bad_corr, 3, {}, 1e-8, 200, nc) == st::StatsStatus::kOk, "nearcorr ok");
    check_near(nc[0], 1.0, 1e-12, "nearcorr unit diag 0");
    check_near(nc[4], 1.0, 1e-12, "nearcorr unit diag 1");
    check_near(nc[8], 1.0, 1e-12, "nearcorr unit diag 2");

    // 3. kmedoids_pam_step / dbscan / silhouette
    const f64 pts[12] = {0.0, 0.1, 0.2, 10.0, 10.1, 10.2,
                         0.0, 0.1, -0.1, 10.0, 10.1, 9.9};
    const i32 init_m[2] = {0, 3};
    i32 km_idx[6]{}, km_med[2]{};
    f64 km_sumd[2]{}, km_tot = 0.0;
    i32 km_iters = 0;
    check(st::kmedoids_pam_step(pts, 6, 2, init_m, 50, st::DistanceMetric::kEuclidean, 2.0,
                                km_idx, km_med, km_sumd, km_tot, km_iters) == st::StatsStatus::kOk,
          "kmedoids ok");
    check(km_idx[0] == km_idx[1] && km_idx[3] == km_idx[4] && km_idx[0] != km_idx[3],
          "kmedoids separates 2 clusters");

    i32 db_lbl[6]{};
    u8 db_core[6]{};
    check(st::dbscan(pts, 6, 2, 1.0, 2, st::DistanceMetric::kEuclidean, 2.0, false, db_lbl,
                     db_core) == st::StatsStatus::kOk,
          "dbscan ok");
    check(db_lbl[0] == 1 && db_lbl[2] == 1 && db_lbl[3] == 2 && db_lbl[5] == 2,
          "dbscan separates 2 clusters");

    f64 sil[6]{};
    check(st::silhouette(pts, 6, 2, db_lbl, st::DistanceMetric::kEuclidean, 2.0, sil) ==
              st::StatsStatus::kOk,
          "silhouette ok");
    check(sil[0] > 0.9 && sil[3] > 0.9, "silhouette high for well-separated clusters");

    // 4. adtest / dwtest
    const f64 norm_samp[10] = {-1.5, -1.0, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 1.0, 1.5};
    st::AdTestResult ad{};
    check(st::adtest(norm_samp, 0.05, ad) == st::StatsStatus::kOk, "adtest ok");
    check(ad.h == 0 && ad.p > 0.05, "adtest accepts normal sample");

    st::DwTestResult dw{};
    check(st::dwtest(r, x_ols, 5, 2, true, dw) == st::StatsStatus::kOk, "dwtest exact ok");
    check(dw.dw > 0.0 && dw.dw < 4.0 && dw.p_both >= 0.0 && dw.p_both <= 1.0, "dwtest valid range");
}

void test_copulas_and_mvdist() {
    namespace st = keel::stats;
    using keel::f64;
    // 1. Copulas
    const f64 u[4] = {0.25, 0.75, 0.5, 0.5}; // 2 x 2 col-major: rows (0.25, 0.5), (0.75, 0.5)
    const f64 r_id[4] = {1.0, 0.0, 0.0, 1.0};
    const f64 r_pos[4] = {1.0, 0.5, 0.5, 1.0};
    f64 out[2]{};

    check(st::copula_pdf_gaussian(u, 2, 2, r_id, out) == st::StatsStatus::kOk, "copula gauss id");
    check_near(out[0], 1.0, 1e-12, "copula gauss indep pdf = 1");
    check(st::copula_cdf_gaussian_2d(u, 2, 0.0, out) == st::StatsStatus::kOk, "copula gauss cdf");
    check_near(out[0], 0.125, 1e-10, "copula gauss indep cdf");
    check(st::copula_cdf_gaussian_2d(u, 2, 0.5, out) == st::StatsStatus::kOk, "copula gauss corr");
    check(out[0] > 0.125, "positive rho increases copula cdf");

    check(st::copula_pdf_t(u, 2, 2, r_pos, 5.0, out) == st::StatsStatus::kOk, "copula t pdf");
    check(out[0] > 0.0 && out[1] > 0.0, "copula t pdf positive");

    check(st::copula_cdf_clayton(u, 2, 2.0, out) == st::StatsStatus::kOk, "clayton cdf");
    check(out[0] > 0.125, "clayton positive dependence");
    check(st::copula_pdf_clayton(u, 2, 2.0, out) == st::StatsStatus::kOk, "clayton pdf");
    check(out[0] > 0.0, "clayton pdf > 0");

    check(st::copula_cdf_frank(u, 2, 3.0, out) == st::StatsStatus::kOk, "frank cdf");
    check(out[0] > 0.125, "frank positive dependence");
    check(st::copula_pdf_frank(u, 2, 3.0, out) == st::StatsStatus::kOk, "frank pdf");
    check(out[0] > 0.0, "frank pdf > 0");

    check(st::copula_cdf_gumbel(u, 2, 2.0, out) == st::StatsStatus::kOk, "gumbel cdf");
    check(out[0] > 0.125, "gumbel positive dependence");
    check(st::copula_pdf_gumbel(u, 2, 2.0, out) == st::StatsStatus::kOk, "gumbel pdf");
    check(out[0] > 0.0, "gumbel pdf > 0");

    // 2. mvnpdf / mvtpdf / mnpdf
    const f64 x0[4] = {0.0, 1.0, 0.0, -1.0}; // 2 x 2 col-major: (0,0), (1,-1)
    const f64 sig[4] = {1.0, 0.0, 0.0, 1.0};
    check(st::mvnpdf_full(x0, 2, 2, sig, out) == st::StatsStatus::kOk, "mvnpdf_full ok");
    check_near(out[0], 1.0 / (2.0 * 3.14159265358979323846), 1e-12, "mvnpdf at 0");
    f64 out_d[2]{};
    const f64 dv[2] = {1.0, 1.0};
    check(st::mvnpdf_diag(x0, 2, 2, dv, out_d) == st::StatsStatus::kOk, "mvnpdf_diag ok");
    check_near(out[1], out_d[1], 1e-12, "mvnpdf full == diag");

    const f64 df[2] = {5.0, 5.0};
    check(st::mvtpdf_full(x0, 2, 2, r_pos, df, out) == st::StatsStatus::kOk, "mvtpdf_full ok");
    check(out[0] > out[1] && out[1] > 0.0, "mvtpdf mode at origin");

    const f64 counts[3] = {2.0, 1.0, 1.0}; // 1 x 3
    const f64 probs[3] = {0.5, 0.25, 0.25};
    f64 mn_out[1]{};
    check(st::mnpdf_rows(counts, 1, 3, probs, 1, mn_out) == st::StatsStatus::kOk, "mnpdf ok");
    // 4! / (2! 1! 1!) * 0.5^2 * 0.25 * 0.25 = 12 * 0.015625 = 0.1875
    check_near(mn_out[0], 0.1875, 1e-12, "mnpdf exact value");
}

void test_hypothesis_and_fits() {
    namespace st = keel::stats;
    using keel::f64;
    using keel::i32;

    const f64 x[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    const f64 y[8] = {10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0};

    // 1. ttest_1sample & ttest_2sample
    st::TTestResult tt{};
    check(st::ttest_1sample(x, 4.5, 0.05, st::TestTail::kBoth, tt) == st::StatsStatus::kOk,
          "ttest_1sample ok");
    check(tt.h == 0 && keel::abs(tt.tstat) < 1e-12 && tt.p > 0.99, "ttest_1sample mean=4.5");

    st::TTest2Result tt2{};
    check(st::ttest_2sample(x, y, 0.05, st::TestTail::kBoth, true, tt2) == st::StatsStatus::kOk,
          "ttest_2sample eq ok");
    check(tt2.h == 1 && tt2.p < 1e-4, "ttest_2sample rejects");

    // 2. ztest_1sample, vartest_1sample, vartest_2sample
    st::ZTestResult zt{};
    check(st::ztest_1sample(x, 4.5, 2.0, 0.05, st::TestTail::kBoth, zt) == st::StatsStatus::kOk,
          "ztest_1sample ok");
    check(zt.h == 0 && keel::abs(zt.zval) < 1e-12, "ztest_1sample zero z");

    st::VarTestResult vt{};
    check(st::vartest_1sample(x, 6.0, 0.05, st::TestTail::kBoth, vt) == st::StatsStatus::kOk,
          "vartest_1sample ok");
    check(vt.h == 0 && vt.df == 7.0, "vartest_1sample df");

    st::VarTest2Result vt2{};
    check(st::vartest_2sample(x, y, 0.05, st::TestTail::kBoth, vt2) == st::StatsStatus::kOk,
          "vartest_2sample ok");
    check(vt2.h == 0 && keel::abs(vt2.fstat - 1.0) < 1e-12, "vartest_2sample equal var");

    // 3. kstest_1sample & kstest_2sample
    const f64 stdnorm[10] = {-1.5, -1.0, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 1.0, 1.5};
    st::KsTestResult ks{};
    check(st::kstest_1sample(stdnorm, {}, {}, 0.05, st::TestTail::kBoth, ks) ==
              st::StatsStatus::kOk,
          "kstest_1sample ok");
    check(ks.h == 0 && ks.p > 0.05, "kstest_1sample accepts std normal");

    st::KsTestResult ks2{};
    check(st::kstest_2sample(x, y, 0.05, st::TestTail::kBoth, ks2) == st::StatsStatus::kOk,
          "kstest_2sample ok");
    check(ks2.h == 1 && ks2.ksstat == 1.0, "kstest_2sample disjoint");

    // 4. jbtest
    st::JbTestResult jb{};
    check(st::jbtest(stdnorm, 0.05, keel::Limits<f64>::quiet_nan(), jb) ==
              st::StatsStatus::kOk,
          "jbtest ok");
    check(jb.h == 0 && jb.jbstat >= 0.0, "jbtest accepts normal");

    // 5. signtest, fishertest, chi2gof_freq
    st::SignTestResult sg{};
    check(st::signtest(x, {}, 4.5, 0.05, st::TestTail::kBoth, sg) == st::StatsStatus::kOk,
          "signtest ok");
    check(sg.h == 0 && sg.p > 0.99, "signtest median=4.5");

    st::FisherTestResult ft{};
    check(st::fishertest(10.0, 1.0, 1.0, 10.0, 0.05, st::TestTail::kBoth, ft) ==
              st::StatsStatus::kOk,
          "fishertest ok");
    check(ft.h == 1 && ft.p < 0.005 && ft.odds_ratio > 50.0, "fishertest strong association");

    const f64 obs[4] = {25.0, 25.0, 25.0, 25.0};
    const f64 exp_f[4] = {25.0, 25.0, 25.0, 25.0};
    st::Chi2GofResult cg{};
    check(st::chi2gof_freq(obs, exp_f, 0, 0.05, cg) == st::StatsStatus::kOk, "chi2gof ok");
    check(cg.h == 0 && cg.chi2stat == 0.0 && cg.df == 3.0, "chi2gof exact fit");

    // 6. vartestn
    const f64 vx[8] = {1.0, 2.0, 3.0, 4.0, 11.0, 12.0, 13.0, 14.0};
    const f64 vg[8] = {1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0};
    st::VarTestNResult vtn{};
    check(st::vartestn(vx, vg, st::VarTestNMethod::kBartlett, vtn) == st::StatsStatus::kOk,
          "vartestn bartlett ok");
    check(vtn.p > 0.99, "vartestn equal variance");

    // 7. runstest, ranksum, signrank, ansaribradley, lillietest
    const f64 runs_x[10] = {1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0};
    st::RunsTestResult rt{};
    check(st::runstest(runs_x, 5.0, 0.05, st::TestTail::kBoth, true, rt) == st::StatsStatus::kOk,
          "runstest ok");
    check(rt.nruns == 2 && rt.h == 1, "runstest 2 runs rejects");

    st::RankTestResult rs{};
    check(st::ranksum(x, y, 0.05, st::TestTail::kBoth, 0, rs) == st::StatsStatus::kOk,
          "ranksum ok");
    check(rs.h == 1 && rs.p < 0.005, "ranksum disjoint");

    st::RankTestResult sr{};
    check(st::signrank(x, {}, 4.5, 0.05, st::TestTail::kBoth, 0, sr) == st::StatsStatus::kOk,
          "signrank ok");
    check(sr.h == 0 && sr.p > 0.9, "signrank symmetric");

    st::RankTestResult ab{};
    check(st::ansaribradley(x, x, 0.05, st::TestTail::kBoth, ab) == st::StatsStatus::kOk,
          "ansaribradley ok");
    check(ab.h == 0, "ansaribradley equal scale");

    st::LillieTestResult lt{};
    check(st::lillietest(stdnorm, 0.05, lt) == st::StatsStatus::kOk, "lillietest ok");
    check(lt.h == 0 && lt.p > 0.05, "lillietest accepts normal");

    // 8. Distribution MLE fitters & CIs
    const f64 pos_data[12] = {0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6, 3.0, 3.5, 4.1, 4.8, 5.9};
    st::Fit2Result f2{};
    check(st::gamfit(pos_data, {}, {}, 0.05, true, f2) == st::StatsStatus::kOk, "gamfit ok");
    check(f2.p1 > 0.0 && f2.p2 > 0.0 && f2.ci1_lo < f2.p1 && f2.p1 < f2.ci1_hi, "gamfit ci");

    check(st::wblfit(pos_data, {}, {}, 0.05, true, f2) == st::StatsStatus::kOk, "wblfit ok");
    check(f2.p1 > 0.0 && f2.p2 > 0.0 && f2.ci1_lo < f2.p1 && f2.p1 < f2.ci1_hi, "wblfit ci");

    const f64 unit_data[10] = {0.12, 0.21, 0.30, 0.38, 0.45, 0.52, 0.61, 0.69, 0.78, 0.88};
    check(st::betafit(unit_data, 0.05, true, f2) == st::StatsStatus::kOk, "betafit ok");
    check(f2.p1 > 0.0 && f2.p2 > 0.0 && f2.ci1_lo < f2.p1 && f2.p1 < f2.ci1_hi, "betafit ci");

    const f64 nb_data[10] = {0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 12.0};
    check(st::nbinfit(nb_data, 0.05, true, f2) == st::StatsStatus::kOk, "nbinfit ok");
    check(f2.p1 > 0.0 && f2.p2 > 0.0 && f2.p2 < 1.0, "nbinfit valid params");

    const f64 ev_data[10] = {-2.1, -1.2, -0.5, 0.1, 0.6, 1.0, 1.4, 1.9, 2.3, 2.8};
    check(st::evfit(ev_data, {}, {}, 0.05, true, f2) == st::StatsStatus::kOk, "evfit ok");
    check(f2.p2 > 0.0 && f2.ci1_lo < f2.p1 && f2.p1 < f2.ci1_hi, "evfit ci");

    const f64 gp_data[12] = {0.1, 0.25, 0.4, 0.7, 1.1, 1.6, 2.3, 3.5, 5.2, 8.1, 13.5, 22.0};
    check(st::gpfit(gp_data, 0.05, true, f2) == st::StatsStatus::kOk, "gpfit ok");
    check(f2.p2 > 0.0 && f2.ci2_lo < f2.p2 && f2.p2 < f2.ci2_hi, "gpfit ci");

    st::Fit3Result f3{};
    check(st::gevfit(ev_data, 0.05, true, f3) == st::StatsStatus::kOk, "gevfit ok");
    check(f3.p2 > 0.0 && f3.ci2_lo < f3.p2 && f3.p2 < f3.ci2_hi, "gevfit ci");
}

void test_extra_distributions_and_fits() {
    namespace st = keel::stats;
    using namespace keel;
    // 1. Extra continuous & discrete distributions
    check_near(st::gp_cdf(st::gp_inv(0.7, 0.25, 2.0, 1.0), 0.25, 2.0, 1.0), 0.7, 1e-12, "gp inv/cdf");
    check(st::gp_pdf(2.5, 0.0, 2.0, 1.0) > 0.0 && st::gp_stat(0.2, 2.0, 1.0).mean > 1.0, "gp pdf/stat");

    check_near(st::ev_cdf(st::ev_inv(0.6, 1.0, 2.0), 1.0, 2.0), 0.6, 1e-12, "ev inv/cdf");
    check(st::ev_pdf(1.0, 1.0, 2.0) > 0.0 && st::ev_stat(1.0, 2.0).var > 0.0, "ev pdf/stat");

    check_near(st::rice_cdf(st::rice_inv(0.5, 2.0, 1.0), 2.0, 1.0), 0.5, 1e-9, "rice inv/cdf");
    check(st::rice_pdf(2.0, 2.0, 1.0) > 0.0 && st::rice_stat(2.0, 1.0).mean > 2.0, "rice pdf/stat");

    check_near(st::ncx2_cdf(st::ncx2_inv(0.75, 4.0, 3.0), 4.0, 3.0), 0.75, 1e-9, "ncx2 inv/cdf");
    check_near(st::ncx2_stat(4.0, 3.0).mean, 7.0, 1e-12, "ncx2 mean");

    check_near(st::naka_cdf(st::naka_inv(0.5, 2.0, 3.0), 2.0, 3.0), 0.5, 1e-10, "naka inv/cdf");
    check(st::naka_pdf(1.5, 2.0, 3.0) > 0.0 && st::naka_stat(2.0, 3.0).mean > 0.0, "naka pdf/stat");

    check_near(st::geo_pdf(2.0, 0.3), 0.49 * 0.3, 1e-12, "geo pdf");
    check(st::geo_inv(st::geo_cdf(3.0, 0.3), 0.3) == 3.0, "geo inv/cdf");
    check_near(st::geo_stat(0.25).mean, 3.0, 1e-12, "geo mean");

    check_near(st::unid_pdf(3.0, 6.0), 1.0 / 6.0, 1e-12, "unid pdf");
    check_near(st::unid_cdf(3.0, 6.0), 0.5, 1e-12, "unid cdf");
    check(st::unid_inv(0.5, 6.0) == 3.0, "unid inv");

    check(st::nbin_pdf(2.0, 3.0, 0.4) > 0.0, "nbin pdf");
    check(st::nbin_inv(st::nbin_cdf(4.0, 3.0, 0.4), 3.0, 0.4) == 4.0, "nbin inv/cdf");

    check(st::hyge_pdf(2.0, 20.0, 7.0, 5.0) > 0.0, "hyge pdf");
    check(st::hyge_inv(st::hyge_cdf(2.0, 20.0, 7.0, 5.0), 20.0, 7.0, 5.0) == 2.0, "hyge inv/cdf");

    // 2. Fitters & likelihoods
    const f64 xs[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    st::Fit2Result f2{};
    check(st::normfit(xs, {}, {}, 0.05, f2) == st::StatsStatus::kOk, "normfit ok");
    check_near(f2.p1, 3.5, 1e-12, "normfit mu");
    check(f2.ci1_lo < f2.p1 && f2.p1 < f2.ci1_hi, "normfit mu ci");

    const u8 cens[6] = {0, 0, 0, 0, 1, 1};
    check(st::normfit(xs, cens, {}, 0.05, f2) == st::StatsStatus::kOk, "normfit cens ok");
    check(f2.p1 > 3.5, "normfit right-censored shifts mu up");

    st::Fit1Result f1{};
    check(st::expfit(xs, {}, {}, 0.05, f1) == st::StatsStatus::kOk, "expfit ok");
    check_near(f1.p1, 3.5, 1e-12, "expfit mu");
    check(st::poissfit(xs, 0.05, f1) == st::StatsStatus::kOk && f1.ci_lo < f1.p1, "poissfit ok");
    check(st::unifit(xs, 0.05, f2) == st::StatsStatus::kOk && f2.p1 == 1.0 && f2.p2 == 6.0, "unifit ok");
    check(st::binofit_scalar(5.0, 10.0, 0.05, f1) == st::StatsStatus::kOk && f1.ci_lo < 0.5 && f1.ci_hi > 0.5,
          "binofit ok");
    check(st::raylfit(xs, 0.05, f1) == st::StatsStatus::kOk && f1.p1 > 0.0, "raylfit ok");

    f64 avar1 = 0.0;
    f64 avar2[4] = {};
    check(st::normlike(3.5, 1.5, xs) > 0.0, "normlike");
    check(st::explike(3.5, xs, {}, {}, &avar1) > 0.0 && avar1 > 0.0, "explike + avar");
    check(st::lognlike(1.0, 0.5, xs, {}, {}, avar2) > 0.0 && avar2[0] > 0.0, "lognlike + avar");
    check(st::gamlike(2.0, 1.5, xs) > 0.0, "gamlike");
    const f64 udata[4] = {0.2, 0.4, 0.6, 0.8};
    check(keel::is_finite(st::betalike(2.0, 2.0, udata)), "betalike");
    check(st::wbllike(3.0, 2.0, xs) > 0.0, "wbllike");
    check(keel::is_finite(st::evlike(3.0, 1.5, xs)), "evlike");
    check(keel::is_finite(st::gevlike(0.1, 1.5, 2.0, xs)), "gevlike");
    check(keel::is_finite(st::gplike(0.2, 1.5, xs)), "gplike");
}

void test_mvdist_robust_canon_multcompare() {
    namespace st = keel::stats;
    using namespace keel;
    // 1. cholcov & corrcov
    const f64 cov[4] = {4.0, 1.2, 1.2, 9.0};
    st::CholCovResult cc;
    check(st::cholcov(cov, 2, cc) == st::StatsStatus::kOk && cc.non_pd_count == 0 && cc.rows == 2,
          "cholcov PD");
    const f64 psd[4] = {1.0, 1.0, 1.0, 1.0};
    check(st::cholcov(psd, 2, cc) == st::StatsStatus::kOk && cc.non_pd_count == 0 && cc.rows == 1,
          "cholcov PSD rank 1");

    f64 rmat[4] = {};
    f64 sig[2] = {};
    check(st::corrcov(cov, 2, rmat, sig) == st::StatsStatus::kOk, "corrcov ok");
    check_near(sig[0], 2.0, 1e-12, "corrcov sig0");
    check_near(rmat[1], 0.2, 1e-12, "corrcov rho");

    // 2. genz_sov_mvncdf & mvtcdf_mc
    const f64 eye3[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const f64 neg_inf[3] = {-keel::Limits<f64>::infinity(), -keel::Limits<f64>::infinity(),
                            -keel::Limits<f64>::infinity()};
    const f64 zeros3[3] = {0.0, 0.0, 0.0};
    const f64 p3 = st::genz_sov_mvncdf(eye3, 3, neg_inf, zeros3, 4000, 1e-4);
    check_near(p3, 0.125, 1e-3, "genz_sov_mvncdf orthant");

    const f64 eye2[4] = {1, 0, 0, 1};
    const f64 lo2[2] = {-100.0, -100.0};
    const f64 hi2[2] = {0.0, 0.0};
    const f64 pt2 = st::mvtcdf_mc(eye2, 2, 10.0, lo2, hi2, 10000);
    check_near(pt2, 0.25, 0.02, "mvtcdf_mc orthant");

    // 3. robustfit & robustcov
    const f64 xrf[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8}; // 8 x 2 col-major
    const f64 yrf[8] = {3, 5, 7, 9, 11, 13, 15, 100.0};                  // outlier at end
    f64 beta[2] = {};
    f64 s_rob = 0.0;
    check(st::robustfit(xrf, 8, 2, yrf, st::RobustWeight::kBisquare,
                        keel::Limits<f64>::quiet_nan(), beta, s_rob) == st::StatsStatus::kOk,
          "robustfit ok");
    check_near(beta[0], 1.0, 0.05, "robustfit intercept resists outlier");
    check_near(beta[1], 2.0, 0.05, "robustfit slope resists outlier");

    const f64 xrc[20] = {0, 1, -1, 0.5, -0.5, 0.2, -0.2, 0.1, -0.1, 50.0,
                         0, 0.5, -0.5, 0.2, -0.2, 0.1, -0.1, 0.3, -0.3, -50.0};
    f64 rc_sig[4] = {};
    f64 rc_mu[2] = {};
    check(st::robustcov(xrc, 10, 2, rc_sig, rc_mu) == st::StatsStatus::kOk, "robustcov ok");
    check(keel::abs(rc_mu[0]) < 1.0 && keel::abs(rc_mu[1]) < 1.0, "robustcov ignores outlier");

    // 4. partialcorri & canoncorr
    const f64 xpc[12] = {1, 2, 3, 4, 5, 6, 2, 1, 4, 3, 6, 5}; // 6 x 2
    const f64 ypc[6] = {3, 4, 7, 8, 11, 12};                   // 6 x 1
    f64 r_pc[2] = {};
    check(st::partialcorri(ypc, 6, 1, xpc, 2, {}, 0, r_pc) == st::StatsStatus::kOk, "partialcorri ok");
    check(r_pc[0] > 0.9, "partialcorri strong positive");

    const f64 xcc[16] = {1, 2, 3, 4, 5, 6, 7, 8, 2, 1, 5, 3, 7, 4, 8, 6}; // 8 x 2
    const f64 ycc[16] = {2, 4, 6, 8, 10, 12, 14, 16, 1, 3, 2, 5, 4, 7, 6, 9};
    f64 ca[4] = {}, cb[4] = {}, cr[2] = {};
    check(st::canoncorr(xcc, 8, 2, ycc, 2, ca, cb, cr) == st::StatsStatus::kOk, "canoncorr ok");
    check_near(cr[0], 1.0, 1e-10, "canoncorr first r=1");

    // 5. studentized_range & multcompare
    const f64 q95 = st::studentized_range_inv(0.95, 3, 12.0);
    check_near(st::studentized_range_cdf(q95, 3, 12.0), 0.95, 1e-9, "studentized_range roundtrip");
    const f64 means[3] = {1.0, 2.0, 10.0};
    const f64 cnts[3] = {5.0, 5.0, 5.0};
    f64 mc[18] = {};
    check(st::multcompare(means, cnts, 0.5, 12.0, 0.05, st::McCorrection::kTukeyKramer, mc) ==
              st::StatsStatus::kOk,
          "multcompare ok");
    check(mc[5 * 3 + 1] < 1e-4, "multcompare detects group 1 vs 3 difference");
}

void test_ranks_outliers_smoothing_qmc_kmeans() {
    namespace st = keel::stats;
    using namespace keel;
    // 1. tiedrank
    const f64 tr_in[6] = {10, 20, 30, 20, 10, 40};
    f64 tr_out[6] = {};
    f64 tieadj = 0.0;
    check(st::tiedrank(tr_in, tr_out, tieadj) == st::StatsStatus::kOk, "tiedrank ok");
    check_near(tr_out[0], 1.5, 1e-12, "tiedrank[0]");
    check_near(tr_out[1], 3.5, 1e-12, "tiedrank[1]");
    check_near(tieadj, 6.0, 1e-12, "tiedrank tieadj");

    // 2. correlation p-values
    check_near(st::corr_pearson_pval(st::CorrTail::kBoth, 0.0, 10), 1.0, 1e-12, "pearson p(0)=1");
    f64 klo = 0.0, khi = 0.0;
    check(st::corr_kendall_exact_lu(1.0, 5, klo, khi) == st::StatsStatus::kOk, "kendall exact ok");
    check_near(klo, 1.0 / 120.0, 1e-12, "kendall exact tau=1 n=5");
    const f64 sp_p = st::corr_spearman_no_ties_pval(st::CorrTail::kBoth, 0.9, 15);
    check(sp_p > 0.0 && sp_p < 0.002, "spearman AS89");

    // 3. grubbs & gesd outliers
    const f64 ox[10] = {1, 2, 1.5, 1.2, 1.8, 1.4, 1.6, 1.3, 1.7, 50.0};
    u8 omask[10] = {};
    st::OutlierBounds ob{};
    check(st::grubbs_outliers(ox, 0.05, omask, ob) == st::StatsStatus::kOk && omask[9] == 1 && omask[0] == 0,
          "grubbs detects outlier");
    check(st::gesd_outliers(ox, 0.05, -1, omask, ob) == st::StatsStatus::kOk && omask[9] == 1 && omask[0] == 0,
          "gesd detects outlier");

    // 4. smoothing & hampel
    const f64 sig[7] = {1, 2, 3, 100, 5, 6, 7};
    f64 dst[7] = {};
    u8 hmask[7] = {};
    check(st::hampel_1d(sig, 2, 3.0, dst, hmask, {}, {}) == st::StatsStatus::kOk && hmask[3] == 1,
          "hampel_1d detects spike");
    keel::Array<f64> sg_b;
    check(st::smooth_sgolay_matrix(5, 2, sg_b) == st::StatsStatus::kOk, "sgolay matrix ok");
    const f64 lin[7] = {1, 2, 3, 4, 5, 6, 7};
    st::smooth_sgolay_strided(lin, 7, 1, sg_b.as_ref(), 5, dst);
    check_near(dst[3], 4.0, 1e-12, "sgolay preserves linear");
    st::smooth_local_regress_strided(lin, 7, 1, 5, 1, dst);
    check_near(dst[3], 4.0, 1e-12, "lowess preserves linear");
    st::smooth_gaussian_strided(lin, 7, 1, 2, 2, 1.0, false, dst);
    check_near(dst[3], 4.0, 1e-12, "gaussian symmetric preserves linear");

    // 5. halton_net
    f64 hnet[8] = {};
    check(st::halton_net(4, 2, 1, 0, false, hnet) == st::StatsStatus::kOk, "halton_net ok");
    check_near(hnet[0], 0.5, 1e-12, "halton[0,0]");
    check_near(hnet[4], 1.0 / 3.0, 1e-12, "halton[0,1]");

    // 6. kmeans_lloyd
    const f64 kmx[12] = {0, 0, 0.1, 0.1, -0.1, 0.1, 10, 10, 10.1, 9.9, 9.9, 10.1};
    f64 kmc[4] = {0.0, 0.0, 10.0, 10.0};
    i32 kidx[6] = {};
    f64 ksumd[2] = {};
    f64 ktot = 0.0;
    check(st::kmeans_lloyd(kmx, 6, 2, 2, kmc, 50, kidx, ksumd, ktot) == st::StatsStatus::kOk,
          "kmeans_lloyd ok");
    check(kidx[0] == 0 && kidx[3] == 1 && ktot < 0.2, "kmeans_lloyd clusters");
}

} // namespace

int main() {
    test_continuous_distributions();
    test_discrete_and_descriptive();
    test_distance_and_linkage();
    test_lasso_and_anova();
    test_regress_pca_lda_cluster_tests();
    test_copulas_and_mvdist();
    test_hypothesis_and_fits();
    test_extra_distributions_and_fits();
    test_mvdist_robust_canon_multcompare();
    test_ranks_outliers_smoothing_qmc_kmeans();
    if (g_failures == 0) {
        keel::println("ALL keel::stats TESTS PASSED");
        return 0;
    }
    keel::eprintln("{} FAILURE(S)", g_failures);
    return 1;
}
