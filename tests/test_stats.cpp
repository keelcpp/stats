#include <keel/format.h>
#include <keel/math.h>
#include <keel/stats.h>
#include <keel/string.h>

namespace {

keel::i32 g_failures = 0;

void check(bool cond, const char* msg) {
    if (!cond) {
        keel::eprintln("FAIL: %s", msg);
        ++g_failures;
    }
}

void check_near(keel::f64 got, keel::f64 expected, keel::f64 tol, const char* msg) {
    if (keel::abs(got - expected) > tol || keel::is_nan(got)) {
        keel::eprintln("FAIL: %s (got %.15g, expected %.15g, tol %.15g)", msg, got, expected, tol);
        ++g_failures;
    }
}

void print_vec(const char* key, keel::ArrayRef<keel::f64> vals) {
    keel::print("%s=", key);
    for (keel::usize i = 0; i < vals.size(); ++i) {
        if (i > 0) keel::print(",");
        keel::print("%.17g", vals[i]);
    }
    keel::println("");
}

void print_val(const char* key, keel::f64 v) {
    keel::println("%s=%.17g", key, v);
}

void test_continuous_distributions() {
    // Normal
    check_near(keel::stats::norm_cdf(0.0), 0.5, 1e-15, "norm_cdf(0)");
    check_near(keel::stats::norm_inv(keel::stats::norm_cdf(1.96, 2.0, 3.0), 2.0, 3.0), 1.96, 1e-12,
               "norm roundtrip");
    auto ns = keel::stats::norm_stat(2.0, 3.0);
    check_near(ns.mean, 2.0, 1e-15, "norm mean");
    check_near(ns.var, 9.0, 1e-15, "norm var");

    // Student's t
    check_near(keel::stats::t_cdf(0.0, 5.0), 0.5, 1e-15, "t_cdf(0)");
    check_near(keel::stats::t_inv(keel::stats::t_cdf(1.5, 7.0), 7.0), 1.5, 1e-10, "t roundtrip");
    check_near(keel::stats::t_inv(0.975, keel::Limits<keel::f64>::infinity()), 1.959963984540054,
               1e-12, "t_inv inf");

    // Chi2, F, Gamma, Beta
    check_near(keel::stats::chi2_inv(keel::stats::chi2_cdf(6.5, 4.0), 4.0), 6.5, 1e-10,
               "chi2 roundtrip");
    check_near(keel::stats::f_inv(keel::stats::f_cdf(2.5, 5.0, 10.0), 5.0, 10.0), 2.5, 1e-10,
               "f roundtrip");
    check_near(keel::stats::gam_inv(keel::stats::gam_cdf(3.2, 2.5, 1.5), 2.5, 1.5), 3.2, 1e-10,
               "gam roundtrip");
    check_near(keel::stats::beta_inv(keel::stats::beta_cdf(0.4, 2.0, 5.0), 2.0, 5.0), 0.4, 1e-10,
               "beta roundtrip");

    // Exp, Weibull, Lognormal, Uniform, Rayleigh, GEV
    check_near(keel::stats::exp_inv(keel::stats::exp_cdf(2.0, 3.0), 3.0), 2.0, 1e-13,
               "exp roundtrip");
    check_near(keel::stats::wbl_inv(keel::stats::wbl_cdf(1.5, 2.0, 3.0), 2.0, 3.0), 1.5, 1e-13,
               "wbl roundtrip");
    check_near(keel::stats::logn_inv(keel::stats::logn_cdf(2.5, 0.5, 0.8), 0.5, 0.8), 2.5, 1e-12,
               "logn roundtrip");
    check_near(keel::stats::unif_inv(keel::stats::unif_cdf(3.5, 1.0, 5.0), 1.0, 5.0), 3.5, 1e-14,
               "unif roundtrip");
    check_near(keel::stats::rayl_inv(keel::stats::rayl_cdf(2.0, 1.5), 1.5), 2.0, 1e-13,
               "rayl roundtrip");
    check_near(keel::stats::gev_inv(keel::stats::gev_cdf(1.2, 0.2, 1.5, 0.5), 0.2, 1.5, 0.5), 1.2,
               1e-12, "gev roundtrip");
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
    check(st::kmedoids_pam_step(pts, 6, 2, init_m, 50, st::DistanceMetric::kEuclidean, 2.0, km_idx,
                                km_med, km_sumd, km_tot, km_iters) == st::StatsStatus::kOk,
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
    check(dw.dw > 0.0 && dw.dw < 4.0 && dw.p_both >= 0.0 && dw.p_both <= 1.0,
          "dwtest valid range");
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
    check_near(mn_out[0], 0.1875, 1e-12, "mnpdf exact value");
}

void test_hypothesis_and_fits() {
    namespace st = keel::stats;
    using keel::f64;

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
    check(st::kstest_1sample(stdnorm, {}, {}, 0.05, st::TestTail::kBoth, ks) == st::StatsStatus::kOk,
          "kstest_1sample ok");
    check(ks.h == 0 && ks.p > 0.05, "kstest_1sample accepts std normal");

    st::KsTestResult ks2{};
    check(st::kstest_2sample(x, y, 0.05, st::TestTail::kBoth, ks2) == st::StatsStatus::kOk,
          "kstest_2sample ok");
    check(ks2.h == 1 && ks2.ksstat == 1.0, "kstest_2sample disjoint");

    // 4. jbtest
    st::JbTestResult jb{};
    check(st::jbtest(stdnorm, 0.05, keel::Limits<f64>::quiet_nan(), jb) == st::StatsStatus::kOk,
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
    check_near(st::gp_cdf(st::gp_inv(0.7, 0.25, 2.0, 1.0), 0.25, 2.0, 1.0), 0.7, 1e-12,
               "gp inv/cdf");
    check(st::gp_pdf(2.5, 0.0, 2.0, 1.0) > 0.0 && st::gp_stat(0.2, 2.0, 1.0).mean > 1.0,
          "gp pdf/stat");

    check_near(st::ev_cdf(st::ev_inv(0.6, 1.0, 2.0), 1.0, 2.0), 0.6, 1e-12, "ev inv/cdf");
    check(st::ev_pdf(1.0, 1.0, 2.0) > 0.0 && st::ev_stat(1.0, 2.0).var > 0.0, "ev pdf/stat");

    check_near(st::rice_cdf(st::rice_inv(0.5, 2.0, 1.0), 2.0, 1.0), 0.5, 1e-9, "rice inv/cdf");
    check(st::rice_pdf(2.0, 2.0, 1.0) > 0.0 && st::rice_stat(2.0, 1.0).mean > 2.0,
          "rice pdf/stat");

    check_near(st::ncx2_cdf(st::ncx2_inv(0.75, 4.0, 3.0), 4.0, 3.0), 0.75, 1e-9, "ncx2 inv/cdf");
    check_near(st::ncx2_stat(4.0, 3.0).mean, 7.0, 1e-12, "ncx2 mean");

    check_near(st::naka_cdf(st::naka_inv(0.5, 2.0, 3.0), 2.0, 3.0), 0.5, 1e-10, "naka inv/cdf");
    check(st::naka_pdf(1.5, 2.0, 3.0) > 0.0 && st::naka_stat(2.0, 3.0).mean > 0.0,
          "naka pdf/stat");

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
    check(st::unifit(xs, 0.05, f2) == st::StatsStatus::kOk && f2.p1 == 1.0 && f2.p2 == 6.0,
          "unifit ok");
    check(st::binofit_scalar(5.0, 10.0, 0.05, f1) == st::StatsStatus::kOk && f1.ci_lo < 0.5 &&
              f1.ci_hi > 0.5,
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
    const f64 ypc[6] = {3, 4, 7, 8, 11, 12};                  // 6 x 1
    f64 r_pc[2] = {};
    check(st::partialcorri(ypc, 6, 1, xpc, 2, {}, 0, r_pc) == st::StatsStatus::kOk,
          "partialcorri ok");
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
    check(st::grubbs_outliers(ox, 0.05, omask, ob) == st::StatsStatus::kOk && omask[9] == 1 &&
              omask[0] == 0,
          "grubbs detects outlier");
    check(st::gesd_outliers(ox, 0.05, -1, omask, ob) == st::StatsStatus::kOk && omask[9] == 1 &&
              omask[0] == 0,
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

    // 7. pdist cosine row-normalization invariance (collinear rows get uniform tie distance)
    const f64 cos_x[12] = {100.0, 90.0, 10.0, 10.0, 90.0, 50.0,
                           100.0, 90.0, 10.0, 20.0, 70.0, 50.0};
    f64 cos_d[15] = {};
    check(st::pdist(cos_x, 6, 2, st::DistanceMetric::kCosine, 2.0, {}, cos_d) ==
              st::StatsStatus::kOk,
          "pdist cosine ok");
    check(cos_d[0] == cos_d[1] && cos_d[0] == cos_d[4] && cos_d[0] == cos_d[5] &&
              cos_d[0] == cos_d[8] && cos_d[0] == cos_d[11],
          "pdist cosine collinear pairs identical");
    f64 cos_z[15] = {};
    check(st::linkage(cos_d, 6, st::LinkageMethod::kWeighted, cos_z) == st::StatsStatus::kOk,
          "linkage weighted on cosine ok");
    check(cos_z[0] == 3.0 && cos_z[5] == 6.0, "linkage weighted first merge (3, 6)");
}

void test_hardening_regressions() {
    namespace st = keel::stats;
    using namespace keel;
    const f64 inf = Limits<f64>::infinity();
    const f64 nan = Limits<f64>::quiet_nan();

    // 1. Infinite argument PDF/CDF safety (no Inf*0 or Inf-Inf NaN)
    check(st::t_pdf(inf, 5.0) == 0.0, "t_pdf(+Inf) == 0");
    check_near(st::t_pdf(1.0, 1e12), st::norm_pdf(1.0, 0.0, 1.0), 1e-11, "t_pdf huge nu");
    check(st::chi2_pdf(inf, 4.0) == 0.0, "chi2_pdf(+Inf) == 0");
    check(is_inf(st::chi2_pdf(0.0, 0.0)), "chi2_pdf(0,0) == Inf");
    check(st::chi2_cdf(0.0, 0.0) == 1.0, "chi2_cdf(0,0) == 1");
    check(st::f_pdf(inf, 5.0, 10.0) == 0.0, "f_pdf(+Inf) == 0");
    check(st::gam_pdf(inf, 3.0, 2.0) == 0.0, "gam_pdf(+Inf) == 0");
    check(st::wbl_pdf(inf, 2.0, 3.0) == 0.0, "wbl_pdf(+Inf) == 0");
    check(st::wbl_pdf(100.0, 1.0, 50.0) == 0.0, "wbl_pdf large r,b no NaN");
    check(st::rayl_pdf(inf, 2.0) == 0.0, "rayl_pdf(+Inf) == 0");
    check(st::gev_pdf(-inf, 0.0, 1.0, 0.0) == 0.0, "gev_pdf(-Inf, k=0) == 0");
    check(st::gev_pdf(inf, 0.2, 1.0, 0.0) == 0.0, "gev_pdf(+Inf) == 0");
    check(st::ev_pdf(inf, 0.0, 1.0) == 0.0, "ev_pdf(+Inf) == 0");
    check(st::poiss_pdf(inf, 4.0) == 0.0, "poiss_pdf(+Inf) == 0");
    check(is_nan(st::bino_pdf(1.0, inf, 0.5)), "bino_pdf(n=Inf) == NaN");
    check(st::nbin_pdf(inf, 3.0, 0.4) == 0.0, "nbin_pdf(+Inf) == 0");
    check(is_nan(st::unid_pdf(1.0, inf)), "unid_pdf(n=Inf) == NaN");

    // 2. Discrete inverse CDFs at large parameters (no underflow / hang)
    check_near(st::bino_inv(0.5, 5000.0, 0.5), 2500.0, 1.0, "bino_inv large n=5000");
    check(st::poiss_inv(1.0 - 1e-15, 5.0) >= 20.0 && st::poiss_inv(1.0 - 1e-15, 5.0) < 100.0,
          "poiss_inv extreme p terminates");
    check_near(st::poiss_inv(0.5, 1000.0), 1000.0, 1.0, "poiss_inv large lambda");
    const f64 nb_q = st::nbin_inv(0.5, 500.0, 0.2);
    check(is_finite(nb_q) && nb_q > 1900.0 && nb_q < 2100.0, "nbin_inv large r=500");
    check_near(st::hyge_cdf(500.0, 2000.0, 1000.0, 1000.0), 0.525, 0.05, "hyge_cdf large m=2000");
    check_near(st::hyge_inv(0.5, 2000.0, 1000.0, 1000.0), 500.0, 1.0, "hyge_inv large m=2000");

    // 3. Overflow-free Rice, Noncentral Chi2, Nakagami
    const f64 rp_big = st::rice_pdf(100.0, 100.0, 1.0);
    check(is_finite(rp_big) && rp_big > 0.3 && rp_big < 0.5, "rice_pdf large s=100 no NaN");
    const auto rs_big = st::rice_stat(1e6, 1.0);
    check_near(rs_big.mean, 1e6, 1e-3, "rice_stat huge s mean");
    check_near(rs_big.var, 1.0, 1e-6, "rice_stat huge s var");
    check(is_inf(st::ncx2_pdf(0.0, 1.0, 2.0)), "ncx2_pdf(0, 1, 2) == Inf");
    check_near(st::ncx2_pdf(0.0, 2.0, 2.0), 0.5 * keel::exp(-1.0), 1e-14, "ncx2_pdf(0, 2, 2)");
    const f64 nc_big_pdf = st::ncx2_pdf(2004.0, 4.0, 2000.0);
    check(is_finite(nc_big_pdf) && nc_big_pdf > 0.0, "ncx2_pdf large lambda=2000");
    const f64 nc_big_cdf = st::ncx2_cdf(2004.0, 4.0, 2000.0);
    check(nc_big_cdf > 0.45 && nc_big_cdf < 0.55, "ncx2_cdf large lambda=2000");
    check(is_finite(st::naka_pdf(1.0, 200.0, 1.0)) && st::naka_pdf(1.0, 200.0, 1.0) > 0.0,
          "naka_pdf large mu=200");
    const auto nk_big = st::naka_stat(200.0, 4.0);
    check(is_finite(nk_big.mean) && is_finite(nk_big.var) && nk_big.var > 0.0,
          "naka_stat large mu=200");

    // 4. Descriptive edge cases
    const f64 one_nan[1] = {nan};
    check(is_nan(st::var(one_nan, true)), "var([NaN], pop) is NaN");
    const f64 big_x[4] = {1e150, 2e150, 3e150, 5e150};
    check(is_finite(st::skewness(big_x, false)), "skewness large scale finite");
    check(is_finite(st::kurtosis(big_x, false)), "kurtosis large scale finite");

    // 5. Distance edge cases (non-binary Jaccard, Minkowski p=Inf, Cosine scaling)
    const f64 jac_x[6] = {1.0, 1.0, 2.0, 3.0, 0.0, 0.0}; // 2x3 col-major: [1,2,0], [1,3,0]
    f64 jac_d[1]{};
    check(st::pdist(jac_x, 2, 3, st::DistanceMetric::kJaccard, 2.0, {}, jac_d) ==
              st::StatsStatus::kOk,
          "pdist jaccard non-binary ok");
    check_near(jac_d[0], 0.5, 1e-14, "pdist jaccard non-binary == 0.5");
    const f64 zero_x[4] = {0.0, 0.0, 0.0, 0.0};
    check(st::pdist(zero_x, 2, 2, st::DistanceMetric::kJaccard, 2.0, {}, jac_d) ==
              st::StatsStatus::kOk &&
              is_nan(jac_d[0]),
          "pdist jaccard all-zero == NaN");
    const f64 m_x[4] = {0.0, 3.0, 0.0, 4.0};
    f64 m_d[1]{};
    check(st::pdist(m_x, 2, 2, st::DistanceMetric::kMinkowski, inf, {}, m_d) ==
              st::StatsStatus::kOk,
          "pdist minkowski p=Inf ok");
    check_near(m_d[0], 4.0, 1e-14, "pdist minkowski p=Inf == chebychev");
    check(st::pdist(m_x, 2, 2, st::DistanceMetric::kMinkowski, -1.0, {}, m_d) ==
              st::StatsStatus::kInvalidArgument,
          "pdist minkowski p<0 invalid");

    // 6. Corrupted linkage tree rejected safely
    const f64 bad_z[3] = {1.0, 99.0, 0.5};
    i32 lbl2[2]{};
    check(st::cluster_maxclust(bad_z, 1, 1, lbl2) == st::StatsStatus::kInvalidArgument,
          "cluster_maxclust rejects invalid node id");
    f64 bad_inc[4]{};
    check(st::inconsistent(bad_z, 1, 2, bad_inc) == st::StatsStatus::kInvalidArgument,
          "inconsistent rejects invalid node id");

    // 7. LASSO constant column + ANOVA empty groups / constant data
    const f64 x_const[8] = {5.0, 5.0, 5.0, 5.0, 1.0, 2.0, 3.0, 4.0};
    const f64 y_const[4] = {2.0, 4.0, 6.0, 8.0};
    const f64 lam1[1] = {0.01};
    f64 lb[2]{}, li[1]{};
    check(st::lasso(x_const, 4, 2, y_const, lam1, 1.0, lb, li) == st::StatsStatus::kOk &&
              is_finite(lb[0]) && lb[0] == 0.0 && is_finite(lb[1]),
          "lasso constant col no NaN");
    const f64 a_vals[2] = {1.0, 2.0};
    const usize a_offs[4] = {0, 1, 1, 2};
    st::Anova1Result a1{};
    check(st::anova1_grouped(a_vals, a_offs, a1) == st::StatsStatus::kOk && a1.df_within == 0.0,
          "anova1_grouped no usize underflow");
    const f64 const_m[4] = {3.0, 3.0, 3.0, 3.0};
    st::Anova2Result a2{};
    check(st::anova2(const_m, 2, 2, a2) == st::StatsStatus::kOk && is_nan(a2.f_cols) &&
              a2.ss_err == 0.0,
          "anova2 constant matrix NaN F");

    // 8. One-sided hypothesis test CIs & kstest_2sample with ties & vartestn O'Brien
    const f64 tx[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const f64 ty[6] = {2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    st::TTest2Result tt2_r{}, tt2_l{};
    check(st::ttest_2sample(tx, ty, 0.05, st::TestTail::kRight, true, tt2_r) ==
              st::StatsStatus::kOk &&
              is_inf(tt2_r.ci_hi) && tt2_r.ci_hi > 0.0 && is_finite(tt2_r.ci_lo),
          "ttest_2sample right tail CI");
    check(st::ttest_2sample(tx, ty, 0.05, st::TestTail::kLeft, true, tt2_l) ==
              st::StatsStatus::kOk &&
              is_inf(tt2_l.ci_lo) && tt2_l.ci_lo < 0.0 && is_finite(tt2_l.ci_hi),
          "ttest_2sample left tail CI");
    st::ZTestResult zt_r{};
    check(st::ztest_1sample(tx, 3.0, 1.5, 0.05, st::TestTail::kRight, zt_r) ==
              st::StatsStatus::kOk &&
              is_inf(zt_r.ci_hi) && is_finite(zt_r.ci_lo),
          "ztest right tail CI");
    st::VarTestResult vt_l{};
    check(st::vartest_1sample(tx, 3.0, 0.05, st::TestTail::kLeft, vt_l) == st::StatsStatus::kOk &&
              vt_l.ci_lo == 0.0 && is_finite(vt_l.ci_hi),
          "vartest left tail CI");
    const f64 ks_t1[3] = {1.0, 1.0, 2.0};
    const f64 ks_t2[4] = {1.0, 1.0, 1.0, 2.0};
    st::KsTestResult ks_ties{};
    check(st::kstest_2sample(ks_t1, ks_t2, 0.05, st::TestTail::kRight, ks_ties) ==
              st::StatsStatus::kOk,
          "kstest_2sample ties ok");
    check_near(ks_ties.ksstat, 0.0, 1e-14, "kstest_2sample ties d_plus == 0");
    check(st::kstest_2sample(ks_t1, ks_t2, 0.05, st::TestTail::kLeft, ks_ties) ==
              st::StatsStatus::kOk,
          "kstest_2sample ties left ok");
    check_near(ks_ties.ksstat, 1.0 / 12.0, 1e-14, "kstest_2sample ties d_minus == 1/12");

    const f64 ob_x[4] = {1.0, 2.0, 3.0, 4.0};
    const f64 ob_g[4] = {1.0, 1.0, 2.0, 2.0};
    st::VarTestNResult vtn_ob{};
    check(st::vartestn(ob_x, ob_g, st::VarTestNMethod::kOBrien, vtn_ob) == st::StatsStatus::kOk &&
              is_nan(vtn_ob.p),
          "vartestn O'Brien size-2 groups no div-by-0");

    // 9. Copulas & mvdist boundary/singularity regressions
    check_near(st::bivnorm_cdf(0.5, -0.2, 1.0), st::norm_cdf(-0.2), 1e-14, "bivnorm rho=1");
    check_near(st::bivnorm_cdf(0.5, 0.2, -1.0), st::norm_cdf(0.5) - st::norm_cdf(-0.2), 1e-14,
               "bivnorm rho=-1");
    check(is_nan(st::bivnorm_cdf(0.0, 0.0, 1.5)), "bivnorm invalid rho");
    const f64 uf[2] = {0.3, 0.7};
    f64 cf[1]{};
    check(st::copula_pdf_frank(uf, 1, 0.0, cf) == st::StatsStatus::kOk, "frank pdf alpha=0 ok");
    check_near(cf[0], 1.0, 1e-14, "frank pdf alpha=0 == 1");
    check(st::copula_cdf_frank(uf, 1, 0.0, cf) == st::StatsStatus::kOk, "frank cdf alpha=0 ok");
    check_near(cf[0], 0.21, 1e-14, "frank cdf alpha=0 == u*v");

    const f64 sing_c[4] = {1.0, 1.0, 1.0, 1.0};
    const f64 xv2[2] = {0.0, 0.0};
    const f64 df1[1] = {5.0};
    f64 mvt_out[1]{};
    check(st::mvtpdf_full(xv2, 1, 2, sing_c, df1, mvt_out) == st::StatsStatus::kSingularMatrix,
          "mvtpdf singular corr rejected");

    // 10. Distribution fits at large scale & likelihood validation
    const f64 wbl_big[6] = {1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0};
    st::Fit2Result wf{};
    check(st::wblfit(wbl_big, {}, {}, 0.05, false, wf) == st::StatsStatus::kOk &&
              is_finite(wf.p1) && is_finite(wf.p2) && wf.p1 > 999.0 && wf.p2 > 100.0,
          "wblfit large scale no overflow");
    const f64 neg_x[2] = {-1.0, 2.0};
    check(is_nan(st::explike(2.0, neg_x)), "explike negative x -> NaN");
    check(is_inf(st::gplike(0.2, -1.0, wbl_big)) && st::gplike(0.2, -1.0, wbl_big) < 0.0,
          "gplike negative sigma -> -Inf");
    check(is_nan(st::gplike(0.2, 0.0, wbl_big)), "gplike zero sigma -> NaN");
    const f64 gp_neg_ok[3] = {-1.0, 1.0, 2.0};
    check_near(st::gplike(0.5, 1.0, gp_neg_ok), 1.21639532432449, 1e-12,
               "gplike negative x within support");
    const f64 gev_viol[3] = {-100.0, -1.0, 0.0};
    check(is_nan(st::gevlike(0.5, 1.0, 0.0, gev_viol)), "gevlike support violation -> NaN");

    // 11. Spearman small-n tail orientation, Grubbs n<3, GESD DOF, smoothers OOB, halton_net
    const f64 sp_right = st::corr_spearman_no_ties_pval(st::CorrTail::kRight, 0.942857142857, 6);
    const f64 sp_left = st::corr_spearman_no_ties_pval(st::CorrTail::kLeft, 0.942857142857, 6);
    check(sp_right < 0.02 && sp_left > 0.98, "spearman n<10 right vs left tail orientation");

    const f64 g2[2] = {10.0, 20.0};
    u8 g2_mask[2] = {1, 1};
    st::OutlierBounds g2_b{};
    check(st::grubbs_outliers(g2, 0.05, g2_mask, g2_b) == st::StatsStatus::kOk &&
              g2_mask[0] == 0 && g2_mask[1] == 0,
          "grubbs n=2 flags no outliers");

    // Smoothers OOB & invalid deg guard (no ASan violation)
    f64 sm_dst[2] = {99.0, 99.0};
    st::smooth_local_regress_strided(g2, 2, 1, 5, 1, sm_dst);
    st::smooth_local_regress_strided(g2, 2, 1, 2, 5, sm_dst);
    check(sm_dst[0] == 99.0, "smooth_local_regress guards frame_len>n and deg>2");

    f64 h_bad[4]{};
    check(st::halton_net(2, 2, -1, 0, true, h_bad) == st::StatsStatus::kInvalidArgument,
          "halton_net rejects negative skip");

    // 12. Studentized range NaN df & canoncorr unit variance scaling
    check(is_nan(st::studentized_range_cdf(3.0, 3, nan)), "studentized_range_cdf NaN df -> NaN");
    const f64 xcc[16] = {1, 2, 3, 4, 5, 6, 7, 8, 2, 1, 5, 3, 7, 4, 8, 6};
    const f64 ycc[16] = {2, 4, 6, 8, 10, 12, 14, 16, 1, 3, 2, 5, 4, 7, 6, 9};
    f64 ca[4]{}, cb[4]{}, cr[2]{};
    check(st::canoncorr(xcc, 8, 2, ycc, 2, ca, cb, cr) == st::StatsStatus::kOk, "canoncorr ok");
    // Column 1 of X is 1..8 with sample var = 6.0 -> |ca[0]| == 1/sqrt(6)
    check_near(keel::abs(ca[0]), 1.0 / keel::sqrt(6.0), 1e-12, "canoncorr unit variance scaling");

    // 13. adtest dynamic critical value & dwtest approximate exact-moment parity
    const f64 norm10[10] = {-1.5, -1.0, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 1.0, 1.5};
    st::AdTestResult ad01{}, ad05{};
    check(st::adtest(norm10, 0.01, ad01) == st::StatsStatus::kOk &&
              st::adtest(norm10, 0.05, ad05) == st::StatsStatus::kOk && ad01.cv > ad05.cv + 0.2,
          "adtest cv varies with alpha");
    const f64 dwr[5] = {0.1, -0.1, 0.1, -0.1, 0.0};
    const f64 dwx[10] = {1, 1, 1, 1, 1, 1, 2, 3, 4, 5};
    st::DwTestResult dw_ap{};
    check(st::dwtest(dwr, dwx, 5, 2, false, dw_ap) == st::StatsStatus::kOk, "dwtest approx ok");
    check_near(dw_ap.p_both, 0.21510439330176645, 1e-12, "dwtest approx matches MATLAB");
}

void dump_reference() {
    namespace st = keel::stats;
    using namespace keel;

    // 1. Continuous distributions
    const f64 d_norm[5] = {st::norm_pdf(1.2, 0.5, 2.0), st::norm_cdf(1.2, 0.5, 2.0),
                           st::norm_inv(0.85, 0.5, 2.0), st::norm_stat(0.5, 2.0).mean,
                           st::norm_stat(0.5, 2.0).var};
    print_vec("DIST_NORM", d_norm);
    const f64 d_t[5] = {st::t_pdf(1.3, 7.0), st::t_cdf(1.3, 7.0), st::t_inv(0.9, 7.0),
                        st::t_stat(7.0).mean, st::t_stat(7.0).var};
    print_vec("DIST_T", d_t);
    const f64 d_chi2[5] = {st::chi2_pdf(4.2, 5.0), st::chi2_cdf(4.2, 5.0), st::chi2_inv(0.95, 5.0),
                           st::chi2_stat(5.0).mean, st::chi2_stat(5.0).var};
    print_vec("DIST_CHI2", d_chi2);
    const f64 d_f[5] = {st::f_pdf(2.1, 5.0, 12.0), st::f_cdf(2.1, 5.0, 12.0),
                        st::f_inv(0.95, 5.0, 12.0), st::f_stat(5.0, 12.0).mean,
                        st::f_stat(5.0, 12.0).var};
    print_vec("DIST_F", d_f);
    const f64 d_gam[5] = {st::gam_pdf(3.2, 2.5, 1.5), st::gam_cdf(3.2, 2.5, 1.5),
                          st::gam_inv(0.8, 2.5, 1.5), st::gam_stat(2.5, 1.5).mean,
                          st::gam_stat(2.5, 1.5).var};
    print_vec("DIST_GAM", d_gam);
    const f64 d_beta[5] = {st::beta_pdf(0.4, 2.0, 5.0), st::beta_cdf(0.4, 2.0, 5.0),
                           st::beta_inv(0.7, 2.0, 5.0), st::beta_stat(2.0, 5.0).mean,
                           st::beta_stat(2.0, 5.0).var};
    print_vec("DIST_BETA", d_beta);
    const f64 d_exp[5] = {st::exp_pdf(2.0, 3.0), st::exp_cdf(2.0, 3.0), st::exp_inv(0.6, 3.0),
                          st::exp_stat(3.0).mean, st::exp_stat(3.0).var};
    print_vec("DIST_EXP", d_exp);
    const f64 d_wbl[5] = {st::wbl_pdf(1.8, 2.0, 3.0), st::wbl_cdf(1.8, 2.0, 3.0),
                          st::wbl_inv(0.75, 2.0, 3.0), st::wbl_stat(2.0, 3.0).mean,
                          st::wbl_stat(2.0, 3.0).var};
    print_vec("DIST_WBL", d_wbl);
    const f64 d_logn[5] = {st::logn_pdf(2.5, 0.5, 0.8), st::logn_cdf(2.5, 0.5, 0.8),
                           st::logn_inv(0.6, 0.5, 0.8), st::logn_stat(0.5, 0.8).mean,
                           st::logn_stat(0.5, 0.8).var};
    print_vec("DIST_LOGN", d_logn);
    const f64 d_unif[5] = {st::unif_pdf(2.5, 1.0, 5.0), st::unif_cdf(2.5, 1.0, 5.0),
                           st::unif_inv(0.4, 1.0, 5.0), st::unif_stat(1.0, 5.0).mean,
                           st::unif_stat(1.0, 5.0).var};
    print_vec("DIST_UNIF", d_unif);
    const f64 d_rayl[5] = {st::rayl_pdf(2.0, 1.5), st::rayl_cdf(2.0, 1.5), st::rayl_inv(0.7, 1.5),
                           st::rayl_stat(1.5).mean, st::rayl_stat(1.5).var};
    print_vec("DIST_RAYL", d_rayl);
    const f64 d_gev[5] = {st::gev_pdf(1.2, 0.2, 1.5, 0.5), st::gev_cdf(1.2, 0.2, 1.5, 0.5),
                          st::gev_inv(0.8, 0.2, 1.5, 0.5), st::gev_stat(0.2, 1.5, 0.5).mean,
                          st::gev_stat(0.2, 1.5, 0.5).var};
    print_vec("DIST_GEV", d_gev);
    const f64 d_gp[5] = {st::gp_pdf(2.5, 0.25, 2.0, 1.0), st::gp_cdf(2.5, 0.25, 2.0, 1.0),
                         st::gp_inv(0.7, 0.25, 2.0, 1.0), st::gp_stat(0.25, 2.0, 1.0).mean,
                         st::gp_stat(0.25, 2.0, 1.0).var};
    print_vec("DIST_GP", d_gp);
    const f64 d_ev[5] = {st::ev_pdf(0.5, 1.0, 2.0), st::ev_cdf(0.5, 1.0, 2.0),
                         st::ev_inv(0.6, 1.0, 2.0), st::ev_stat(1.0, 2.0).mean,
                         st::ev_stat(1.0, 2.0).var};
    print_vec("DIST_EV", d_ev);
    const f64 d_rice[5] = {st::rice_pdf(2.2, 1.5, 1.2), st::rice_cdf(2.2, 1.5, 1.2),
                           st::rice_inv(0.6, 1.5, 1.2), st::rice_stat(1.5, 1.2).mean,
                           st::rice_stat(1.5, 1.2).var};
    print_vec("DIST_RICE", d_rice);
    const f64 d_ncx2[5] = {st::ncx2_pdf(6.5, 4.0, 3.0), st::ncx2_cdf(6.5, 4.0, 3.0),
                           st::ncx2_inv(0.75, 4.0, 3.0), st::ncx2_stat(4.0, 3.0).mean,
                           st::ncx2_stat(4.0, 3.0).var};
    print_vec("DIST_NCX2", d_ncx2);
    const f64 d_naka[5] = {st::naka_pdf(1.5, 2.5, 3.0), st::naka_cdf(1.5, 2.5, 3.0),
                           st::naka_inv(0.65, 2.5, 3.0), st::naka_stat(2.5, 3.0).mean,
                           st::naka_stat(2.5, 3.0).var};
    print_vec("DIST_NAKA", d_naka);

    // 2. Discrete distributions
    const f64 d_poiss[5] = {st::poiss_pdf(3.0, 4.5), st::poiss_cdf(3.0, 4.5),
                            st::poiss_inv(0.8, 4.5), st::poiss_stat(4.5).mean,
                            st::poiss_stat(4.5).var};
    print_vec("DIST_POISS", d_poiss);
    const f64 d_bino[5] = {st::bino_pdf(4.0, 12.0, 0.35), st::bino_cdf(4.0, 12.0, 0.35),
                           st::bino_inv(0.75, 12.0, 0.35), st::bino_stat(12.0, 0.35).mean,
                           st::bino_stat(12.0, 0.35).var};
    print_vec("DIST_BINO", d_bino);
    const f64 d_geo[5] = {st::geo_pdf(2.0, 0.3), st::geo_cdf(2.0, 0.3), st::geo_inv(0.8, 0.3),
                          st::geo_stat(0.3).mean, st::geo_stat(0.3).var};
    print_vec("DIST_GEO", d_geo);
    const f64 d_unid[5] = {st::unid_pdf(3.0, 6.0), st::unid_cdf(3.0, 6.0), st::unid_inv(0.55, 6.0),
                           st::unid_stat(6.0).mean, st::unid_stat(6.0).var};
    print_vec("DIST_UNID", d_unid);
    const f64 d_nbin[5] = {st::nbin_pdf(5.0, 3.0, 0.4), st::nbin_cdf(5.0, 3.0, 0.4),
                           st::nbin_inv(0.7, 3.0, 0.4), st::nbin_stat(3.0, 0.4).mean,
                           st::nbin_stat(3.0, 0.4).var};
    print_vec("DIST_NBIN", d_nbin);
    const f64 d_hyge[5] = {st::hyge_pdf(2.0, 20.0, 7.0, 5.0), st::hyge_cdf(2.0, 20.0, 7.0, 5.0),
                           st::hyge_inv(0.8, 20.0, 7.0, 5.0), st::hyge_stat(20.0, 7.0, 5.0).mean,
                           st::hyge_stat(20.0, 7.0, 5.0).var};
    print_vec("DIST_HYGE", d_hyge);

    // 3. Descriptive statistics
    const f64 dx[8] = {1.2, 2.5, 2.5, 4.0, 5.3, 7.1, 8.9, 12.4};
    const f64 dy[8] = {2.1, 3.0, 4.2, 5.1, 6.8, 7.5, 10.2, 11.9};
    const f64 desc[13] = {
        st::mean(dx),           st::var(dx, false),      st::var(dx, true),
        st::stddev(dx, false),  st::skewness(dx, false), st::skewness(dx, true),
        st::kurtosis(dx, false), st::kurtosis(dx, true), st::cov(dx, dy, false),
        st::corrcoef(dx, dy),   st::quantile(dx, 0.35),  st::median(dx),
        st::iqr(dx)};
    print_vec("DESC_BASIC", desc);

    // 4. Pairwise distances (4 x 3 col-major)
    const f64 px[12] = {1.0, 2.0, 0.0, 4.0, 0.0, 3.0, 1.0, 2.0, 2.0, 1.0, 5.0, 0.0};
    f64 pd[6]{};
    (void)st::pdist(px, 4, 3, st::DistanceMetric::kEuclidean, 2.0, {}, pd);
    print_vec("PDIST_EUC", pd);
    (void)st::pdist(px, 4, 3, st::DistanceMetric::kCityblock, 2.0, {}, pd);
    print_vec("PDIST_L1", pd);
    (void)st::pdist(px, 4, 3, st::DistanceMetric::kChebychev, 2.0, {}, pd);
    print_vec("PDIST_CHEB", pd);
    (void)st::pdist(px, 4, 3, st::DistanceMetric::kMinkowski, 3.0, {}, pd);
    print_vec("PDIST_MINK3", pd);
    (void)st::pdist(px, 4, 3, st::DistanceMetric::kCosine, 2.0, {}, pd);
    print_vec("PDIST_COS", pd);
    (void)st::pdist(px, 4, 3, st::DistanceMetric::kCorrelation, 2.0, {}, pd);
    print_vec("PDIST_CORR", pd);
    (void)st::pdist(px, 4, 3, st::DistanceMetric::kSpearman, 2.0, {}, pd);
    print_vec("PDIST_SPEAR", pd);
    (void)st::pdist(px, 4, 3, st::DistanceMetric::kSeuclidean, 2.0, {}, pd);
    print_vec("PDIST_SEUC", pd);
    (void)st::pdist(px, 4, 3, st::DistanceMetric::kHamming, 2.0, {}, pd);
    print_vec("PDIST_HAM", pd);
    (void)st::pdist(px, 4, 3, st::DistanceMetric::kJaccard, 2.0, {}, pd);
    print_vec("PDIST_JAC", pd);
    // Binary 4x4 matrix (col-major) where MATLAB and SciPy Jaccard agree identically
    const f64 px_bin[16] = {1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0};
    (void)st::pdist(px_bin, 4, 4, st::DistanceMetric::kJaccard, 2.0, {}, pd);
    print_vec("PDIST_JAC_BIN", pd);

    const f64 mx_ref[10] = {1.0, 2.0, 3.0, 4.0, 6.0, 2.0, 1.0, 4.0, 3.0, 7.0}; // 5x2 col-major
    const f64 my_qry[4] = {2.5, 5.0, 3.0, 1.0};                               // 2x2 col-major
    f64 md2[2]{};
    (void)st::mahal_qr(my_qry, 2, mx_ref, 5, 2, md2);
    print_vec("MAHAL_QR", md2);

    // 5. Linkage, cophenet, inconsistent, silhouette
    (void)st::pdist(px, 4, 3, st::DistanceMetric::kEuclidean, 2.0, {}, pd);
    f64 z_avg[9]{}, z_ward[9]{}, dcoph[6]{}, inc[12]{};
    f64 coph_r = 0.0;
    (void)st::linkage(pd, 4, st::LinkageMethod::kAverage, z_avg);
    (void)st::linkage(pd, 4, st::LinkageMethod::kWard, z_ward);
    (void)st::cophenet(z_avg, 3, pd, dcoph, coph_r);
    (void)st::inconsistent(z_avg, 3, 2, inc);
    print_vec("LINK_AVG", z_avg);
    print_vec("LINK_WARD", z_ward);
    print_val("COPHENET_R", coph_r);
    print_vec("INCONSIST_AVG", inc);

    const i32 slbl[4] = {1, 1, 2, 2};
    f64 sil[4]{};
    (void)st::silhouette(px, 4, 3, slbl, st::DistanceMetric::kEuclidean, 2.0, sil);
    print_vec("SILHOUETTE", sil);

    // 6. ANOVA & nonparametric rank tests
    const f64 a1_v[9] = {1.2, 2.4, 3.1, 5.5, 6.1, 7.3, 10.2, 11.8, 12.1};
    const usize a1_o[4] = {0, 3, 6, 9};
    st::Anova1Result a1{};
    (void)st::anova1_grouped(a1_v, a1_o, a1);
    const f64 a1_arr[4] = {a1.p, a1.f, a1.ss_between, a1.ss_within};
    print_vec("ANOVA1", a1_arr);

    st::KruskalResult kw{};
    (void)st::kruskal_wallis(a1_v, a1_o, kw);
    const f64 kw_arr[2] = {kw.p, kw.h};
    print_vec("KRUSKAL", kw_arr);

    const f64 a2_m[12] = {1.0, 2.5, 4.1, 5.2, 2.1, 3.8, 5.0, 6.7, 3.4, 5.1, 6.2, 8.0}; // 4x3
    st::Anova2Result a2{};
    (void)st::anova2(a2_m, 4, 3, a2);
    const f64 a2_arr[6] = {a2.p_cols, a2.p_rows, a2.f_cols, a2.f_rows, a2.ss_cols, a2.ss_err};
    print_vec("ANOVA2", a2_arr);

    st::FriedmanResult fr{};
    (void)st::friedman(a2_m, 4, 3, 1, fr);
    const f64 fr_arr[2] = {fr.p, fr.q};
    print_vec("FRIEDMAN", fr_arr);

    // 7. Regression: regress, lscov, ridge, glmfit
    const f64 rx[18] = {1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 2, 1, 4, 3, 6, 5}; // 6x3
    const f64 ry[6] = {5.2, 6.8, 11.1, 12.9, 17.3, 18.8};
    f64 rbeta[3]{}, rbint[6]{}, rres[6]{}, rrint[12]{};
    st::RegressStats rstat{};
    (void)st::regress(ry, rx, 6, 3, 0.05, rbeta, rbint, rres, rrint, rstat);
    print_vec("REGRESS_BETA", rbeta);
    print_vec("REGRESS_BINT", rbint);
    const f64 rstat_arr[4] = {rstat.r2, rstat.f, rstat.p_f, rstat.sigma2};
    print_vec("REGRESS_STATS", rstat_arr);

    const f64 lw[6] = {1.0, 2.0, 1.5, 0.5, 2.5, 1.0};
    f64 lxb[3]{}, lstd[3]{}, lS[9]{};
    f64 lmse = 0.0;
    (void)st::lscov(rx, ry, 6, 3, lw, false, lxb, lstd, lmse, lS);
    print_vec("LSCOV_W_X", lxb);
    print_vec("LSCOV_W_STD", lstd);
    print_val("LSCOV_W_MSE", lmse);

    const f64 rdg_x[12] = {1, 2, 3, 4, 5, 6, 2, 1, 4, 3, 6, 5}; // 6x2
    const f64 rdg_k[2] = {0.0, 2.5};
    f64 rdg_b0[6]{}, rdg_b1[4]{};
    (void)st::ridge(ry, rdg_x, 6, 2, rdg_k, false, rdg_b0);
    (void)st::ridge(ry, rdg_x, 6, 2, rdg_k, true, rdg_b1);
    print_vec("RIDGE_UNSCALED", rdg_b0);
    print_vec("RIDGE_SCALED", rdg_b1);

    const f64 gx[8] = {-2.0, -1.5, -0.8, -0.2, 0.3, 0.9, 1.6, 2.2};
    const f64 gy_bin[8] = {0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0};
    const f64 gy_poi[8] = {1.0, 1.0, 2.0, 3.0, 4.0, 7.0, 11.0, 18.0};
    st::GlmFitOut g_bin, g_poi;
    (void)st::glmfit(gx, gy_bin, 8, 1, st::GlmDist::kBinomial, st::GlmLink::kLogit, g_bin);
    (void)st::glmfit(gx, gy_poi, 8, 1, st::GlmDist::kPoisson, st::GlmLink::kLog, g_poi);
    print_vec("GLMFIT_BIN_BETA", g_bin.beta.as_ref());
    print_val("GLMFIT_BIN_DEV", g_bin.deviance);
    print_vec("GLMFIT_POI_BETA", g_poi.beta.as_ref());
    print_val("GLMFIT_POI_DEV", g_poi.deviance);

    // 8. PCA, LDA, nearcorr
    const f64 xp[15] = {1.0, 2.0, 3.5, 4.1, 6.2, 2.1, 1.8, 4.0, 5.2, 5.9,
                        0.5, 1.5, 2.0, 3.8, 4.7}; // 5x3
    f64 p_coeff[9]{}, p_score[15]{}, p_lat[3]{}, p_tsq[5]{}, p_exp[3]{}, p_mu[3]{};
    (void)st::pca(xp, 5, 3, p_coeff, p_score, p_lat, p_tsq, p_exp, p_mu);
    print_vec("PCA_COEFF", p_coeff);
    print_vec("PCA_LATENT", p_lat);
    print_vec("PCA_TSQ", p_tsq);
    print_vec("PCA_EXPLAINED", p_exp);

    const f64 bad_c[9] = {1.0, 0.9, 0.9, 0.9, 1.0, -0.9, 0.9, -0.9, 1.0};
    f64 nc[9]{};
    (void)st::nearcorr(bad_c, 3, {}, 1e-10, 500, nc);
    print_vec("NEARCORR", nc);

    // 9. Copulas & Multivariate Densities
    const f64 cu[4] = {0.25, 0.75, 0.4, 0.8}; // 2x2 col-major: (0.25,0.4), (0.75,0.8)
    const f64 cr2[4] = {1.0, 0.6, 0.6, 1.0};
    f64 cout2[2]{};
    print_val("BIVNORM_CDF", st::bivnorm_cdf(0.8, -0.4, 0.65));
    (void)st::copula_pdf_gaussian(cu, 2, 2, cr2, cout2);
    print_vec("COP_GAUSS_PDF", cout2);
    (void)st::copula_cdf_gaussian_2d(cu, 2, 0.6, cout2);
    print_vec("COP_GAUSS_CDF", cout2);
    (void)st::copula_pdf_t(cu, 2, 2, cr2, 5.0, cout2);
    print_vec("COP_T_PDF", cout2);
    (void)st::copula_pdf_clayton(cu, 2, 2.5, cout2);
    print_vec("COP_CLAYTON_PDF", cout2);
    (void)st::copula_cdf_clayton(cu, 2, 2.5, cout2);
    print_vec("COP_CLAYTON_CDF", cout2);
    (void)st::copula_pdf_frank(cu, 2, 3.0, cout2);
    print_vec("COP_FRANK_PDF", cout2);
    (void)st::copula_cdf_frank(cu, 2, 3.0, cout2);
    print_vec("COP_FRANK_CDF", cout2);
    (void)st::copula_pdf_gumbel(cu, 2, 2.2, cout2);
    print_vec("COP_GUMBEL_PDF", cout2);
    (void)st::copula_cdf_gumbel(cu, 2, 2.2, cout2);
    print_vec("COP_GUMBEL_CDF", cout2);

    const f64 mvx[4] = {0.5, -1.2, -0.3, 0.7}; // 2x2 col-major
    const f64 mvs[4] = {2.0, 0.7, 0.7, 1.5};
    const f64 mvdf[2] = {5.0, 10.0};
    (void)st::mvnpdf_full(mvx, 2, 2, mvs, cout2);
    print_vec("MVNPDF", cout2);
    (void)st::mvtpdf_full(mvx, 2, 2, cr2, mvdf, cout2);
    print_vec("MVTPDF", cout2);
    const f64 mnc[3] = {3.0, 2.0, 1.0};
    const f64 mnp[3] = {0.5, 0.3, 0.2};
    f64 mno[1]{};
    (void)st::mnpdf_rows(mnc, 1, 3, mnp, 1, mno);
    print_val("MNPDF", mno[0]);

    f64 cor_r[4]{}, cor_s[2]{};
    (void)st::corrcov(mvs, 2, cor_r, cor_s);
    print_vec("CORRCOV_R", cor_r);
    print_vec("CORRCOV_S", cor_s);

    // 10. Hypothesis tests
    const f64 hx[8] = {1.2, 2.4, 3.1, 4.5, 5.0, 6.2, 7.1, 8.9};
    const f64 hy[8] = {3.0, 4.1, 5.8, 6.2, 7.5, 8.9, 9.4, 11.2};
    st::TTestResult t1{};
    (void)st::ttest_1sample(hx, 3.5, 0.05, st::TestTail::kBoth, t1);
    const f64 t1_arr[5] = {t1.p, t1.ci_lo, t1.ci_hi, t1.tstat, t1.sd};
    print_vec("TTEST1", t1_arr);

    st::TTest2Result t2eq{}, t2un{}, t2r{};
    (void)st::ttest_2sample(hx, hy, 0.05, st::TestTail::kBoth, true, t2eq);
    (void)st::ttest_2sample(hx, hy, 0.05, st::TestTail::kBoth, false, t2un);
    (void)st::ttest_2sample(hx, hy, 0.05, st::TestTail::kRight, true, t2r);
    const f64 t2eq_arr[5] = {t2eq.p, t2eq.ci_lo, t2eq.ci_hi, t2eq.tstat, t2eq.df};
    const f64 t2un_arr[5] = {t2un.p, t2un.ci_lo, t2un.ci_hi, t2un.tstat, t2un.df};
    const f64 t2r_arr[2] = {t2r.p, t2r.ci_lo};
    print_vec("TTEST2_EQ", t2eq_arr);
    print_vec("TTEST2_UNEQ", t2un_arr);
    print_vec("TTEST2_RIGHT", t2r_arr);

    st::ZTestResult zt{};
    (void)st::ztest_1sample(hx, 3.5, 2.5, 0.05, st::TestTail::kBoth, zt);
    const f64 zt_arr[4] = {zt.p, zt.ci_lo, zt.ci_hi, zt.zval};
    print_vec("ZTEST", zt_arr);

    st::VarTestResult vt1{};
    (void)st::vartest_1sample(hx, 5.0, 0.05, st::TestTail::kBoth, vt1);
    const f64 vt1_arr[4] = {vt1.p, vt1.ci_lo, vt1.ci_hi, vt1.chisqstat};
    print_vec("VARTEST1", vt1_arr);

    st::VarTest2Result vt2{};
    (void)st::vartest_2sample(hx, hy, 0.05, st::TestTail::kBoth, vt2);
    const f64 vt2_arr[4] = {vt2.p, vt2.ci_lo, vt2.ci_hi, vt2.fstat};
    print_vec("VARTEST2", vt2_arr);

    const f64 ks_x[10] = {-1.2, -0.8, -0.3, 0.1, 0.4, 0.7, 1.1, 1.5, 1.9, 2.3};
    st::KsTestResult ks1{}, ks2{};
    (void)st::kstest_1sample(ks_x, {}, {}, 0.05, st::TestTail::kBoth, ks1);
    const f64 ks1_arr[2] = {ks1.p, ks1.ksstat};
    print_vec("KSTEST1", ks1_arr);
    (void)st::kstest_2sample(hx, hy, 0.05, st::TestTail::kBoth, ks2);
    const f64 ks2_arr[2] = {ks2.p, ks2.ksstat};
    print_vec("KSTEST2", ks2_arr);

    st::SignTestResult sgn{};
    (void)st::signtest(hx, hy, 0.0, 0.05, st::TestTail::kBoth, sgn);
    const f64 sgn_arr[2] = {sgn.p, as<f64>(sgn.sign)};
    print_vec("SIGNTEST", sgn_arr);

    st::FisherTestResult fsh{};
    (void)st::fishertest(8.0, 2.0, 1.0, 9.0, 0.05, st::TestTail::kBoth, fsh);
    const f64 fsh_arr[2] = {fsh.p, fsh.odds_ratio};
    print_vec("FISHERTEST", fsh_arr);

    const f64 c_obs[4] = {18.0, 22.0, 31.0, 29.0};
    const f64 c_exp[4] = {25.0, 25.0, 25.0, 25.0};
    st::Chi2GofResult c2g{};
    (void)st::chi2gof_freq(c_obs, c_exp, 0, 0.05, c2g);
    const f64 c2g_arr[3] = {c2g.p, c2g.chi2stat, c2g.df};
    print_vec("CHI2GOF", c2g_arr);

    const f64 vn_x[9] = {1.1, 2.5, 3.2, 10.0, 10.2, 10.5, 20.0, 24.0, 29.0};
    const f64 vn_g[9] = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0};
    st::VarTestNResult v_bart{}, v_bf{};
    (void)st::vartestn(vn_x, vn_g, st::VarTestNMethod::kBartlett, v_bart);
    (void)st::vartestn(vn_x, vn_g, st::VarTestNMethod::kBrownForsythe, v_bf);
    const f64 vb_arr[2] = {v_bart.p, v_bart.stat};
    const f64 vbf_arr[2] = {v_bf.p, v_bf.stat};
    print_vec("VARTESTN_BART", vb_arr);
    print_vec("VARTESTN_BF", vbf_arr);

    const f64 rns_x[10] = {1.0, 2.0, 1.5, 8.0, 9.0, 2.0, 1.0, 7.0, 8.5, 9.5};
    st::RunsTestResult rns{};
    (void)st::runstest(rns_x, 5.0, 0.05, st::TestTail::kBoth, true, rns);
    const f64 rns_arr[2] = {rns.p, as<f64>(rns.nruns)};
    print_vec("RUNSTEST", rns_arr);

    st::RankTestResult rks_ex{}, rks_ap{}, snr_ex{}, snr_ap{}, ab{};
    (void)st::ranksum(hx, hy, 0.05, st::TestTail::kBoth, 1, rks_ex);
    (void)st::ranksum(hx, hy, 0.05, st::TestTail::kBoth, 2, rks_ap);
    const f64 rke_arr[2] = {rks_ex.p, rks_ex.stat};
    const f64 rka_arr[3] = {rks_ap.p, rks_ap.stat, rks_ap.zval};
    print_vec("RANKSUM_EX", rke_arr);
    print_vec("RANKSUM_AP", rka_arr);

    const f64 sr_x[10] = {1.0, 2.5, 3.1, 4.8, 5.2, 6.7, 7.9, 8.4, 9.1, 10.5};
    (void)st::signrank(sr_x, {}, 4.0, 0.05, st::TestTail::kBoth, 1, snr_ex);
    (void)st::signrank(sr_x, {}, 4.0, 0.05, st::TestTail::kBoth, 2, snr_ap);
    const f64 sre_arr[2] = {snr_ex.p, snr_ex.stat};
    const f64 sra_arr[3] = {snr_ap.p, snr_ap.stat, snr_ap.zval};
    print_vec("SIGNRANK_EX", sre_arr);
    print_vec("SIGNRANK_AP", sra_arr);

    (void)st::ansaribradley(hx, hy, 0.05, st::TestTail::kBoth, ab);
    const f64 ab_arr[2] = {ab.p, ab.stat};
    print_vec("ANSARI", ab_arr);

    const f64 dw_r[5] = {0.1, -0.1, 0.1, -0.1, 0.0};
    const f64 dw_x[10] = {1, 1, 1, 1, 1, 1, 2, 3, 4, 5};
    st::DwTestResult dw_ex{}, dw_ap{};
    (void)st::dwtest(dw_r, dw_x, 5, 2, true, dw_ex);
    (void)st::dwtest(dw_r, dw_x, 5, 2, false, dw_ap);
    const f64 dwe_arr[2] = {dw_ex.dw, dw_ex.p_both};
    const f64 dwa_arr[2] = {dw_ap.dw, dw_ap.p_both};
    print_vec("DWTEST_EX", dwe_arr);
    print_vec("DWTEST_AP", dwa_arr);

    // 11. Distribution MLE fitters & likelihoods
    const f64 pos_d[12] = {0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6, 3.0, 3.5, 4.1, 4.8, 5.9};
    const f64 uni_d[10] = {0.12, 0.21, 0.30, 0.38, 0.45, 0.52, 0.61, 0.69, 0.78, 0.88};
    const f64 nb_d[10] = {0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 12.0};
    const f64 ev_d[10] = {-2.1, -1.2, -0.5, 0.1, 0.6, 1.0, 1.4, 1.9, 2.3, 2.8};
    const f64 gp_d[12] = {0.1, 0.25, 0.4, 0.7, 1.1, 1.6, 2.3, 3.5, 5.2, 8.1, 13.5, 22.0};
    st::Fit1Result f1{};
    st::Fit2Result f2{};
    st::Fit3Result f3{};

    (void)st::normfit(pos_d, {}, {}, 0.05, f2);
    const f64 nf_arr[6] = {f2.p1, f2.p2, f2.ci1_lo, f2.ci1_hi, f2.ci2_lo, f2.ci2_hi};
    print_vec("FIT_NORM", nf_arr);

    (void)st::expfit(pos_d, {}, {}, 0.05, f1);
    const f64 ef_arr[3] = {f1.p1, f1.ci_lo, f1.ci_hi};
    print_vec("FIT_EXP", ef_arr);

    (void)st::poissfit(nb_d, 0.05, f1);
    const f64 pf_arr[3] = {f1.p1, f1.ci_lo, f1.ci_hi};
    print_vec("FIT_POISS", pf_arr);

    (void)st::unifit(pos_d, 0.05, f2);
    const f64 uf_arr[6] = {f2.p1, f2.p2, f2.ci1_lo, f2.ci1_hi, f2.ci2_lo, f2.ci2_hi};
    print_vec("FIT_UNIF", uf_arr);

    (void)st::binofit_scalar(7.0, 20.0, 0.05, f1);
    const f64 bf_arr[3] = {f1.p1, f1.ci_lo, f1.ci_hi};
    print_vec("FIT_BINO", bf_arr);

    (void)st::raylfit(pos_d, 0.05, f1);
    const f64 rf_arr[3] = {f1.p1, f1.ci_lo, f1.ci_hi};
    print_vec("FIT_RAYL", rf_arr);

    (void)st::gamfit(pos_d, {}, {}, 0.05, true, f2);
    const f64 gf_arr[2] = {f2.p1, f2.p2};
    print_vec("FIT_GAM", gf_arr);

    (void)st::wblfit(pos_d, {}, {}, 0.05, true, f2);
    const f64 wf_arr[2] = {f2.p1, f2.p2};
    print_vec("FIT_WBL", wf_arr);

    (void)st::betafit(uni_d, 0.05, true, f2);
    const f64 btf_arr[2] = {f2.p1, f2.p2};
    print_vec("FIT_BETA", btf_arr);

    (void)st::nbinfit(nb_d, 0.05, true, f2);
    const f64 nbf_arr[2] = {f2.p1, f2.p2};
    print_vec("FIT_NBIN", nbf_arr);

    (void)st::evfit(ev_d, {}, {}, 0.05, true, f2);
    const f64 evf_arr[2] = {f2.p1, f2.p2};
    print_vec("FIT_EV", evf_arr);

    (void)st::gpfit(gp_d, 0.05, true, f2);
    const f64 gpf_arr[2] = {f2.p1, f2.p2};
    print_vec("FIT_GP", gpf_arr);

    (void)st::gevfit(ev_d, 0.05, true, f3);
    const f64 gevf_arr[3] = {f3.p1, f3.p2, f3.p3};
    print_vec("FIT_GEV", gevf_arr);

    const f64 likes[9] = {
        st::normlike(2.5, 1.5, pos_d),    st::explike(2.5, pos_d),
        st::lognlike(0.8, 0.5, pos_d),    st::gamlike(2.5, 1.2, pos_d),
        st::betalike(2.0, 2.5, uni_d),    st::wbllike(3.0, 2.0, pos_d),
        st::evlike(1.0, 1.5, ev_d),       st::gevlike(0.1, 1.5, 0.5, ev_d),
        st::gplike(0.2, 2.0, gp_d)};
    print_vec("LIKELIHOODS", likes);

    // 12. Robust, canonical, studentized range, multcompare, tiedrank, corr pvals, halton
    const f64 xrf[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8};
    const f64 yrf[8] = {3.1, 4.9, 7.2, 8.8, 11.1, 13.0, 14.9, 100.0};
    f64 rfb[2]{};
    f64 rfs = 0.0;
    (void)st::robustfit(xrf, 8, 2, yrf, st::RobustWeight::kBisquare, Limits<f64>::quiet_nan(), rfb,
                        rfs);
    print_vec("ROBUSTFIT_BISQ", rfb);

    const f64 xcc[16] = {1, 2, 3, 4, 5, 6, 7, 8, 2, 1, 5, 3, 7, 4, 8, 6};
    const f64 ycc[16] = {2, 4, 6, 8, 10, 12, 14, 16, 1, 3, 2, 5, 4, 7, 6, 9};
    f64 cca[4]{}, ccb[4]{}, ccr[2]{};
    (void)st::canoncorr(xcc, 8, 2, ycc, 2, cca, ccb, ccr);
    print_vec("CANONCORR_R", ccr);
    print_vec("CANONCORR_A", cca);

    const f64 srng[2] = {st::studentized_range_cdf(3.773, 3, 12.0),
                         st::studentized_range_inv(0.95, 3, 12.0)};
    print_vec("STUD_RANGE", srng);

    const f64 tr_in[6] = {10, 20, 30, 20, 10, 40};
    f64 tr_out[6]{};
    f64 tieadj = 0.0;
    (void)st::tiedrank(tr_in, tr_out, tieadj);
    print_vec("TIEDRANK", tr_out);
    print_val("TIEDRANK_ADJ", tieadj);

    const f64 cpv[4] = {
        st::corr_pearson_pval(st::CorrTail::kBoth, 0.75, 12),
        st::corr_kendall_normal_pval(st::CorrTail::kBoth, 0.5, 12),
        st::corr_spearman_no_ties_pval(st::CorrTail::kBoth, 0.85, 15),
        st::corr_spearman_no_ties_pval(st::CorrTail::kRight, 0.8285714285714286, 6)};
    print_vec("CORR_PVALS", cpv);

    f64 hnet[8]{};
    (void)st::halton_net(4, 2, 1, 0, false, hnet);
    print_vec("HALTON", hnet);
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1 && keel::StringRef(argv[1]) == "--dump-ref") {
        dump_reference();
        return 0;
    }
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
    test_hardening_regressions();
    if (g_failures == 0) {
        keel::println("ALL keel::stats TESTS PASSED");
        return 0;
    }
    keel::eprintln("%d FAILURE(S)", g_failures);
    return 1;
}
