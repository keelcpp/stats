"""Differential verification of keel::stats against SciPy/NumPy AND MATLAB R2025b."""

import math
import os
import shutil
import subprocess
import sys
import numpy as np
from scipy import stats as sp_stats
from scipy.spatial.distance import pdist as sp_pdist
from scipy.cluster.hierarchy import linkage as sp_linkage, cophenet as sp_cophenet, inconsistent as sp_inconsistent


def parse_kv(text: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if not k or any(c in k for c in " []{}()"):
            continue
        parts = [p.strip() for p in v.split(",") if p.strip()]
        try:
            out[k] = [float(p) for p in parts]
        except ValueError:
            continue
    return out


def find_exe() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(root, "build", "Release", "test_stats.exe"),
        os.path.join(root, "build", "RelWithDebInfo", "test_stats.exe"),
        os.path.join(root, "build", "Debug", "test_stats.exe"),
        os.path.join(root, "build", "test_stats.exe"),
        os.path.join(root, "build", "test_stats"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Could not find test_stats binary in {candidates}")


def run_keel(exe: str) -> dict[str, list[float]]:
    proc = subprocess.run([exe, "--dump-ref"], capture_output=True, text=True, check=True)
    return parse_kv(proc.stdout)


def compute_scipy() -> dict[str, tuple[list[float], float]]:
    ref: dict[str, tuple[list[float], float]] = {}

    # 1. Continuous distributions
    ref["DIST_NORM"] = ([
        sp_stats.norm.pdf(1.2, 0.5, 2.0),
        sp_stats.norm.cdf(1.2, 0.5, 2.0),
        sp_stats.norm.ppf(0.85, 0.5, 2.0),
        0.5, 4.0,
    ], 1e-12)
    ref["DIST_T"] = ([
        sp_stats.t.pdf(1.3, 7.0),
        sp_stats.t.cdf(1.3, 7.0),
        sp_stats.t.ppf(0.9, 7.0),
        float(sp_stats.t.mean(7.0)),
        float(sp_stats.t.var(7.0)),
    ], 1e-10)
    ref["DIST_CHI2"] = ([
        sp_stats.chi2.pdf(4.2, 5.0),
        sp_stats.chi2.cdf(4.2, 5.0),
        sp_stats.chi2.ppf(0.95, 5.0),
        5.0, 10.0,
    ], 1e-10)
    ref["DIST_F"] = ([
        sp_stats.f.pdf(2.1, 5.0, 12.0),
        sp_stats.f.cdf(2.1, 5.0, 12.0),
        sp_stats.f.ppf(0.95, 5.0, 12.0),
        float(sp_stats.f.mean(5.0, 12.0)),
        float(sp_stats.f.var(5.0, 12.0)),
    ], 1e-10)
    ref["DIST_GAM"] = ([
        sp_stats.gamma.pdf(3.2, 2.5, scale=1.5),
        sp_stats.gamma.cdf(3.2, 2.5, scale=1.5),
        sp_stats.gamma.ppf(0.8, 2.5, scale=1.5),
        float(sp_stats.gamma.mean(2.5, scale=1.5)),
        float(sp_stats.gamma.var(2.5, scale=1.5)),
    ], 1e-10)
    ref["DIST_BETA"] = ([
        sp_stats.beta.pdf(0.4, 2.0, 5.0),
        sp_stats.beta.cdf(0.4, 2.0, 5.0),
        sp_stats.beta.ppf(0.7, 2.0, 5.0),
        float(sp_stats.beta.mean(2.0, 5.0)),
        float(sp_stats.beta.var(2.0, 5.0)),
    ], 1e-10)
    ref["DIST_EXP"] = ([
        sp_stats.expon.pdf(2.0, scale=3.0),
        sp_stats.expon.cdf(2.0, scale=3.0),
        sp_stats.expon.ppf(0.6, scale=3.0),
        3.0, 9.0,
    ], 1e-12)
    ref["DIST_WBL"] = ([
        sp_stats.weibull_min.pdf(1.8, 3.0, scale=2.0),
        sp_stats.weibull_min.cdf(1.8, 3.0, scale=2.0),
        sp_stats.weibull_min.ppf(0.75, 3.0, scale=2.0),
        float(sp_stats.weibull_min.mean(3.0, scale=2.0)),
        float(sp_stats.weibull_min.var(3.0, scale=2.0)),
    ], 1e-12)
    ref["DIST_LOGN"] = ([
        sp_stats.lognorm.pdf(2.5, 0.8, scale=math.exp(0.5)),
        sp_stats.lognorm.cdf(2.5, 0.8, scale=math.exp(0.5)),
        sp_stats.lognorm.ppf(0.6, 0.8, scale=math.exp(0.5)),
        float(sp_stats.lognorm.mean(0.8, scale=math.exp(0.5))),
        float(sp_stats.lognorm.var(0.8, scale=math.exp(0.5))),
    ], 1e-11)
    ref["DIST_UNIF"] = ([
        sp_stats.uniform.pdf(2.5, 1.0, 4.0),
        sp_stats.uniform.cdf(2.5, 1.0, 4.0),
        sp_stats.uniform.ppf(0.4, 1.0, 4.0),
        3.0, 16.0 / 12.0,
    ], 1e-13)
    ref["DIST_RAYL"] = ([
        sp_stats.rayleigh.pdf(2.0, scale=1.5),
        sp_stats.rayleigh.cdf(2.0, scale=1.5),
        sp_stats.rayleigh.ppf(0.7, scale=1.5),
        float(sp_stats.rayleigh.mean(scale=1.5)),
        float(sp_stats.rayleigh.var(scale=1.5)),
    ], 1e-12)
    # Note SciPy genextreme uses c = -k
    ref["DIST_GEV"] = ([
        sp_stats.genextreme.pdf(1.2, -0.2, loc=0.5, scale=1.5),
        sp_stats.genextreme.cdf(1.2, -0.2, loc=0.5, scale=1.5),
        sp_stats.genextreme.ppf(0.8, -0.2, loc=0.5, scale=1.5),
        float(sp_stats.genextreme.mean(-0.2, loc=0.5, scale=1.5)),
        float(sp_stats.genextreme.var(-0.2, loc=0.5, scale=1.5)),
    ], 1e-11)
    ref["DIST_GP"] = ([
        sp_stats.genpareto.pdf(2.5, 0.25, loc=1.0, scale=2.0),
        sp_stats.genpareto.cdf(2.5, 0.25, loc=1.0, scale=2.0),
        sp_stats.genpareto.ppf(0.7, 0.25, loc=1.0, scale=2.0),
        float(sp_stats.genpareto.mean(0.25, loc=1.0, scale=2.0)),
        float(sp_stats.genpareto.var(0.25, loc=1.0, scale=2.0)),
    ], 1e-11)
    ref["DIST_EV"] = ([
        sp_stats.gumbel_l.pdf(0.5, loc=1.0, scale=2.0),
        sp_stats.gumbel_l.cdf(0.5, loc=1.0, scale=2.0),
        sp_stats.gumbel_l.ppf(0.6, loc=1.0, scale=2.0),
        float(sp_stats.gumbel_l.mean(loc=1.0, scale=2.0)),
        float(sp_stats.gumbel_l.var(loc=1.0, scale=2.0)),
    ], 1e-11)
    ref["DIST_RICE"] = ([
        sp_stats.rice.pdf(2.2, 1.5 / 1.2, scale=1.2),
        sp_stats.rice.cdf(2.2, 1.5 / 1.2, scale=1.2),
        sp_stats.rice.ppf(0.6, 1.5 / 1.2, scale=1.2),
        float(sp_stats.rice.mean(1.5 / 1.2, scale=1.2)),
        float(sp_stats.rice.var(1.5 / 1.2, scale=1.2)),
    ], 1e-8)
    ref["DIST_NCX2"] = ([
        sp_stats.ncx2.pdf(6.5, 4.0, 3.0),
        sp_stats.ncx2.cdf(6.5, 4.0, 3.0),
        sp_stats.ncx2.ppf(0.75, 4.0, 3.0),
        float(sp_stats.ncx2.mean(4.0, 3.0)),
        float(sp_stats.ncx2.var(4.0, 3.0)),
    ], 1e-8)
    ref["DIST_NAKA"] = ([
        sp_stats.nakagami.pdf(1.5, 2.5, scale=math.sqrt(3.0)),
        sp_stats.nakagami.cdf(1.5, 2.5, scale=math.sqrt(3.0)),
        sp_stats.nakagami.ppf(0.65, 2.5, scale=math.sqrt(3.0)),
        float(sp_stats.nakagami.mean(2.5, scale=math.sqrt(3.0))),
        float(sp_stats.nakagami.var(2.5, scale=math.sqrt(3.0))),
    ], 1e-10)

    # 2. Discrete distributions
    ref["DIST_POISS"] = ([
        sp_stats.poisson.pmf(3, 4.5),
        sp_stats.poisson.cdf(3, 4.5),
        sp_stats.poisson.ppf(0.8, 4.5),
        4.5, 4.5,
    ], 1e-11)
    ref["DIST_BINO"] = ([
        sp_stats.binom.pmf(4, 12, 0.35),
        sp_stats.binom.cdf(4, 12, 0.35),
        sp_stats.binom.ppf(0.75, 12, 0.35),
        float(sp_stats.binom.mean(12, 0.35)),
        float(sp_stats.binom.var(12, 0.35)),
    ], 1e-11)
    # Geometric (number of failures before first success -> SciPy geom loc=-1)
    ref["DIST_GEO"] = ([
        sp_stats.geom.pmf(2, 0.3, loc=-1),
        sp_stats.geom.cdf(2, 0.3, loc=-1),
        sp_stats.geom.ppf(0.8, 0.3, loc=-1),
        float(sp_stats.geom.mean(0.3, loc=-1)),
        float(sp_stats.geom.var(0.3, loc=-1)),
    ], 1e-12)
    ref["DIST_UNID"] = ([
        sp_stats.randint.pmf(3, 1, 7),
        sp_stats.randint.cdf(3, 1, 7),
        sp_stats.randint.ppf(0.55, 1, 7),
        3.5, 35.0 / 12.0,
    ], 1e-12)
    ref["DIST_NBIN"] = ([
        sp_stats.nbinom.pmf(5, 3.0, 0.4),
        sp_stats.nbinom.cdf(5, 3.0, 0.4),
        sp_stats.nbinom.ppf(0.7, 3.0, 0.4),
        float(sp_stats.nbinom.mean(3.0, 0.4)),
        float(sp_stats.nbinom.var(3.0, 0.4)),
    ], 1e-11)
    ref["DIST_HYGE"] = ([
        sp_stats.hypergeom.pmf(2, 20, 7, 5),
        sp_stats.hypergeom.cdf(2, 20, 7, 5),
        sp_stats.hypergeom.ppf(0.8, 20, 7, 5),
        float(sp_stats.hypergeom.mean(20, 7, 5)),
        float(sp_stats.hypergeom.var(20, 7, 5)),
    ], 1e-11)

    # 3. Descriptive
    dx = np.array([1.2, 2.5, 2.5, 4.0, 5.3, 7.1, 8.9, 12.4])
    dy = np.array([2.1, 3.0, 4.2, 5.1, 6.8, 7.5, 10.2, 11.9])
    ref["DESC_BASIC"] = ([
        float(np.mean(dx)),
        float(np.var(dx, ddof=1)),
        float(np.var(dx, ddof=0)),
        float(np.std(dx, ddof=1)),
        float(sp_stats.skew(dx, bias=True)),
        float(sp_stats.skew(dx, bias=False)),
        float(sp_stats.kurtosis(dx, fisher=False, bias=True)),
        float(sp_stats.kurtosis(dx, fisher=False, bias=False)),
        float(np.cov(dx, dy, ddof=1)[0, 1]),
        float(np.corrcoef(dx, dy)[0, 1]),
        float(np.quantile(dx, 0.35, method="hazen")),
        float(np.median(dx)),
        float(np.quantile(dx, 0.75, method="hazen") - np.quantile(dx, 0.25, method="hazen")),
    ], 1e-11)

    # 4. Distances
    px = np.array([[1.0, 0.0, 2.0], [2.0, 3.0, 1.0], [0.0, 1.0, 5.0], [4.0, 2.0, 0.0]])
    ref["PDIST_EUC"] = (sp_pdist(px, "euclidean").tolist(), 1e-12)
    ref["PDIST_L1"] = (sp_pdist(px, "cityblock").tolist(), 1e-12)
    ref["PDIST_CHEB"] = (sp_pdist(px, "chebyshev").tolist(), 1e-12)
    ref["PDIST_MINK3"] = (sp_pdist(px, "minkowski", p=3.0).tolist(), 1e-12)
    ref["PDIST_COS"] = (sp_pdist(px, "cosine").tolist(), 1e-12)
    ref["PDIST_CORR"] = (sp_pdist(px, "correlation").tolist(), 1e-12)
    ref["PDIST_SEUC"] = (sp_pdist(px, "seuclidean").tolist(), 1e-12)
    ref["PDIST_HAM"] = (sp_pdist(px, "hamming").tolist(), 1e-12)
    px_bin = np.array([[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 1.0],
                       [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
    ref["PDIST_JAC_BIN"] = (sp_pdist(px_bin, "jaccard").tolist(), 1e-12)

    # 5. Linkage, cophenet, inconsistent
    d_euc = sp_pdist(px, "euclidean")
    z_avg = sp_linkage(d_euc, method="average")
    z_ward = sp_linkage(d_euc, method="ward")
    coph_r, _ = sp_cophenet(z_avg, d_euc)
    inc = sp_inconsistent(z_avg, d=2)
    ref["COPHENET_R"] = ([float(coph_r)], 1e-12)
    # Inconsistent columns: mean, std, count, coeff -> col-major flatten
    ref["INCONSIST_AVG"] = (inc.T.ravel().tolist(), 1e-11)

    # 6. ANOVA & Kruskal-Wallis
    g1 = np.array([1.2, 2.4, 3.1])
    g2 = np.array([5.5, 6.1, 7.3])
    g3 = np.array([10.2, 11.8, 12.1])
    f_a1, p_a1 = sp_stats.f_oneway(g1, g2, g3)
    kw_h, kw_p = sp_stats.kruskal(g1, g2, g3)
    ref["KRUSKAL"] = ([float(kw_p), float(kw_h)], 1e-11)

    # 7. Multivariate densities
    mvx = np.array([[0.5, -0.3], [-1.2, 0.7]])
    mvs = np.array([[2.0, 0.7], [0.7, 1.5]])
    ref["MVNPDF"] = (sp_stats.multivariate_normal.pdf(mvx, mean=[0, 0], cov=mvs).tolist(), 1e-12)
    cr2 = np.array([[1.0, 0.6], [0.6, 1.0]])
    ref["MVTPDF"] = ([
        float(sp_stats.multivariate_t.pdf(mvx[0], loc=[0, 0], shape=cr2, df=5.0)),
        float(sp_stats.multivariate_t.pdf(mvx[1], loc=[0, 0], shape=cr2, df=10.0)),
    ], 1e-12)
    ref["MNPDF"] = ([float(sp_stats.multinomial.pmf([3, 2, 1], 6, [0.5, 0.3, 0.2]))], 1e-12)
    ref["BIVNORM_CDF"] = ([
        float(sp_stats.multivariate_normal.cdf([0.8, -0.4], mean=[0, 0], cov=[[1.0, 0.65], [0.65, 1.0]]))
    ], 1e-8)

    # 8. Hypothesis tests
    hx = np.array([1.2, 2.4, 3.1, 4.5, 5.0, 6.2, 7.1, 8.9])
    hy = np.array([3.0, 4.1, 5.8, 6.2, 7.5, 8.9, 9.4, 11.2])
    t1 = sp_stats.ttest_1samp(hx, 3.5)
    ci1 = t1.confidence_interval(0.95)
    ref["TTEST1"] = ([float(t1.pvalue), float(ci1.low), float(ci1.high), float(t1.statistic), float(np.std(hx, ddof=1))], 1e-11)

    t2eq = sp_stats.ttest_ind(hx, hy, equal_var=True)
    ci2eq = t2eq.confidence_interval(0.95)
    ref["TTEST2_EQ"] = ([float(t2eq.pvalue), float(ci2eq.low), float(ci2eq.high), float(t2eq.statistic), 14.0], 1e-11)

    t2un = sp_stats.ttest_ind(hx, hy, equal_var=False)
    ci2un = t2un.confidence_interval(0.95)
    ref["TTEST2_UNEQ"] = ([float(t2un.pvalue), float(ci2un.low), float(ci2un.high), float(t2un.statistic), float(t2un.df)], 1e-11)

    ks_x = np.array([-1.2, -0.8, -0.3, 0.1, 0.4, 0.7, 1.1, 1.5, 1.9, 2.3])
    ks1 = sp_stats.kstest(ks_x, "norm")
    ref["KSTEST1"] = ([float(ks1.pvalue), float(ks1.statistic)], 1e-6)

    fsh = sp_stats.fisher_exact([[8, 2], [1, 9]])
    ref["FISHERTEST"] = ([float(fsh.pvalue), float(fsh.statistic)], 1e-11)

    c2g = sp_stats.chisquare([18.0, 22.0, 31.0, 29.0], [25.0, 25.0, 25.0, 25.0])
    ref["CHI2GOF"] = ([float(c2g.pvalue), float(c2g.statistic), 3.0], 1e-11)

    vn1, vn2, vn3 = [1.1, 2.5, 3.2], [10.0, 10.2, 10.5], [20.0, 24.0, 29.0]
    bart = sp_stats.bartlett(vn1, vn2, vn3)
    bf = sp_stats.levene(vn1, vn2, vn3, center="median")
    ref["VARTESTN_BART"] = ([float(bart.pvalue), float(bart.statistic)], 1e-11)
    ref["VARTESTN_BF"] = ([float(bf.pvalue), float(bf.statistic)], 1e-11)

    # Studentized range
    ref["STUD_RANGE"] = ([
        float(sp_stats.studentized_range.cdf(3.773, 3, 12.0)),
        float(sp_stats.studentized_range.ppf(0.95, 3, 12.0)),
    ], 1e-5)

    # Tied ranks
    ref["TIEDRANK"] = (sp_stats.rankdata([10, 20, 30, 20, 10, 40]).tolist(), 1e-12)

    return ref


MATLAB_SCRIPT = r"""
format long g
pvec = @(k, v) fprintf('%s=%s\n', k, strjoin(arrayfun(@(z) sprintf('%.17g', z), v(:)', 'UniformOutput', false), ','));

% 1. Continuous distributions
[m, v] = normstat(0.5, 2.0);
pvec('DIST_NORM', [normpdf(1.2,0.5,2.0), normcdf(1.2,0.5,2.0), norminv(0.85,0.5,2.0), m, v]);
[m, v] = tstat(7.0);
pvec('DIST_T', [tpdf(1.3,7.0), tcdf(1.3,7.0), tinv(0.9,7.0), m, v]);
[m, v] = chi2stat(5.0);
pvec('DIST_CHI2', [chi2pdf(4.2,5.0), chi2cdf(4.2,5.0), chi2inv(0.95,5.0), m, v]);
[m, v] = fstat(5.0, 12.0);
pvec('DIST_F', [fpdf(2.1,5.0,12.0), fcdf(2.1,5.0,12.0), finv(0.95,5.0,12.0), m, v]);
[m, v] = gamstat(2.5, 1.5);
pvec('DIST_GAM', [gampdf(3.2,2.5,1.5), gamcdf(3.2,2.5,1.5), gaminv(0.8,2.5,1.5), m, v]);
[m, v] = betastat(2.0, 5.0);
pvec('DIST_BETA', [betapdf(0.4,2.0,5.0), betacdf(0.4,2.0,5.0), betainv(0.7,2.0,5.0), m, v]);
[m, v] = expstat(3.0);
pvec('DIST_EXP', [exppdf(2.0,3.0), expcdf(2.0,3.0), expinv(0.6,3.0), m, v]);
[m, v] = wblstat(2.0, 3.0);
pvec('DIST_WBL', [wblpdf(1.8,2.0,3.0), wblcdf(1.8,2.0,3.0), wblinv(0.75,2.0,3.0), m, v]);
[m, v] = lognstat(0.5, 0.8);
pvec('DIST_LOGN', [lognpdf(2.5,0.5,0.8), logncdf(2.5,0.5,0.8), logninv(0.6,0.5,0.8), m, v]);
[m, v] = unifstat(1.0, 5.0);
pvec('DIST_UNIF', [unifpdf(2.5,1.0,5.0), unifcdf(2.5,1.0,5.0), unifinv(0.4,1.0,5.0), m, v]);
[m, v] = raylstat(1.5);
pvec('DIST_RAYL', [raylpdf(2.0,1.5), raylcdf(2.0,1.5), raylinv(0.7,1.5), m, v]);
[m, v] = gevstat(0.2, 1.5, 0.5);
pvec('DIST_GEV', [gevpdf(1.2,0.2,1.5,0.5), gevcdf(1.2,0.2,1.5,0.5), gevinv(0.8,0.2,1.5,0.5), m, v]);
[m, v] = gpstat(0.25, 2.0, 1.0);
pvec('DIST_GP', [gppdf(2.5,0.25,2.0,1.0), gpcdf(2.5,0.25,2.0,1.0), gpinv(0.7,0.25,2.0,1.0), m, v]);
[m, v] = evstat(1.0, 2.0);
pvec('DIST_EV', [evpdf(0.5,1.0,2.0), evcdf(0.5,1.0,2.0), evinv(0.6,1.0,2.0), m, v]);
pd_r = makedist('Rician', 's', 1.5, 'sigma', 1.2);
pvec('DIST_RICE', [pdf(pd_r,2.2), cdf(pd_r,2.2), icdf(pd_r,0.6), mean(pd_r), var(pd_r)]);
[m, v] = ncx2stat(4.0, 3.0);
pvec('DIST_NCX2', [ncx2pdf(6.5,4.0,3.0), ncx2cdf(6.5,4.0,3.0), ncx2inv(0.75,4.0,3.0), m, v]);
pd_nk = makedist('Nakagami', 'mu', 2.5, 'omega', 3.0);
pvec('DIST_NAKA', [pdf(pd_nk,1.5), cdf(pd_nk,1.5), icdf(pd_nk,0.65), mean(pd_nk), var(pd_nk)]);

% 2. Discrete distributions
[m, v] = poisstat(4.5);
pvec('DIST_POISS', [poisspdf(3,4.5), poisscdf(3,4.5), poissinv(0.8,4.5), m, v]);
[m, v] = binostat(12, 0.35);
pvec('DIST_BINO', [binopdf(4,12,0.35), binocdf(4,12,0.35), binoinv(0.75,12,0.35), m, v]);
[m, v] = geostat(0.3);
pvec('DIST_GEO', [geopdf(2,0.3), geocdf(2,0.3), geoinv(0.8,0.3), m, v]);
[m, v] = unidstat(6);
pvec('DIST_UNID', [unidpdf(3,6), unidcdf(3,6), unidinv(0.55,6), m, v]);
[m, v] = nbinstat(3.0, 0.4);
pvec('DIST_NBIN', [nbinpdf(5,3.0,0.4), nbincdf(5,3.0,0.4), nbininv(0.7,3.0,0.4), m, v]);
[m, v] = hygestat(20, 7, 5);
pvec('DIST_HYGE', [hygepdf(2,20,7,5), hygecdf(2,20,7,5), hygeinv(0.8,20,7,5), m, v]);

% 3. Descriptive
dx = [1.2 2.5 2.5 4.0 5.3 7.1 8.9 12.4];
dy = [2.1 3.0 4.2 5.1 6.8 7.5 10.2 11.9];
Cxy = cov(dx, dy);
Rxy = corrcoef(dx, dy);
pvec('DESC_BASIC', [mean(dx), var(dx,0), var(dx,1), std(dx,0), skewness(dx,1), skewness(dx,0), ...
    kurtosis(dx,1), kurtosis(dx,0), Cxy(1,2), Rxy(1,2), quantile(dx,0.35), median(dx), iqr(dx)]);

% 4. Pairwise distances
px = [1 0 2; 2 3 1; 0 1 5; 4 2 0];
pvec('PDIST_EUC', pdist(px, 'euclidean'));
pvec('PDIST_L1', pdist(px, 'cityblock'));
pvec('PDIST_CHEB', pdist(px, 'chebychev'));
pvec('PDIST_MINK3', pdist(px, 'minkowski', 3));
pvec('PDIST_COS', pdist(px, 'cosine'));
pvec('PDIST_CORR', pdist(px, 'correlation'));
pvec('PDIST_SPEAR', pdist(px, 'spearman'));
pvec('PDIST_SEUC', pdist(px, 'seuclidean'));
pvec('PDIST_HAM', pdist(px, 'hamming'));
pvec('PDIST_JAC', pdist(px, 'jaccard'));
px_bin = [1 0 1 0; 1 1 0 1; 0 1 1 1; 1 1 0 0];
pvec('PDIST_JAC_BIN', pdist(px_bin, 'jaccard'));

mx_ref = [1 2; 2 1; 3 4; 4 3; 6 7];
my_qry = [2.5 3.0; 5.0 1.0];
pvec('MAHAL_QR', mahal(my_qry, mx_ref));

% 5. Linkage, cophenet, inconsistent, silhouette
d_euc = pdist(px, 'euclidean');
z_avg = linkage(d_euc, 'average');
z_ward = linkage(d_euc, 'ward');
pvec('LINK_AVG', z_avg);
pvec('LINK_WARD', z_ward);
pvec('COPHENET_R', cophenet(z_avg, d_euc));
pvec('INCONSIST_AVG', inconsistent(z_avg, 2));
pvec('SILHOUETTE', silhouette(px, [1;1;2;2], 'Euclidean'));

% 6. ANOVA & nonparametric
a1_m = [1.2 5.5 10.2; 2.4 6.1 11.8; 3.1 7.3 12.1];
[p_a1, tbl_a1] = anova1(a1_m, [], 'off');
pvec('ANOVA1', [p_a1, tbl_a1{2,5}, tbl_a1{2,2}, tbl_a1{3,2}]);
[p_kw, tbl_kw] = kruskalwallis(a1_m, [], 'off');
pvec('KRUSKAL', [p_kw, tbl_kw{2,5}]);
a2_m = [1.0 2.1 3.4; 2.5 3.8 5.1; 4.1 5.0 6.2; 5.2 6.7 8.0];
[p_a2, tbl_a2] = anova2(a2_m, 1, 'off');
pvec('ANOVA2', [p_a2(1), p_a2(2), tbl_a2{2,5}, tbl_a2{3,5}, tbl_a2{2,2}, tbl_a2{4,2}]);
[p_fr, tbl_fr] = friedman(a2_m, 1, 'off');
pvec('FRIEDMAN', [p_fr, tbl_fr{2,5}]);

% 7. Regression
rx = [ones(6,1), (1:6)', [2;1;4;3;6;5]];
ry = [5.2; 6.8; 11.1; 12.9; 17.3; 18.8];
[rb, rbint, ~, ~, rstat] = regress(ry, rx, 0.05);
pvec('REGRESS_BETA', rb);
pvec('REGRESS_BINT', rbint);
pvec('REGRESS_STATS', rstat);
lw = [1.0; 2.0; 1.5; 0.5; 2.5; 1.0];
[lxb, lstd, lmse] = lscov(rx, ry, lw);
pvec('LSCOV_W_X', lxb);
pvec('LSCOV_W_STD', lstd);
pvec('LSCOV_W_MSE', lmse);
rdg_x = [(1:6)', [2;1;4;3;6;5]];
pvec('RIDGE_UNSCALED', ridge(ry, rdg_x, [0 2.5], 0));
pvec('RIDGE_SCALED', ridge(ry, rdg_x, [0 2.5], 1));

gx = [-2.0; -1.5; -0.8; -0.2; 0.3; 0.9; 1.6; 2.2];
gy_bin = [0; 0; 0; 1; 0; 1; 1; 1];
gy_poi = [1; 1; 2; 3; 4; 7; 11; 18];
[gb, gdev] = glmfit(gx, gy_bin, 'binomial', 'link', 'logit');
pvec('GLMFIT_BIN_BETA', gb);
pvec('GLMFIT_BIN_DEV', gdev);
[gp_b, gp_dev] = glmfit(gx, gy_poi, 'poisson', 'link', 'log');
pvec('GLMFIT_POI_BETA', gp_b);
pvec('GLMFIT_POI_DEV', gp_dev);

% 8. PCA, nearcorr
xp = [1.0 2.1 0.5; 2.0 1.8 1.5; 3.5 4.0 2.0; 4.1 5.2 3.8; 6.2 5.9 4.7];
[pc, ~, plat, ptsq, pexp] = pca(xp);
pvec('PCA_COEFF', pc);
pvec('PCA_LATENT', plat);
pvec('PCA_TSQ', ptsq);
pvec('PCA_EXPLAINED', pexp);
bad_c = [1 0.9 0.9; 0.9 1 -0.9; 0.9 -0.9 1];
pvec('NEARCORR', nearcorr(bad_c, 'Method', 'projection', 'Tolerance', 1e-10));

% 9. Copulas & Multivariate Densities
cu = [0.25 0.4; 0.75 0.8];
cr2 = [1 0.6; 0.6 1];
pvec('BIVNORM_CDF', mvncdf([0.8 -0.4], [0 0], [1 0.65; 0.65 1]));
pvec('COP_GAUSS_PDF', copulapdf('Gaussian', cu, cr2));
pvec('COP_GAUSS_CDF', copulacdf('Gaussian', cu, cr2));
pvec('COP_T_PDF', copulapdf('t', cu, cr2, 5));
pvec('COP_CLAYTON_PDF', copulapdf('Clayton', cu, 2.5));
pvec('COP_CLAYTON_CDF', copulacdf('Clayton', cu, 2.5));
pvec('COP_FRANK_PDF', copulapdf('Frank', cu, 3.0));
pvec('COP_FRANK_CDF', copulacdf('Frank', cu, 3.0));
pvec('COP_GUMBEL_PDF', copulapdf('Gumbel', cu, 2.2));
pvec('COP_GUMBEL_CDF', copulacdf('Gumbel', cu, 2.2));

mvx = [0.5 -0.3; -1.2 0.7];
mvs = [2.0 0.7; 0.7 1.5];
pvec('MVNPDF', mvnpdf(mvx, [0 0], mvs));
pvec('MVTPDF', [mvtpdf(mvx(1,:), cr2, 5), mvtpdf(mvx(2,:), cr2, 10)]);
pvec('MNPDF', mnpdf([3 2 1], [0.5 0.3 0.2]));
[cor_r, cor_s] = corrcov(mvs);
pvec('CORRCOV_R', cor_r);
pvec('CORRCOV_S', cor_s);

% 10. Hypothesis tests
hx = [1.2 2.4 3.1 4.5 5.0 6.2 7.1 8.9];
hy = [3.0 4.1 5.8 6.2 7.5 8.9 9.4 11.2];
[~, p1, ci1, st1] = ttest(hx, 3.5);
pvec('TTEST1', [p1, ci1(1), ci1(2), st1.tstat, st1.sd]);
[~, p2e, ci2e, st2e] = ttest2(hx, hy, 'Vartype', 'equal');
pvec('TTEST2_EQ', [p2e, ci2e(1), ci2e(2), st2e.tstat, st2e.df]);
[~, p2u, ci2u, st2u] = ttest2(hx, hy, 'Vartype', 'unequal');
pvec('TTEST2_UNEQ', [p2u, ci2u(1), ci2u(2), st2u.tstat, st2u.df]);
[~, p2r, ci2r] = ttest2(hx, hy, 'Tail', 'right', 'Vartype', 'equal');
pvec('TTEST2_RIGHT', [p2r, ci2r(1)]);
[~, pz, ciz, zv] = ztest(hx, 3.5, 2.5);
pvec('ZTEST', [pz, ciz(1), ciz(2), zv]);
[~, pv1, civ1, stv1] = vartest(hx, 5.0);
pvec('VARTEST1', [pv1, civ1(1), civ1(2), stv1.chisqstat]);
[~, pv2, civ2, stv2] = vartest2(hx, hy);
pvec('VARTEST2', [pv2, civ2(1), civ2(2), stv2.fstat]);
ks_x = [-1.2 -0.8 -0.3 0.1 0.4 0.7 1.1 1.5 1.9 2.3];
[~, pks1, kss1] = kstest(ks_x);
pvec('KSTEST1', [pks1, kss1]);
[~, pks2, kss2] = kstest2(hx, hy);
pvec('KSTEST2', [pks2, kss2]);
[psg, ~, stsg] = signtest(hx, hy);
pvec('SIGNTEST', [psg, stsg.sign]);
[~, pfsh, stfsh] = fishertest([8 2; 1 9]);
pvec('FISHERTEST', [pfsh, stfsh.OddsRatio]);
vn_x = [1.1 2.5 3.2 10.0 10.2 10.5 20.0 24.0 29.0]';
vn_g = [1 1 1 2 2 2 3 3 3]';
[pvb, stvb] = vartestn(vn_x, vn_g, 'TestType', 'Bartlett', 'Display', 'off');
pvec('VARTESTN_BART', [pvb, stvb.chisqstat]);
[pvbf, stvbf] = vartestn(vn_x, vn_g, 'TestType', 'BrownForsythe', 'Display', 'off');
pvec('VARTESTN_BF', [pvbf, stvbf.fstat]);
rns_x = [1.0 2.0 1.5 8.0 9.0 2.0 1.0 7.0 8.5 9.5];
[~, prns, strns] = runstest(rns_x, 5.0, 'method', 'exact');
pvec('RUNSTEST', [prns, strns.nruns]);
[prke, ~, strke] = ranksum(hx, hy, 'method', 'exact');
pvec('RANKSUM_EX', [prke, strke.ranksum]);
[prka, ~, strka] = ranksum(hx, hy, 'method', 'approximate');
pvec('RANKSUM_AP', [prka, strka.ranksum, strka.zval]);
sr_x = [1.0 2.5 3.1 4.8 5.2 6.7 7.9 8.4 9.1 10.5];
[psre, ~, stsre] = signrank(sr_x, 4.0, 'method', 'exact');
pvec('SIGNRANK_EX', [psre, stsre.signedrank]);
[psra, ~, stsra] = signrank(sr_x, 4.0, 'method', 'approximate');
pvec('SIGNRANK_AP', [psra, stsra.signedrank, stsra.zval]);
[~, pab, stab] = ansaribradley(hx, hy);
pvec('ANSARI', [pab, stab.W]);
dw_r = [0.1; -0.1; 0.1; -0.1; 0.0];
dw_x = [ones(5,1) (1:5)'];
[pdwe, dwe] = dwtest(dw_r, dw_x, 'exact');
[pdwa, dwa] = dwtest(dw_r, dw_x, 'approximate');
pvec('DWTEST_EX', [dwe, pdwe]);
pvec('DWTEST_AP', [dwa, pdwa]);

% 11. Distribution fitters & likelihoods
pos_d = [0.8 1.1 1.4 1.7 2.0 2.3 2.6 3.0 3.5 4.1 4.8 5.9];
uni_d = [0.12 0.21 0.30 0.38 0.45 0.52 0.61 0.69 0.78 0.88];
nb_d = [0 1 2 2 3 4 5 7 9 12];
ev_d = [-2.1 -1.2 -0.5 0.1 0.6 1.0 1.4 1.9 2.3 2.8];
gp_d = [0.1 0.25 0.4 0.7 1.1 1.6 2.3 3.5 5.2 8.1 13.5 22.0];
[nm, ns, nmci, nsci] = normfit(pos_d);
pvec('FIT_NORM', [nm, ns, nmci(1), nmci(2), nsci(1), nsci(2)]);
[em, eci] = expfit(pos_d);
pvec('FIT_EXP', [em, eci(1), eci(2)]);
[pm, pci] = poissfit(nb_d);
pvec('FIT_POISS', [pm, pci(1), pci(2)]);
[ua, ub, uaci, ubci] = unifit(pos_d);
pvec('FIT_UNIF', [ua, ub, uaci(1), uaci(2), ubci(1), ubci(2)]);
[bm, bci] = binofit(7, 20);
pvec('FIT_BINO', [bm, bci(1), bci(2)]);
[rm, rci] = raylfit(pos_d);
pvec('FIT_RAYL', [rm, rci(1), rci(2)]);
pvec('FIT_GAM', gamfit(pos_d));
pvec('FIT_WBL', wblfit(pos_d));
pvec('FIT_BETA', betafit(uni_d));
pvec('FIT_NBIN', nbinfit(nb_d));
pvec('FIT_EV', evfit(ev_d));
pvec('FIT_GP', gpfit(gp_d));
pvec('FIT_GEV', gevfit(ev_d));
pvec('LIKELIHOODS', [normlike([2.5 1.5], pos_d), explike(2.5, pos_d), lognlike([0.8 0.5], pos_d), ...
    gamlike([2.5 1.2], pos_d), betalike([2.0 2.5], uni_d), wbllike([3.0 2.0], pos_d), ...
    evlike([1.0 1.5], ev_d), gevlike([0.1 1.5 0.5], ev_d), gplike([0.2 2.0], gp_d)]);

% 12. Robust, canonical, tiedrank, corr pvals, halton
xrf = (1:8)';
yrf = [3.1; 4.9; 7.2; 8.8; 11.1; 13.0; 14.9; 100.0];
pvec('ROBUSTFIT_BISQ', robustfit(xrf, yrf, 'bisquare'));
xcc = [1 2; 2 1; 3 5; 4 3; 5 7; 6 4; 7 8; 8 6];
ycc = [2 1; 4 3; 6 2; 8 5; 10 4; 12 7; 14 6; 16 9];
[cca, ~, ccr] = canoncorr(xcc, ycc);
pvec('CANONCORR_R', ccr);
pvec('CANONCORR_A', abs(cca));
[tro, tradj] = tiedrank([10 20 30 20 10 40]);
pvec('TIEDRANK', tro);
pvec('TIEDRANK_ADJ', tradj);
hs = haltonset(2, 'Skip', 1, 'Leap', 0);
pvec('HALTON', net(hs, 4));
fprintf('MATLAB_DONE=1\n');
"""


def run_matlab() -> dict[str, list[float]]:
    if shutil.which("matlab") is None:
        raise RuntimeError("matlab executable not found in PATH")
    proc = subprocess.run(
        ["matlab", "-batch", MATLAB_SCRIPT],
        capture_output=True,
        text=True,
    )
    kv = parse_kv(proc.stdout)
    # Windows MATLAB R2025b can hit a shutdown crash after emitting all stdout.
    if kv.get("MATLAB_DONE") != [1.0]:
        raise RuntimeError(
            f"MATLAB script did not reach MATLAB_DONE=1 (rc={proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
        )
    kv.pop("MATLAB_DONE", None)
    return kv


def compare_suite(
    name: str,
    keel_kv: dict[str, list[float]],
    ref_kv: dict[str, tuple[list[float], float]],
    abs_keys: set[str] | None = None,
) -> tuple[int, int]:
    passed = 0
    failed = 0
    abs_keys = abs_keys or set()
    for key, (expected, tol) in sorted(ref_kv.items()):
        if key not in keel_kv:
            print(f"[{name}] FAIL {key}: missing from keel output")
            failed += 1
            continue
        got = keel_kv[key]
        if len(got) != len(expected):
            print(f"[{name}] FAIL {key}: len {len(got)} != {len(expected)}")
            failed += 1
            continue
        ok = True
        max_err = 0.0
        for i, (g, e) in enumerate(zip(got, expected)):
            if key in abs_keys:
                g, e = abs(g), abs(e)
            if math.isnan(e) and math.isnan(g):
                continue
            err = abs(g - e) / max(1.0, abs(e))
            if err > max_err:
                max_err = err
            if math.isnan(g) or math.isnan(e) or err > tol:
                print(
                    f"[{name}] FAIL {key}[{i}]: got={g:.17g} expected={e:.17g} "
                    f"rel_err={err:.3e} tol={tol:.1e}"
                )
                ok = False
        if ok:
            passed += 1
        else:
            failed += 1
    print(f"[{name}] {passed}/{passed + failed} checks PASSED")
    return passed, failed


def main() -> int:
    exe = find_exe()
    keel_kv = run_keel(exe)
    print(f"Loaded {len(keel_kv)} reference vectors from {exe}")

    sp_ref = compute_scipy()
    sp_pass, sp_fail = compare_suite("SciPy/NumPy", keel_kv, sp_ref)

    ml_raw = run_matlab()
    ml_tols: dict[str, float] = {
        "DIST_RICE": 1e-8,
        "DIST_NCX2": 1e-8,
        "BIVNORM_CDF": 1e-8,
        "COP_GAUSS_CDF": 1e-7,
        "KSTEST1": 1e-6,
        "KSTEST2": 1e-6,
        "DWTEST_EX": 1e-5,
        "DWTEST_AP": 1e-6,
        "GLMFIT_BIN_BETA": 1e-6,
        "GLMFIT_BIN_DEV": 1e-6,
        "GLMFIT_POI_BETA": 1e-6,
        "GLMFIT_POI_DEV": 1e-6,
        "NEARCORR": 1e-6,
        "FIT_GAM": 1e-5,
        "FIT_WBL": 1e-5,
        "FIT_BETA": 1e-5,
        "FIT_NBIN": 1e-5,
        "FIT_EV": 1e-5,
        "FIT_GP": 1e-4,
        "FIT_GEV": 1e-4,
        "ROBUSTFIT_BISQ": 1e-3,
        "CANONCORR_A": 1e-8,
        "CANONCORR_R": 1e-10,
    }
    ml_ref = {k: (v, ml_tols.get(k, 1e-10)) for k, v in ml_raw.items()}
    ml_pass, ml_fail = compare_suite(
        "MATLAB R2025b",
        keel_kv,
        ml_ref,
        abs_keys={"CANONCORR_A"},
    )

    total_pass = sp_pass + ml_pass
    total_fail = sp_fail + ml_fail
    print(f"TOTAL: {total_pass}/{total_pass + total_fail} PASSED ({total_fail} failed)")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
