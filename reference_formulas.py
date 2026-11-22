#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Independent reference values for validating the ``ThermalCompton`` generator.

The functions below evaluate the physics simulated by ``main.ThermalCompton``
deterministically (closed formulas and Gauss-Legendre quadrature), without
calling any of its sampling routines. They are used by the validation scripts
(validate_*.py) and by the test suite (tests/), and can also serve for quick
analytic estimates, e.g. of the total collision rate of a section:

    R = (I/e) * rho(T) * sigma_T * L * mean_cross_section_ratio(Eb, T).

Notation
--------
k      : thermal photon energy in the lab, x = k/kT
a      : angle between the photon direction and the electron velocity (lab)
k*     : photon energy in the electron rest frame (ERF), k* = gamma*k*(1 - beta*cos a)
eps    : k*/(m_e c^2)
theta* : scattering angle in the ERF (between incoming and outgoing photon)
"""

import math

import numpy as np

from main import GeV_per_eV, kB_eV, m_e_GeV, zeta3

# Riemann zeta(5)
ZETA5 = 1.0369277551433699

# Moments of the Planck photon-number spectrum, x = k/kT:
#   <x>   = pi^4 / (30 zeta(3))    = 2.701178...
#   <x^2> = 4! zeta(5) / (2 zeta(3)) = 10.35153...
PLANCK_MEAN_X = math.pi ** 4 / (30.0 * zeta3)
PLANCK_MEAN_X2 = 12.0 * ZETA5 / zeta3

# Upper integration limit in x = k/kT (the neglected tail is < 1e-22)
X_MAX = 60.0


# =============================================================================
# Basic kinematics and quadrature
# =============================================================================
def gamma_beta(E_GeV):
    """Lorentz factor gamma and velocity beta of an electron of total energy E [GeV]."""
    E = float(E_GeV)
    p = math.sqrt(E * E - m_e_GeV * m_e_GeV)
    return E / m_e_GeV, p / E


def gauss_legendre(a, b, n):
    """Gauss-Legendre nodes and weights for integrating over [a, b] with n points."""
    t, w = np.polynomial.legendre.leggauss(int(n))
    half = 0.5 * (b - a)
    return half * t + 0.5 * (b + a), half * w


def _bin_integrals(func, edges, n_gauss=16):
    """Integral of func over each bin [edges[i], edges[i+1]] (Gauss-Legendre per bin)."""
    edges = np.asarray(edges, dtype=float)
    t, w = np.polynomial.legendre.leggauss(n_gauss)
    lo = edges[:-1, None]
    hi = edges[1:, None]
    half = 0.5 * (hi - lo)
    nodes = half * t + 0.5 * (hi + lo)
    return (half * w * func(nodes)).sum(axis=1)


# =============================================================================
# Planck photon-number spectrum
# =============================================================================
def planck_pdf_x(x):
    """
    Normalized photon-number Planck pdf in x = E/kT:
        P(x) = x^2 / (exp(x) - 1) / (2 zeta(3)),   x >= 0.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    pos = x > 0.0
    out[pos] = x[pos] ** 2 / np.expm1(x[pos]) / (2.0 * zeta3)
    return out


def planck_bin_probabilities(edges_x, n_gauss=16):
    """Probability for x = E/kT to fall in each bin [edges_x[i], edges_x[i+1]]."""
    return _bin_integrals(planck_pdf_x, edges_x, n_gauss)


# =============================================================================
# Thomson and Klein-Nishina cross sections (ERF)
# =============================================================================
def thomson_cos_cdf(c):
    """
    CDF of c = cos(theta) for dsigma_T/dOmega ∝ 1 + cos^2(theta):
        F(c) = (c + 1)(c^2 - c + 4) / 8,   -1 <= c <= 1.
    """
    c = np.asarray(c, dtype=float)
    return (c + 1.0) * (c * c - c + 4.0) / 8.0


def kn_dsigma_dcos_over_sigmaT(c, eps):
    """
    Klein-Nishina differential cross section per unit cos(theta*), divided by sigma_T:
        (1/sigma_T) dsigma_KN/dcos = (3/8) x^2 [x + 1/x - sin^2(theta*)],
        x = k*_out/k* = 1 / (1 + eps (1 - cos theta*)).
    Integrates over [-1, 1] to kn_total_over_thomson(eps); for eps -> 0 it
    becomes the Thomson shape (3/8)(1 + cos^2).
    """
    c = np.asarray(c, dtype=float)
    x = 1.0 / (1.0 + eps * (1.0 - c))
    return 0.375 * x * x * (x + 1.0 / x - (1.0 - c * c))


def kn_total_over_thomson(eps):
    """
    Total Klein-Nishina cross section divided by sigma_T, for eps = k*/(m_e c^2):

      S(eps) = 3/4 { (1+eps)/eps^3 [ 2 eps (1+eps)/(1+2 eps) - ln(1+2 eps) ]
                     + ln(1+2 eps)/(2 eps) - (1+3 eps)/(1+2 eps)^2 }

    The closed form loses precision for small eps (cancellation), so the
    low-energy series 1 - 2 eps + 26/5 eps^2 - 133/10 eps^3 + ... (to eps^7) is
    used for eps < 1e-2. Relative accuracy ~1e-12 for all eps >= 0.
    """
    eps_arr = np.atleast_1d(np.asarray(eps, dtype=float))
    out = np.empty_like(eps_arr)

    small = eps_arr < 1e-2
    e = eps_arr[small]
    out[small] = 1.0 + e * (-2.0 + e * (26.0 / 5.0 + e * (-133.0 / 10.0 + e * (
        1144.0 / 35.0 + e * (-544.0 / 7.0 + e * (3784.0 / 21.0 + e * (-6148.0 / 15.0)))))))

    e = eps_arr[~small]
    log_term = np.log1p(2.0 * e)
    out[~small] = 0.75 * (
        (1.0 + e) / e ** 3 * (2.0 * e * (1.0 + e) / (1.0 + 2.0 * e) - log_term)
        + log_term / (2.0 * e)
        - (1.0 + 3.0 * e) / (1.0 + 2.0 * e) ** 2
    )

    if np.ndim(eps) == 0:
        return float(out[0])
    return out.reshape(np.shape(eps))


def kn_cos_bin_probabilities(edges_c, eps, n_gauss=16):
    """
    Probability for an accepted (Klein-Nishina) scattering at fixed eps to have
    cos(theta*) in each bin [edges_c[i], edges_c[i+1]]. The bins need not cover
    [-1, 1]; for a full cover the probabilities sum to 1.
    """
    def pdf(c):
        return kn_dsigma_dcos_over_sigmaT(c, eps)
    return _bin_integrals(pdf, edges_c, n_gauss) / kn_total_over_thomson(eps)


def compton_edge_GeV(E_e_GeV, k_GeV):
    """
    Largest possible lab energy of the scattered photon, reached for a head-on
    collision and back-scattering in the ERF, for an electron of energy E_e and
    a photon of energy k (both GeV):

        w_max = k (E + p) / (m_e^2/(E + p) + 2k)   [-> 4 gamma^2 k / (1 + 4 gamma k/m_e)]

    (E - p is written as m_e^2/(E + p) to avoid cancellation.)
    """
    E = np.asarray(E_e_GeV, dtype=float)
    k = np.asarray(k_GeV, dtype=float)
    E_plus_p = E + np.sqrt(E * E - m_e_GeV * m_e_GeV)
    return k * E_plus_p / (m_e_GeV * m_e_GeV / E_plus_p + 2.0 * k)


# =============================================================================
# Collision averages over the blackbody photon gas
# =============================================================================
def _collision_grid(Eb_GeV, T_K, flux_weighting, n_x, n_a):
    """
    Quadrature grid over x = k/kT and c = cos(a) for an electron of energy Eb
    along +z. Returns (gamma, beta, c nodes, weights W[x, c], k*[x, c] in GeV),
    where W includes the Planck pdf and the angular weight A(c):
        A(c) = (1 - beta c)/2  (flux factor)   or   1/2  (isotropic).
    """
    gamma, beta = gamma_beta(Eb_GeV)
    kT_GeV = kB_eV * float(T_K) * GeV_per_eV

    x, wx = gauss_legendre(0.0, X_MAX, n_x)
    c, wc = gauss_legendre(-1.0, 1.0, n_a)

    w_x = wx * planck_pdf_x(x)
    w_c = wc * ((1.0 - beta * c) / 2.0 if flux_weighting else np.full_like(c, 0.5))

    k_star = gamma * kT_GeV * x[:, None] * (1.0 - beta * c[None, :])
    return gamma, beta, c, w_x[:, None] * w_c[None, :], k_star


def mean_cross_section_ratio(Eb_GeV, T_K, flux_weighting=True, n_x=200, n_a=200):
    """
    Collision-averaged Klein-Nishina/Thomson cross-section ratio for an
    on-momentum electron of energy Eb [GeV] in a blackbody photon gas at T [K]:

        <S> = int dx P(x) int dc A(c) S(eps),   eps = gamma x kT (1 - beta c) / m_e,

    with S = kn_total_over_thomson. This is the expected acceptance fraction of
    the generator's trials, i.e. the collision rate of a section of length L is
        R = (I/e) * rho(T) * sigma_T * L * <S>.
    flux_weighting selects the angular weight A(c), as in ThermalCompton.
    """
    _, _, _, W, k_star = _collision_grid(Eb_GeV, T_K, flux_weighting, n_x, n_a)
    return float((W * kn_total_over_thomson(k_star / m_e_GeV)).sum())


def mean_cos_alpha_of_collisions(Eb_GeV, T_K, flux_weighting=True, n_x=200, n_a=200):
    """
    Mean cos(a) of the thermal photons that actually collide (accepted trials):
    the angular weight A(c) times the Klein-Nishina total cross section S(eps).
    With the flux factor this is -beta/3 in the Thomson limit, shifted by
    +O(gamma kT/m_e) because head-on photons have larger k* and hence a smaller
    Klein-Nishina cross section; for isotropic sampling it is 0 + O(gamma kT/m_e).
    """
    _, _, c, W, k_star = _collision_grid(Eb_GeV, T_K, flux_weighting, n_x, n_a)
    WS = W * kn_total_over_thomson(k_star / m_e_GeV)
    return float((WS * c[None, :]).sum() / WS.sum())


def mean_scattered_photon_energy_GeV(Eb_GeV, T_K, flux_weighting=True, n_x=96, n_a=96, n_t=96):
    """
    Mean lab energy [GeV] of the scattered photon per collision, for an
    on-momentum electron of energy Eb along +z, by 3D quadrature over
    (x, cos a, cos theta*) with the Klein-Nishina differential cross section.

    In the ERF the incoming photon makes the angle a* with the boost axis,
    cos a* = (cos a - beta) / (1 - beta cos a). Averaging over the azimuth of
    the scattered photon around the incoming direction, its lab energy is
        E_out = gamma * x_C * k* * (1 + beta cos(theta*) cos(a*)),
    with x_C = 1/(1 + eps (1 - cos theta*)).

    Returns (mean E_out [GeV], <sigma_KN/sigma_T>); the second value is an
    independent (differential cross section based) evaluation of
    mean_cross_section_ratio.
    """
    gamma, beta, c, W2, k_star = _collision_grid(Eb_GeV, T_K, flux_weighting, n_x, n_a)
    ct, wt = gauss_legendre(-1.0, 1.0, n_t)

    eps = (k_star / m_e_GeV)[:, :, None]                  # (n_x, n_a, 1)
    x_c = 1.0 / (1.0 + eps * (1.0 - ct))                  # (n_x, n_a, n_t)
    dsig = kn_dsigma_dcos_over_sigmaT(ct, eps)            # (n_x, n_a, n_t)
    cos_a_star = (c - beta) / (1.0 - beta * c)            # (n_a,)

    E_out = gamma * x_c * k_star[:, :, None] * (1.0 + beta * ct * cos_a_star[None, :, None])
    W = W2[:, :, None] * wt * dsig

    S = W.sum()
    return float((W * E_out).sum() / S), float(S)


def thomson_limit_mean_photon_energy_GeV(Eb_GeV, T_K, flux_weighting=True):
    """
    Mean scattered photon energy in the Thomson limit (k* << m_e), per collision:
        flux factor : <E_out> = (1 + beta^2/3) gamma^2 <k>   (-> 4/3 gamma^2 <k>)
        isotropic   : <E_out> = gamma^2 <k>
    with <k> = 2.701 kT. Klein-Nishina corrections lower it by O(gamma kT / m_e).
    """
    gamma, beta = gamma_beta(Eb_GeV)
    mean_k_GeV = PLANCK_MEAN_X * kB_eV * float(T_K) * GeV_per_eV
    factor = (1.0 + beta * beta / 3.0) if flux_weighting else 1.0
    return factor * gamma * gamma * mean_k_GeV


# =============================================================================
# Statistical test helpers for the validation scripts
# =============================================================================
def chi2_vs_expected(observed, expected, min_expected=5.0):
    """
    Pearson chi^2 of histogram counts against expected counts, using only bins
    with expected >= min_expected. Returns (chi2, ndf, z), where
    z = (chi2 - ndf)/sqrt(2 ndf) is approximately a standard normal variable
    when the model is correct (|z| < 5 is used as the pass criterion).
    """
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    use = expected >= min_expected
    chi2 = float((((observed[use] - expected[use]) ** 2) / expected[use]).sum())
    ndf = max(int(use.sum()) - 1, 1)
    return chi2, ndf, (chi2 - ndf) / math.sqrt(2.0 * ndf)


def print_checks(checks, title="Summary"):
    """
    Print a PASS/FAIL table. ``checks`` is a list of (description, passed, detail)
    tuples. Returns True if all checks passed.
    """
    print(f"--- {title} ---")
    all_ok = True
    for description, passed, detail in checks:
        all_ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {description}: {detail}")
    print(f"  => {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    return all_ok
