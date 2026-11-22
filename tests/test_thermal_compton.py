# -*- coding: utf-8 -*-
"""
Automated tests for the ThermalCompton generator (main.py).

Run from the repository root with:
    python -m pytest -q

The statistical tests use fixed seeds and 5-sigma tolerances; the whole suite
runs in well under a minute.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the repository root importable (main.py, reference_formulas.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import reference_formulas as rf  # noqa: E402
from main import ThermalCompton, e_SI, m_e_GeV, sigma_T_m2  # noqa: E402

EPS = np.finfo(float).eps


def make_tc(seed=1, **kwargs):
    params = dict(Eb_GeV=18.0, I_A=0.227, T_K=300.0, rng=np.random.default_rng(seed))
    params.update(kwargs)
    return ThermalCompton(**params)


def pull(sample, expected):
    """(sample mean - expected) / standard error of the mean."""
    sample = np.asarray(sample, dtype=float)
    return (sample.mean() - expected) / (sample.std(ddof=1) / math.sqrt(sample.size))


# =============================================================================
# Constants, densities, coordinate conversions
# =============================================================================
def test_photon_density_and_reference_momentum():
    # 16 pi zeta(3) (kT/hc)^3 at 300 K, evaluated with sympy at 15 digits
    assert ThermalCompton._photon_density_blackbody(300.0) == pytest.approx(5.47744842915490e14, rel=1e-12)
    # rho ∝ T^3
    ratio = ThermalCompton._photon_density_blackbody(600.0) / ThermalCompton._photon_density_blackbody(300.0)
    assert ratio == pytest.approx(8.0, rel=1e-14)
    tc = make_tc()
    assert tc.p0_GeV == pytest.approx(math.sqrt(18.0 ** 2 - m_e_GeV ** 2), rel=1e-15)
    assert tc.rho_m3 == ThermalCompton._photon_density_blackbody(300.0)


def test_sixd_four_vector_roundtrip():
    p0 = ThermalCompton._p0_from_Eb(18.0)
    for xp, yp, dp in [(0.0, 0.0, 0.0), (1e-3, -2e-4, 5e-3), (-3e-2, 4e-2, -0.2)]:
        E, p = ThermalCompton._sixd_to_four_electron(1.0, 2.0, 3.0, xp, yp, dp, p0)
        assert E == pytest.approx(math.sqrt(p @ p + m_e_GeV ** 2), rel=1e-15)
        xp2, yp2, dp2 = ThermalCompton._four_to_sixd_slopes(p[0], p[1], p[2], p0)
        assert (xp2, yp2) == pytest.approx((xp, yp), abs=1e-15)
        assert dp2 == pytest.approx(dp, abs=1e-14)
    # Backward-moving particles are not representable
    assert ThermalCompton._four_to_sixd_slopes(0.0, 0.0, -1.0, p0) is None


# =============================================================================
# Samplers
# =============================================================================
def test_planck_sampler_moments():
    tc = make_tc(seed=11)
    x = tc.photon_energy_sampler_eV.sample_eV(1_000_000) / tc.photon_energy_sampler_eV.kT_eV
    assert x.min() > 0.0
    assert abs(pull(x, rf.PLANCK_MEAN_X)) < 5.0
    assert abs(pull(x * x, rf.PLANCK_MEAN_X2)) < 5.0


def test_planck_sampler_scalar_and_vector_paths_agree():
    # __call__ is a fast path that must reproduce sample_eV(1) exactly
    s1 = ThermalCompton.BlackbodyPhotonEnergySampler(300.0, rng=np.random.default_rng(5))
    s2 = ThermalCompton.BlackbodyPhotonEnergySampler(300.0, rng=np.random.default_rng(5))
    a = [float(s1.sample_eV(1)[0]) for _ in range(2000)]
    b = [s2() for _ in range(2000)]
    assert a == b


def test_thomson_angle_moments():
    rng = np.random.default_rng(21)
    samples = np.array([ThermalCompton._sample_thomson_angles(rng) for _ in range(200_000)])
    c, s, phi = samples.T
    assert np.all(np.abs(c) <= 1.0) and np.all((phi >= 0.0) & (phi < 2.0 * np.pi))
    assert np.max(np.abs(c * c + s * s - 1.0)) < 1e-12
    assert abs(pull(c, 0.0)) < 5.0
    assert abs(pull(c * c, 0.4)) < 5.0          # <cos^2> = 2/5 for 1 + cos^2
    assert abs(pull(phi, np.pi)) < 5.0


def test_flux_weighted_direction_distribution():
    rng = np.random.default_rng(31)
    # Off-axis velocity to exercise the basis construction
    beta_hat = np.array([0.3, -0.2, 0.9])
    beta_hat /= np.linalg.norm(beta_hat)
    for beta in (0.999999999, 0.6):
        beta_vec = beta * beta_hat
        n = np.array([ThermalCompton._sample_flux_weighted_direction(rng, beta_vec) for _ in range(100_000)])
        assert np.max(np.abs(np.linalg.norm(n, axis=1) - 1.0)) < 1e-14
        cos_a = n @ beta_hat
        assert abs(pull(cos_a, -beta / 3.0)) < 5.0              # pdf (1 - beta c)/2
        assert abs(pull(cos_a ** 2, 1.0 / 3.0)) < 5.0           # <c^2> = 1/3 for any beta
        # Azimuth around beta is uniform: the transverse part averages to zero
        transverse = n - np.outer(cos_a, beta_hat)
        assert np.all(np.abs(transverse.mean(axis=0)) < 5.0 / math.sqrt(len(n)))
    # beta = 0: isotropic
    n = np.array([ThermalCompton._sample_flux_weighted_direction(rng, np.zeros(3)) for _ in range(50_000)])
    assert np.all(np.abs(n.mean(axis=0)) < 5.0 / math.sqrt(3.0 * len(n)))


# =============================================================================
# Cross sections and transformations
# =============================================================================
def test_kn_total_reference_values():
    # 50-digit mpmath evaluations of the closed-form total cross section ratio
    reference = {1e-3: 0.99800518673260818, 1e-2: 0.98050701926189595,
                 0.0195696: 0.96275714298903574, 0.1: 0.84133814963142989, 1.0: 0.43072784191504326}
    for eps, value in reference.items():
        assert rf.kn_total_over_thomson(eps) == pytest.approx(value, rel=1e-11)
    # differential cross section integrates to the total one
    c, w = rf.gauss_legendre(-1.0, 1.0, 64)
    for eps in (1e-6, 0.02, 3.0):
        integral = np.sum(w * rf.kn_dsigma_dcos_over_sigmaT(c, eps))
        assert integral == pytest.approx(rf.kn_total_over_thomson(eps), rel=1e-12)


def test_kn_ratio_is_a_probability_with_thomson_limit():
    for k_star in (1e-12, 1e-6, 1e-4, 1e-2):
        for c in np.linspace(-1.0, 1.0, 41):
            R, x = ThermalCompton._compton_over_thomson_ratio(k_star, c)
            assert 0.0 <= R <= 1.0
            assert x == pytest.approx(1.0 / (1.0 + k_star / m_e_GeV * (1.0 - c)), rel=1e-15)
    R, x = ThermalCompton._compton_over_thomson_ratio(1e-15, -0.3)
    assert R == pytest.approx(1.0, abs=1e-10) and x == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("kstar_keV", [10.0, 300.0])
def test_kn_acceptance_fraction_matches_total_cross_section(kstar_keV):
    rng = np.random.default_rng(41)
    k_star = kstar_keV * 1e-6
    N = 150_000
    accepted = 0
    for _ in range(N):
        c, _s, _phi = ThermalCompton._sample_thomson_angles(rng)
        R, _x = ThermalCompton._compton_over_thomson_ratio(k_star, c)
        accepted += rng.random() < R
    S = rf.kn_total_over_thomson(k_star / m_e_GeV)
    assert abs(accepted / N - S) < 5.0 * math.sqrt(S * (1.0 - S) / N)


def test_lorentz_boost_roundtrip_and_rest_frame():
    E, p = ThermalCompton._sixd_to_four_electron(0, 0, 0, 2e-4, -1e-4, 3e-3, ThermalCompton._p0_from_Eb(18.0))
    beta, gamma = p / E, E / m_e_GeV
    # The electron is at rest in its rest frame. In double precision a lab
    # four-vector with Lorentz factor gamma is on its mass shell only to
    # ~eps*gamma^2 (E^2 - p^2 cancels), which bounds this check.
    floor = 10.0 * EPS * gamma ** 2
    E_star, p_star = ThermalCompton._lorentz_boost(E, p, beta, gamma)
    assert E_star == pytest.approx(m_e_GeV, rel=floor)
    assert np.max(np.abs(p_star)) < floor * m_e_GeV
    # A photon is boosted forth and back unchanged and stays massless
    k = 1e-10 * np.array([0.3, -0.4, -0.866])
    Ek = float(np.linalg.norm(k))
    Ek_star, k_star = ThermalCompton._lorentz_boost(Ek, k, beta, gamma)
    assert Ek_star == pytest.approx(float(np.linalg.norm(k_star)), rel=1e-12)
    Ek_back, k_back = ThermalCompton._lorentz_boost(Ek_star, k_star, -beta, gamma)
    assert Ek_back == pytest.approx(Ek, rel=1e-6) and np.allclose(k_back, k, rtol=1e-6, atol=1e-18)
    # Without an explicit gamma the boost is consistent with 1/sqrt(1 - beta^2)
    b = np.array([0.1, 0.2, 0.3])
    E1, p1 = ThermalCompton._lorentz_boost(2.0, np.array([0.5, -0.3, 1.0]), b)
    E2, p2 = ThermalCompton._lorentz_boost(2.0, np.array([0.5, -0.3, 1.0]), b, 1.0 / math.sqrt(1.0 - b @ b))
    assert E1 == E2 and np.array_equal(p1, p2)


def test_rotation_matrix_maps_vector_to_z():
    rng = np.random.default_rng(51)
    vectors = list(rng.normal(size=(200, 3)))
    vectors += [np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, -3.0]),
                np.array([1e-9, 0.0, 1.0]), np.array([0.0, 1e-9, -1.0]), np.array([1.0, 0.0, 0.0])]
    z = np.array([0.0, 0.0, 1.0])
    for v in vectors:
        R = ThermalCompton._rotation_matrix_from_vec_to_z(v)
        assert np.allclose(R @ (v / np.linalg.norm(v)), z, atol=1e-12)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)
    assert np.array_equal(ThermalCompton._rotation_matrix_from_vec_to_z(np.zeros(3)), np.eye(3))


# =============================================================================
# Complete scatterings
# =============================================================================
@pytest.mark.parametrize("Eb, xp, yp, dp", [(18.0, 0.0, 0.0, 0.0), (275.0, 2e-3, -1e-3, -5e-3)])
def test_single_scattering_kinematics(Eb, xp, yp, dp):
    tc = make_tc(seed=61, Eb_GeV=Eb)
    Pe = tc._sixd_to_four_electron(0, 0, 0, xp, yp, dp, tc.p0_GeV)
    E_e, p_e = Pe
    n_acc = 0
    for _ in range(20_000):
        Pg = tc._sample_thermal_photon(Pe, tc.photon_energy_sampler_eV, tc.rng)
        accepted, e_out, g_out = tc._scatter_photon(Pe, Pg, tc.rng)
        if not accepted:
            assert e_out is None and g_out is None
            continue
        n_acc += 1
        (E1, p1), (E2, p2) = e_out, g_out
        assert abs(E2 * E2 - p2 @ p2) < 1e-8 * E2 * E2                      # photon massless
        assert abs(E1 * E1 - p1 @ p1 - m_e_GeV ** 2) < 100 * EPS * E1 * E1   # electron on shell
        assert abs(E_e + Pg[0] - E1 - E2) < 1e-14 * E_e                      # energy conservation
        assert np.max(np.abs(p_e + Pg[1] - p1 - p2)) < 1e-14 * E_e           # momentum conservation
        assert E2 <= rf.compton_edge_GeV(E_e, Pg[0]) * (1.0 + 1e-10)         # Compton edge
        assert abs(p1[0] / p1[2] - xp) < 1e-5 and abs(p1[1] / p1[2] - yp) < 1e-5  # tiny deflection
    assert n_acc > 15_000


def test_mean_photon_energy_flux_factor_in_thomson_limit():
    """
    In the Thomson limit the mean scattered photon energy is (4/3) gamma^2 <k>
    with the flux factor (textbook inverse-Compton result) and gamma^2 <k> for
    isotropic photon directions (Burkhardt 1993 sampling). At 1 GeV the
    Klein-Nishina corrections are ~0.2 %.
    """
    Eb, T = 1.0, 300.0
    gamma, _ = rf.gamma_beta(Eb)
    gamma2_k = gamma ** 2 * rf.PLANCK_MEAN_X * 8.617333262145e-5 * T * 1e-9
    for flux, factor in ((True, 4.0 / 3.0), (False, 1.0)):
        tc = make_tc(seed=71, Eb_GeV=Eb, T_K=T, flux_weighting=flux)
        Pe = tc._sixd_to_four_electron(0, 0, 0, 0, 0, 0, tc.p0_GeV)
        Eg = []
        for _ in range(60_000):
            accepted, _e, g_out = tc._compton_scatter_one_trial(Pe, tc.photon_energy_sampler_eV, tc.rng,
                                                                flux_weighted=flux)
            if accepted:
                Eg.append(g_out[0])
        Eg = np.array(Eg)
        assert Eg.mean() / gamma2_k == pytest.approx(factor, rel=0.02)
        reference, _ = rf.mean_scattered_photon_energy_GeV(Eb, T, flux_weighting=flux)
        assert abs(pull(Eg, reference)) < 5.0


# =============================================================================
# Section events, weights and bookkeeping
# =============================================================================
def test_section_rate_matches_exact_expectation():
    tc = make_tc(seed=81, I_A=2.5, n_trials_per_macro=100)
    n_macro, L = 300, 1.5
    events = tc.generate_section_events_6d([(0.0,) * 6] * n_macro, L_m=L, dp_threshold=0.0)
    w = np.array([ev["w_Hz"] for ev in events])
    R_ref = (tc.I_A / e_SI) * tc.rho_m3 * sigma_T_m2 * L * rf.mean_cross_section_ratio(18.0, 300.0)
    assert abs(w.sum() - R_ref) < 5.0 * math.sqrt(n_macro * w.var(ddof=1))


def test_rate_is_exactly_linear_in_current_and_length():
    macros = [(0.0,) * 6] * 50

    def rate(I_A, L_m):
        tc = make_tc(seed=91, I_A=I_A, n_trials_per_macro=40)
        return sum(ev["w_Hz"] for ev in tc.generate_section_events_6d(macros, L_m=L_m, dp_threshold=0.0))

    R = rate(1.0, 1.0)
    assert rate(3.0, 1.0) == pytest.approx(3.0 * R, rel=1e-12)
    assert rate(1.0, 2.5) == pytest.approx(2.5 * R, rel=1e-12)
    assert rate(1.0, 0.0) == 0.0


def test_representative_events_and_weights():
    tc = make_tc(seed=101, n_trials_per_macro=30)
    n_macro, L, thr = 120, 2.0, 5e-3
    macros = [(1e-4, -2e-4, 3e-3, 1e-5, -2e-5, 1e-4)] * n_macro
    events = tc.generate_section_events_6d(macros, L_m=L, section_start_s_m=10.0, dp_threshold=thr)
    assert len(events) == n_macro  # exactly one event per macro
    w_trial = (tc.I_A / (e_SI * n_macro)) * tc.rho_m3 * sigma_T_m2 * (L / 30)
    for ev in events:
        assert 10.0 <= ev["s_m"] <= 12.0
        n_pass = ev["w_Hz"] / w_trial
        assert n_pass == pytest.approx(round(n_pass), abs=1e-9) and 0 <= round(n_pass) <= 30
        x, y, z, xp, yp, dp = ev["e6d_out"]
        assert (x, y, z) == (1e-4, -2e-4, 3e-3)       # interaction at the macro position
        if ev["w_Hz"] > 0.0:
            assert abs(dp) >= thr                     # weighted events are tail events
        assert ev["photon_E_GeV"] > 0.0
        # photon dp is relative to the electron reference momentum
        assert ev["g6d_out"][5] == pytest.approx(ev["photon_E_GeV"] / tc.p0_GeV - 1.0, rel=1e-12)


def test_unreachable_threshold_gives_zero_weight_events():
    # Regression test: this used to raise AttributeError ('max_tries')
    tc = make_tc(seed=111, n_trials_per_macro=10)
    events = tc.generate_section_events_6d([(0.0,) * 6] * 20, L_m=1.0, dp_threshold=0.5)
    assert len(events) == 20
    assert all(ev["w_Hz"] == 0.0 for ev in events)
    assert all(abs(ev["e6d_out"][5]) < 0.5 for ev in events)


def test_tail_rate_estimate_is_unbiased():
    # Rate of collisions with |dp| >= thr, estimated (a) with the threshold and
    # (b) from a full (dp_threshold = 0) run; both are unbiased estimators.
    thr, n_macro = 8e-3, 400
    macros = [(0.0,) * 6] * n_macro

    def estimate(seed, threshold):
        tc = make_tc(seed=seed, n_trials_per_macro=60)
        events = tc.generate_section_events_6d(macros, L_m=1.0, dp_threshold=threshold)
        w = np.array([ev["w_Hz"] * (abs(ev["e6d_out"][5]) >= thr) for ev in events])
        return w.sum(), math.sqrt(n_macro * w.var(ddof=1))

    (Ra, ea), (Rb, eb) = estimate(121, thr), estimate(122, 0.0)
    assert Ra > 0.0
    assert abs(Ra - Rb) < 5.0 * math.hypot(ea, eb)


def test_reproducibility_and_input_types():
    macros = np.array([[0.0, 0.0, 0.0, 1e-5, 0.0, 1e-3]] * 15)
    e1 = make_tc(seed=131, n_trials_per_macro=20).generate_section_events_6d(macros, 1.0)
    e2 = make_tc(seed=131, n_trials_per_macro=20).generate_section_events_6d((tuple(m) for m in macros), 1.0)
    assert e1 == e2  # same seed -> identical events; numpy array and generator inputs work
    assert make_tc().generate_section_events_6d([], 1.0) == []


def test_events_to_macros_cols():
    tc = make_tc(seed=141, n_trials_per_macro=20)
    events = tc.generate_section_events_6d([(1.0, 2.0, 3.0, 0.0, 0.0, 0.0)] * 7, 1.0)
    w, (x, xp, y, yp, z, dp) = tc.events_to_macros_cols(events, particle="e")
    assert w.shape == (7,) and np.array_equal(w, [ev["w_Hz"] for ev in events])
    assert np.all(x == 1.0) and np.all(y == 2.0) and np.all(z == 3.0)
    assert np.array_equal(dp, [ev["e6d_out"][5] for ev in events])
    _, (_, xp_g, _, _, _, dp_g) = tc.events_to_macros_cols(events, particle="g")
    assert np.array_equal(dp_g, [ev["g6d_out"][5] for ev in events])
    # Missing records: skipped by default, NaN rows with skip_none=False
    broken = events[:2] + [{"w_Hz": 1.0, "e6d_out": None}]
    assert tc.events_to_macros_cols(broken)[0].shape == (2,)
    w2, cols2 = tc.events_to_macros_cols(broken, skip_none=False)
    assert w2.shape == (3,) and np.isnan(cols2[0][2])
    # Empty input and invalid particle
    w3, cols3 = tc.events_to_macros_cols([])
    assert w3.size == 0 and all(c.size == 0 for c in cols3)
    with pytest.raises(ValueError):
        tc.events_to_macros_cols(events, particle="x")


@pytest.mark.parametrize("kwargs", [
    dict(Eb_GeV=0.0001), dict(I_A=-1.0), dict(T_K=0.0), dict(n_trials_per_macro=0),
    dict(dp_threshold=-1e-3), dict(photon_energy_sampler_eV=0.07),
])
def test_constructor_input_validation(kwargs):
    with pytest.raises((ValueError, TypeError)):
        make_tc(**kwargs)


def test_section_input_validation():
    tc = make_tc(n_trials_per_macro=5)
    with pytest.raises(ValueError):
        tc.generate_section_events_6d([(0.0,) * 6], L_m=-1.0)
    with pytest.raises(ValueError):
        tc.generate_section_events_6d([(0.0,) * 6], L_m=1.0, dp_threshold=-0.1)
    with pytest.raises(ValueError):
        tc.generate_section_events_6d([(0.0, 0.0, 0.0)], L_m=1.0)
    with pytest.raises(ValueError):
        tc.generate_section_events_6d([(0.0, 0.0, 0.0, 0.0, 0.0, -1.0)], L_m=1.0)


def test_custom_photon_energy_sampler():
    # A user-supplied sampler (here monochromatic 0.1 eV) is used for every trial
    tc = make_tc(seed=151, photon_energy_sampler_eV=lambda: 0.1, n_trials_per_macro=20)
    Pe = tc._sixd_to_four_electron(0, 0, 0, 0, 0, 0, tc.p0_GeV)
    for _ in range(200):
        Eg_in, _pg = tc._sample_thermal_photon(Pe, tc.photon_energy_sampler_eV, tc.rng)
        assert Eg_in == pytest.approx(0.1e-9, rel=1e-15)
    events = tc.generate_section_events_6d([(0.0,) * 6] * 5, 1.0, dp_threshold=0.0)
    assert all(ev["photon_E_GeV"] <= rf.compton_edge_GeV(18.0, 0.1e-9) * (1 + 1e-10) for ev in events)
