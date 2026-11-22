#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thermal Compton scattering of ultra-relativistic electrons on blackbody photons.

This module provides ``ThermalCompton``, a self-contained Monte Carlo event
generator designed to be called from a tracking code, one ring section at a
time. For every incoming macro-electron it simulates Compton scattering on the
thermal (blackbody) photon gas inside the vacuum chamber and returns the
scattered electron and photon in "6D-like" coordinates

    (x, y, z, x', y', dp),   x' = px/pz,  y' = py/pz,  dp = p/p0 - 1,

together with an absolute event weight in Hz. Summing the weights of the
electrons that are subsequently lost gives a loss rate, hence a lifetime.

Algorithm of one trial collision (H. Burkhardt, CERN SL/Note 93-73 (OP), 1993)
------------------------------------------------------------------------------
  1. Sample the thermal photon energy from the Planck photon-number spectrum.
  2. Sample the photon direction in the lab. The collision rate in an isotropic
     photon gas carries the relativistic flux factor (1 - beta*cos(a)), where a
     is the angle between the photon and the electron velocity, so head-on
     photons collide more often. (Set ``flux_weighting=False`` to sample the
     directions isotropically, as literally described in the 1993 note.)
  3. Boost the photon to the electron rest frame (ERF), rotate it onto +z.
  4. Propose the scattering angle from the Thomson distribution (1 + cos^2(theta))
     and accept it with probability (dsigma_KN/dOmega) / (dsigma_T/dOmega), so
     that accepted events follow the Klein-Nishina cross section.
  5. Rotate and boost the scattered photon back to the lab; the scattered
     electron follows from four-momentum conservation.

Units: energies and momenta in GeV (c = 1), lengths in m, current in A,
temperature in K, rates (weights) in Hz.

@author: Andrii Natochii natochii@bnl.gov
February 2026
"""

import math

import numpy as np

# =============================================================================
# Physical constants (exact SI values since 2019; CODATA 2018 otherwise)
# =============================================================================

# Speed of light [m/s] -- blackbody photon density (h*c factor).
c_SI = 299792458.0

# Boltzmann constant [J/K] -- blackbody photon density (k_B*T in J).
kB_SI = 1.380649e-23

# Planck constant [J*s] -- blackbody photon density (h*c in J*m).
h_SI = 6.62607015e-34

# Elementary charge [C] -- converts the beam current I [C/s] to I/e [electrons/s].
e_SI = 1.602176634e-19

# Boltzmann constant [eV/K] -- k_B*T in eV for the photon energy sampler.
kB_eV = 8.617333262145e-5

# Apery's constant zeta(3) [dimensionless] -- integral of the Planck photon-number spectrum.
zeta3 = 1.202056903159594

# Thomson cross section sigma_T [m^2] -- the "trial" (proposal) cross section.
# The Klein-Nishina cross section is obtained from it by accept/reject.
sigma_T_m2 = 6.6524587321e-29

# Electron rest energy m_e*c^2 [GeV] (c = 1 units).
m_e_GeV = 0.000510998950

# Energy conversion [GeV/eV].
GeV_per_eV = 1e-9


class ThermalCompton:
    """
    Monte Carlo generator of Compton scattering of beam electrons on thermal
    (blackbody) photons, with absolute event weights in Hz.

    Example
    -------
        tc = ThermalCompton(Eb_GeV=18.0, I_A=0.227, T_K=300.0,
                            rng=np.random.default_rng(12345))
        # macros_6d: sequence of (x, y, z, x', y', dp); s0: section start [m]
        events = tc.generate_section_events_6d(macros_6d, L_m=1.0,
                                               section_start_s_m=s0)
        rate_Hz = sum(ev["w_Hz"] for ev in events)

    Parameters
    ----------
    Eb_GeV : float
        Reference beam energy [GeV]. Defines the reference momentum
        p0 = sqrt(Eb^2 - m_e^2) used in dp = p/p0 - 1. Must exceed m_e*c^2.
    I_A : float
        Beam current [A] (>= 0). Sets the absolute normalization of the weights.
    T_K : float
        Temperature of the blackbody photon gas [K] (> 0). Sets the photon
        number density rho(T) and the photon energy spectrum.
    n_trials_per_macro : int, default 100
        Number K of trial collisions per macro-electron and per section call.
        A larger K lowers the statistical error of the weights; the CPU time
        scales as (number of macros) * K.
    nmax_sampler : int, default 2000
        Number of terms kept in the series expansion used by the built-in
        Planck photon-energy sampler (truncated probability ~1e-7).
    rng : numpy.random.Generator, optional
        Random generator used for all sampling. Pass a seeded generator, e.g.
        ``np.random.default_rng(12345)``, for reproducible results.
    photon_energy_sampler_eV : callable, optional
        Replacement for the built-in Planck sampler: a callable without
        arguments that returns one photon energy in eV. Note that the weight
        normalization still uses the blackbody photon density rho(T_K).
    dp_threshold : float, default 1e-3
        Default thinning threshold on |dp| of the scattered electron, used when
        ``generate_section_events_6d`` is called without ``dp_threshold``.
        It is a numerical bookkeeping cut, not the machine momentum acceptance.
    flux_weighting : bool, default True
        If True, thermal photon directions are sampled with the relativistic
        flux factor (1 - beta*cos(a)), as required for the collision rate in an
        isotropic photon gas. If False, the directions are isotropic in the lab
        frame, exactly as described in Burkhardt's 1993 note; this reproduces
        the numbers of that note but underestimates the mean energy transfer by
        ~25 % (gamma^2*<k> instead of (4/3)*gamma^2*<k>) and the high-energy tail
        of the spectrum. Use False only for benchmarking against the note.

    Derived attributes
    ------------------
    p0_GeV : float
        Reference momentum [GeV].
    rho_m3 : float
        Blackbody photon number density rho(T) [1/m^3].
    max_tries_rep_event : int
        Safety cap on the trials used to build a zero-weight representative
        event (see ``generate_section_events_6d``).

    Absolute weighting
    ------------------
    A beam current I corresponds to I/e electrons per second passing any point
    of the ring. In an isotropic photon gas of density rho(T), an
    ultra-relativistic electron undergoes rho(T) * <sigma> collisions per metre,
    where <sigma> is the Klein-Nishina cross section averaged over the photon
    spectrum and the flux factor (the isotropic average of (1 - beta*cos(a)) is
    exactly 1; the factor 1/beta between collisions per unit time and per unit
    length is neglected, 1 - beta = 1/(2 gamma^2) = 4e-10 at 18 GeV). The
    collision rate in a section of length L is therefore

        R = (I/e) * rho(T) * <sigma> * L     [Hz].

    With N macro-electrons and K = n_trials_per_macro trials per macro, each
    trial carries the Thomson-normalized weight

        w_trial = (I / (e*N)) * rho(T) * sigma_T * (L/K)     [Hz]

    and is accepted with probability (dsigma_KN/dOmega)/(dsigma_T/dOmega), so the
    accepted trials reproduce R on average (Burkhardt's proposal method). Summing
    w_Hz over all returned events estimates the collision rate (for
    dp_threshold = 0) or the rate of collisions with |dp| >= dp_threshold;
    summing w_Hz over the events whose electron is later lost estimates the
    loss rate.
    """

    # =========================================================================
    # Blackbody photon energy sampler (photon-number spectrum)
    # =========================================================================
    class BlackbodyPhotonEnergySampler:
        """
        Sample thermal photon energies from the blackbody photon-NUMBER spectrum

            p(E) dE  ∝  E^2 / (exp(E/kT) - 1) dE.

        With x = E/kT and the identity 1/(exp(x) - 1) = sum_{n>=1} exp(-n*x), the
        pdf becomes a mixture of Gamma distributions (x^2*exp(-n*x) integrates to
        2/n^3):

          1) choose the integer n with probability ∝ 1/n^3  (n = 1 .. nmax),
          2) sample x ~ Gamma(shape=3, scale=1/n),
          3) return E = x * kT.

        Truncating the series at nmax drops a probability ~1/(2*zeta(3)*nmax^2)
        (~1e-7 for nmax = 2000), concentrated at the very lowest energies.
        The mean energy is <E> = pi^4 / (30*zeta(3)) * kT ≈ 2.701 kT.

        Energies are returned in eV.
        """

        def __init__(self, T_K: float, nmax: int = 2000, rng: np.random.Generator | None = None):
            # Temperature [K]
            self.T_K = float(T_K)
            if not self.T_K > 0.0:
                raise ValueError(f"T_K must be > 0 K. Given: {T_K}")

            nmax = int(nmax)
            if nmax < 1:
                raise ValueError(f"nmax must be >= 1. Given: {nmax}")

            # kT in eV, so that E = x*kT is directly in eV
            self.kT_eV = kB_eV * self.T_K

            # Provided RNG (reproducibility) or a fresh default one
            self.rng = np.random.default_rng() if rng is None else rng

            # Discrete CDF of n = 1..nmax with weights ∝ 1/n^3, normalized to 1
            n = np.arange(1, nmax + 1, dtype=np.float64)
            cdf = np.cumsum(1.0 / n ** 3)
            cdf /= cdf[-1]
            self._cdf = cdf

        def sample_eV(self, size: int = 1) -> np.ndarray:
            """Return an array of ``size`` photon energies [eV]."""
            # 1) inverse-CDF sampling of the series index n in [1, nmax]
            u = self.rng.random(size)
            n = np.searchsorted(self._cdf, u) + 1.0

            # 2) x = E/kT ~ Gamma(shape=3, scale=1/n)
            x = self.rng.gamma(shape=3.0, scale=1.0 / n, size=size)

            # 3) back to energy in eV
            return x * self.kT_eV

        def __call__(self) -> float:
            """
            Return one photon energy [eV] as a Python float.

            Scalar fast path: it consumes the random stream exactly like
            ``sample_eV(1)`` and returns the identical value, without the
            overhead of creating one-element arrays.
            """
            u = self.rng.random()
            n = float(np.searchsorted(self._cdf, u) + 1)
            return float(self.rng.gamma(3.0, 1.0 / n)) * self.kT_eV

    # =========================================================================
    # Constructor: fixed (per-run) parameters are set only once
    # =========================================================================
    def __init__(
        self,
        Eb_GeV: float,                       # reference beam energy [GeV]
        I_A: float,                          # beam current [A]
        T_K: float,                          # photon gas temperature [K]
        n_trials_per_macro: int = 100,       # trial collisions per macro per section call
        nmax_sampler: int = 2000,            # series truncation of the built-in Planck sampler
        rng: np.random.Generator | None = None,
        photon_energy_sampler_eV=None,       # optional callable returning one photon energy [eV]
        dp_threshold: float = 1e-3,          # default thinning threshold on |dp| of the electron
        flux_weighting: bool = True,         # flux factor (1 - beta*cos a) in photon directions
    ):
        # ---- beam and environment parameters (validated) ----
        self.Eb_GeV = float(Eb_GeV)
        if not self.Eb_GeV > m_e_GeV:
            raise ValueError(
                f"Eb_GeV must exceed the electron rest energy {m_e_GeV} GeV. Given: {Eb_GeV}"
            )

        self.I_A = float(I_A)
        if not self.I_A >= 0.0:
            raise ValueError(f"I_A must be >= 0 A. Given: {I_A}")

        self.T_K = float(T_K)
        if not self.T_K > 0.0:
            raise ValueError(f"T_K must be > 0 K. Given: {T_K}")

        if int(n_trials_per_macro) <= 0:
            raise ValueError(
                f"n_trials_per_macro must be a positive integer. Given: {n_trials_per_macro}"
            )
        self.n_trials_per_macro = int(n_trials_per_macro)

        self.dp_threshold = float(dp_threshold)
        if not self.dp_threshold >= 0.0:
            raise ValueError(f"dp_threshold must be >= 0. Given: {dp_threshold}")

        self.flux_weighting = bool(flux_weighting)

        # Safety cap for the loop that builds a zero-weight representative event
        # (a single accepted trial is needed).
        self.max_tries_rep_event = 100_000

        # RNG shared by everything in this class (photon energy and direction,
        # angles, accept/reject, interaction position).
        self.rng = np.random.default_rng() if rng is None else rng

        # ---- derived quantities ----
        # Reference momentum p0 [GeV] from the reference beam energy
        self.p0_GeV = self._p0_from_Eb(self.Eb_GeV)

        # Blackbody photon number density rho(T) [1/m^3]
        self.rho_m3 = self._photon_density_blackbody(self.T_K)

        # Photon energy sampler (temperature fixed at initialization)
        if photon_energy_sampler_eV is None:
            self.photon_energy_sampler_eV = self.BlackbodyPhotonEnergySampler(
                T_K=self.T_K, nmax=nmax_sampler, rng=self.rng
            )
        else:
            if not callable(photon_energy_sampler_eV):
                raise TypeError(
                    "photon_energy_sampler_eV must be a callable returning one energy in eV."
                )
            self.photon_energy_sampler_eV = photon_energy_sampler_eV

    # =========================================================================
    # Public method: generate weighted events for one ring section
    # =========================================================================
    def generate_section_events_6d(
        self,
        macros_6d,                           # iterable of (x, y, z, x', y', dp)
        L_m: float,                          # section length [m]
        section_start_s_m: float = 0.0,      # s-coordinate of the section start [m]
        dp_threshold: float | None = None,   # thinning threshold on |dp| of the electron
    ):
        """
        Generate Compton scattering events in ONE ring section of length L_m.

        For every incoming macro-electron, K = n_trials_per_macro independent
        trial collisions are simulated (Thomson proposal + Klein-Nishina
        accept/reject). Among the accepted trials, those whose scattered
        electron satisfies |dp| >= dp_threshold are "tail" events. Exactly ONE
        representative event is returned per macro-electron:

          * if n_pass >= 1 tail events occurred, one of them is chosen uniformly
            at random and given the weight w_Hz = n_pass * w_trial, i.e. it
            stands for all n_pass tail events. Rates and any observable of the
            scattered particle (e.g. "is it lost?") remain unbiased:
            E[w_Hz * f(chosen)] = w_trial * E[sum over tail events of f];
          * if no tail event occurred, an accepted scattering is still returned
            (so that exactly one particle per macro can be tracked) but with
            w_Hz = 0. Its |dp| may be below dp_threshold.

        Parameters
        ----------
        macros_6d : iterable
            Incoming macro-electrons, each (x, y, z, x', y', dp) with
            x' = px/pz, y' = py/pz, dp = p/p0 - 1 (dp > -1). A sequence, an
            (N, 6) numpy array or a generator are all accepted. Each macro
            represents an equal share 1/N of the beam current.
        L_m : float
            Section length [m] (>= 0). The interaction point is uniformly
            distributed along the section.
        section_start_s_m : float, default 0.0
            s-coordinate of the section start [m]; only used to tag the events
            with s_m = section_start_s_m + U(0, L_m).
        dp_threshold : float or None, default None
            Keep as tail events only scattered electrons with |dp| >= dp_threshold
            (absolute momentum deviation of the outgoing electron, including the
            incoming dp). None uses the value given to the constructor.
            Use 0.0 to keep every accepted collision (total collision rate).

        Returns
        -------
        events : list of dict, one per input macro-electron, with keys
            "s_m"          : float, interaction position [m]
            "w_Hz"         : float, absolute rate weight [Hz] of this event
            "e6d_out"      : tuple (x, y, z, x', y', dp) of the scattered electron
            "g6d_out"      : tuple (x, y, z, x', y', dp) of the scattered photon,
                             with x' = px/pz, y' = py/pz and dp = |p_gamma|/p0 - 1
                             (photon momentum relative to the electron p0)
            "photon_E_GeV" : float, scattered photon energy [GeV]
            The transverse/longitudinal positions (x, y, z) are those of the
            incoming macro-electron (instantaneous, point-like interaction).
        """
        # -----------------------------
        # Validate / normalize inputs
        # -----------------------------
        L_m = float(L_m)
        if not L_m >= 0.0:
            raise ValueError(f"L_m must be >= 0. Given: {L_m} [m]")

        dp_threshold = self.dp_threshold if dp_threshold is None else float(dp_threshold)
        if not dp_threshold >= 0.0:
            raise ValueError(f"dp_threshold must be >= 0. Given: {dp_threshold}")

        # Materialize macros_6d if it has no __len__ (e.g. a generator)
        if not hasattr(macros_6d, "__len__"):
            macros_6d = list(macros_6d)

        n_macro = len(macros_6d)
        if n_macro == 0:
            return []

        rng = self.rng
        p0 = self.p0_GeV
        s0 = float(section_start_s_m)

        # Number of independent trial collisions per macro-electron. Each trial
        # represents a slice L_m/K of the section.
        K = self.n_trials_per_macro

        # -----------------------------
        # Weight of ONE trial [Hz]
        # -----------------------------
        # I_A / e_SI   : electrons per second passing a point of the ring [1/s]
        # / n_macro    : each macro-electron carries an equal share of that flow
        # rho_m3       : blackbody photon number density [1/m^3]
        # sigma_T_m2   : Thomson cross section [m^2] (proposal cross section)
        # L_m / K      : path length represented by one trial [m]
        # Units: (1/s) * (1/m^3) * (m^2) * (m) = 1/s = Hz.
        # The KN/Thomson acceptance of each trial turns sigma_T into the
        # Klein-Nishina cross section on average, so sigma_T stays here.
        w_trial_Hz = (self.I_A / (e_SI * n_macro)) * self.rho_m3 * sigma_T_m2 * (L_m / K)

        events = []

        # -----------------------------
        # Loop over incoming macro-electrons
        # -----------------------------
        for macro in macros_6d:
            try:
                x, y, z, xp, yp, dp = macro
            except (TypeError, ValueError):
                raise ValueError(
                    "Each macro-electron must be a sequence (x, y, z, x', y', dp) "
                    f"of 6 numbers. Got: {macro!r}"
                ) from None
            if not (1.0 + float(dp)) > 0.0:
                raise ValueError(f"Macro-electron dp must be > -1. Got dp = {dp}")

            # Incoming electron four-momentum (E, p) in the lab [GeV]
            Pe_lab = self._sixd_to_four_electron(x, y, z, xp, yp, dp, p0)

            # ---- (A) K physical trials: they define the normalization ----
            # Each tail candidate is (s_m, x'_e, y'_e, dp_e, E_gamma, p_gamma).
            tail_candidates = []
            for _ in range(K):
                trial = self._run_trial(Pe_lab, s0, L_m)
                if trial is not None and abs(trial[3]) >= dp_threshold:
                    tail_candidates.append(trial)

            n_pass = len(tail_candidates)

            # ---- (B) exactly ONE representative output event per macro ----
            if n_pass > 0:
                # A real tail event, standing for all n_pass tail events
                s_m, xp_e, yp_e, dp_e, Eg_out, pg_out = tail_candidates[rng.integers(n_pass)]
                w_Hz = w_trial_Hz * n_pass
            else:
                # No tail event in the K physical trials: return one accepted
                # scattering with ZERO weight, so that it does not bias the rates.
                for _ in range(self.max_tries_rep_event):
                    trial = self._run_trial(Pe_lab, s0, L_m)
                    if trial is not None:
                        break
                else:
                    raise RuntimeError(
                        "ThermalCompton: failed to generate a representative accepted event "
                        f"after {self.max_tries_rep_event} attempts (no tail event found). "
                        "Check the RNG, the photon energy sampler and the macro coordinates. "
                        f"Eb_GeV={self.Eb_GeV}, T_K={self.T_K}, L_m={L_m}, K={K}"
                    )
                s_m, xp_e, yp_e, dp_e, Eg_out, pg_out = trial
                w_Hz = 0.0

            # -----------------------------
            # Output coordinates
            # -----------------------------
            # Electron: keep (x, y, z) of the incoming macro (point-like interaction)
            e6d_out = (float(x), float(y), float(z), float(xp_e), float(yp_e), float(dp_e))

            # Photon "6D-like" at the same point: slopes px/pz, py/pz (guarded
            # against pz = 0) and dp_g = |p_gamma|/p0 - 1 relative to the ELECTRON p0
            xp_g = self._safe_slope(pg_out[0], pg_out[2])
            yp_g = self._safe_slope(pg_out[1], pg_out[2])
            p_g = float(np.linalg.norm(pg_out))
            dp_g = (p_g / p0) - 1.0
            g6d_out = (float(x), float(y), float(z), float(xp_g), float(yp_g), float(dp_g))

            events.append({
                "s_m": float(s_m),
                "w_Hz": float(w_Hz),
                "e6d_out": e6d_out,
                "g6d_out": g6d_out,
                "photon_E_GeV": float(Eg_out),
            })

        return events

    @staticmethod
    def events_to_macros_cols(events, particle="e", skip_none=True):
        """
        Convert a list of events into weight and coordinate columns:

            [w_arr, [x_arr, xp_arr, y_arr, yp_arr, z_arr, dp_arr]]

        where w_arr holds ev["w_Hz"] and the coordinates come from
        ev["e6d_out"] (particle="e") or ev["g6d_out"] (particle="g"). Note the
        column ORDER (x, x', y, y', z, dp), which differs from the event tuples
        (x, y, z, x', y', dp).

        Parameters
        ----------
        events : list of dict
            Output of ``generate_section_events_6d``.
        particle : {"e", "g"}, default "e"
            "e" for the scattered electrons, "g" for the scattered photons.
        skip_none : bool, default True
            Events whose coordinate tuple is missing or None are skipped
            (True) or kept as rows of NaN (False), so that the columns stay
            aligned with ``events``. ``generate_section_events_6d`` always
            fills both tuples; this only matters for user-modified event lists.

        Returns
        -------
        [w_arr, [x_arr, xp_arr, y_arr, yp_arr, z_arr, dp_arr]] of float64 arrays
        (all empty if no usable event).
        """
        if particle not in ("e", "g"):
            raise ValueError(f'particle must be "e" (electron) or "g" (photon). Given: {particle!r}')
        key = "e6d_out" if particle == "e" else "g6d_out"

        rows = []    # (x, y, z, xp, yp, dp) per event
        w_list = []  # matching weights
        nan_row = (np.nan,) * 6

        for ev in events:
            six = ev.get(key, None)
            if six is None:
                if skip_none:
                    continue
                six = nan_row
            rows.append(six)
            w_list.append(ev["w_Hz"])

        # No usable events: empty arrays with a consistent structure
        if len(rows) == 0:
            empty = np.array([], dtype=np.float64)
            return [empty, [empty.copy() for _ in range(6)]]

        w = np.asarray(w_list, dtype=np.float64)

        # Rows -> independent column arrays in event order (x, y, z, xp, yp, dp)
        x, y, z, xp, yp, dp = (np.ascontiguousarray(col)
                               for col in np.asarray(rows, dtype=np.float64).T)

        # Reorder to [x, x', y, y', z, dp]
        return [w, [x, xp, y, yp, z, dp]]

    # =========================================================================
    # Private helpers: one trial and the scattering physics
    # =========================================================================
    def _run_trial(self, Pe_lab, s0, L_m):
        """
        Run one trial collision of the electron Pe_lab = (E, p_vec) in a section
        starting at s0 [m] with length L_m [m].

        Returns None if the trial is rejected by the Klein-Nishina acceptance
        (or if the scattered electron would move backwards, pz <= 0), else the
        tuple (s_m, x'_e, y'_e, dp_e, E_gamma_GeV, p_gamma_vec_GeV).
        """
        # Interaction point, uniform along the section
        s_m = s0 + L_m * self.rng.random()

        accepted, e_out, g_out = self._compton_scatter_one_trial(
            Pe_lab, self.photon_energy_sampler_eV, self.rng, flux_weighted=self.flux_weighting
        )
        if not accepted:
            return None

        _, pe_out = e_out
        Eg_out, pg_out = g_out

        e_slopes = self._four_to_sixd_slopes(pe_out[0], pe_out[1], pe_out[2], self.p0_GeV)
        if e_slopes is None:
            return None

        xp_e, yp_e, dp_e = e_slopes
        return (s_m, xp_e, yp_e, dp_e, float(Eg_out), np.asarray(pg_out, dtype=float))

    @classmethod
    def _compton_scatter_one_trial(cls, Pe_lab, sampler_eV, rng, flux_weighted=True):
        """
        One trial collision: sample a thermal photon, then one Thomson-proposed
        scattering with a single Klein-Nishina accept/reject.

        Parameters
        ----------
        Pe_lab : (E_GeV, p_vec_GeV)
            Incoming electron four-momentum in the lab.
        sampler_eV : callable
            Returns one thermal photon energy [eV] per call.
        rng : numpy.random.Generator
        flux_weighted : bool
            Photon direction with (True) or without (False) the flux factor.

        Returns
        -------
        accepted : bool
        (E_e_out, p_e_out), (E_g_out, p_g_out) : lab four-momenta [GeV] of the
            scattered electron and photon (None, None if rejected).
        """
        Pg_in = cls._sample_thermal_photon(Pe_lab, sampler_eV, rng, flux_weighted=flux_weighted)
        return cls._scatter_photon(Pe_lab, Pg_in, rng)

    @classmethod
    def _sample_thermal_photon(cls, Pe_lab, sampler_eV, rng, flux_weighted=True):
        """
        Sample the lab four-momentum (E, p_vec) [GeV] of a thermal photon that
        collides with the electron Pe_lab = (E, p_vec).

        The energy comes from ``sampler_eV`` (Planck spectrum by default), the
        direction is drawn with the flux factor (1 - beta*cos(a)) if
        ``flux_weighted`` is True, isotropically otherwise.
        """
        E_e, p_e = Pe_lab

        # Photon energy in the lab [eV -> GeV]
        k_lab_GeV = float(sampler_eV()) * GeV_per_eV

        # Photon direction in the lab (unit vector)
        if flux_weighted:
            n_in = cls._sample_flux_weighted_direction(rng, np.asarray(p_e, dtype=float) / float(E_e))
        else:
            n_in = cls._sample_isotropic_direction(rng)

        return k_lab_GeV, k_lab_GeV * n_in

    @classmethod
    def _scatter_photon(cls, Pe_lab, Pg_in, rng):
        """
        Compton-scatter a GIVEN incoming photon Pg_in = (E, p_vec) on the
        electron Pe_lab = (E, p_vec) (lab frame, GeV): Thomson-proposed
        scattering angle, accepted once with probability
        (dsigma_KN/dOmega)/(dsigma_T/dOmega). The electron must be on its mass
        shell, E^2 = p^2 + m_e^2 (as built by ``_sixd_to_four_electron``).

        Returns (accepted, (E_e_out, p_e_out), (E_g_out, p_g_out)), with
        (None, None) for the four-momenta of a rejected trial.
        """
        E_e, p_e = Pe_lab
        Eg_in, pg_in = Pg_in
        E_e = float(E_e)
        Eg_in = float(Eg_in)
        p_e = np.asarray(p_e, dtype=float)
        pg_in = np.asarray(pg_in, dtype=float)

        # Electron velocity (c = 1): boost vector to the electron rest frame (ERF).
        # The Lorentz factor is taken as E/m_e (the electron is on its mass
        # shell) rather than 1/sqrt(1 - beta^2), which would lose ~log10(gamma^2)
        # digits (~7 at 18 GeV) to cancellation.
        beta_e = p_e / E_e
        gamma_e = E_e / m_e_GeV

        # Photon in the ERF; its energy k* drives the Klein-Nishina corrections
        Eg_star, pg_star = cls._lorentz_boost(Eg_in, pg_in, beta_e, gamma_e)
        k_star = Eg_star

        # Rotation that brings the incoming ERF photon direction onto +z
        R_to_z = cls._rotation_matrix_from_vec_to_z(pg_star)

        # Propose the scattering angles (relative to the incoming photon) from Thomson
        cos_th, sin_th, phi = cls._sample_thomson_angles(rng)

        # Accept with probability (dsigma_KN/dOmega)/(dsigma_T/dOmega) <= 1
        accept_prob, x = cls._compton_over_thomson_ratio(k_star, cos_th)
        if rng.random() >= accept_prob:
            return False, None, None  # rejected trial

        # Accepted: scattered photon in the rotated ERF, energy k*' = x*k*
        k_star_out = x * k_star
        pg_star_out_rot = k_star_out * np.array([
            sin_th * np.cos(phi),
            sin_th * np.sin(phi),
            cos_th,
        ], dtype=float)

        # Undo the rotation (R is orthogonal: R^-1 = R^T)
        pg_star_out = R_to_z.T @ pg_star_out_rot
        Eg_star_out = k_star_out

        # Boost back to the lab
        Eg_out, pg_out = cls._lorentz_boost(Eg_star_out, pg_star_out, -beta_e, gamma_e)

        # Scattered electron from four-momentum conservation in the lab
        E_e_out = E_e + Eg_in - Eg_out
        p_e_out = p_e + pg_in - pg_out

        return True, (float(E_e_out), p_e_out), (float(Eg_out), pg_out)

    # =========================================================================
    # Private helpers: physics formulas and coordinate transforms
    # (static/class methods, free of instance state)
    # =========================================================================
    @staticmethod
    def _photon_density_blackbody(T_K: float) -> float:
        """
        Blackbody photon number density rho(T) [1/m^3]:

            rho(T) = 16*pi*zeta(3) * (k_B*T / (h*c))^3
                   = (2*zeta(3)/pi^2) * (k_B*T / (hbar*c))^3

        (5.477e14 m^-3 at 300 K).
        """
        factor = (kB_SI * float(T_K)) / (h_SI * c_SI)   # [1/m]
        return float(16.0 * np.pi * zeta3 * factor ** 3)  # [1/m^3]

    @staticmethod
    def _p0_from_Eb(Eb_GeV: float) -> float:
        """
        Reference momentum p0 [GeV] from the reference beam energy Eb [GeV]:
            Eb^2 = p0^2 + m_e^2  =>  p0 = sqrt(Eb^2 - m_e^2)
        """
        Eb = float(Eb_GeV)
        return float(np.sqrt(max(0.0, Eb * Eb - m_e_GeV * m_e_GeV)))

    @staticmethod
    def _sixd_to_four_electron(x, y, z, xp, yp, dp, p0_GeV):
        """
        Convert (x, y, z, x', y', dp) to the electron four-momentum (E, p_vec)
        in the lab [GeV]. Positions do not enter the momentum.

        Conventions:
          x' = px/pz,  y' = py/pz,  dp = (p - p0)/p0  =>  p = p0*(1 + dp)
        Then:
          pz = p / sqrt(1 + x'^2 + y'^2),  px = x'*pz,  py = y'*pz,
          E  = sqrt(p^2 + m_e^2)
        """
        p = float(p0_GeV) * (1.0 + float(dp))
        denom = np.sqrt(1.0 + float(xp) * float(xp) + float(yp) * float(yp))
        pz = p / denom
        px = float(xp) * pz
        py = float(yp) * pz
        E = np.sqrt(p * p + m_e_GeV * m_e_GeV)
        return float(E), np.array([px, py, pz], dtype=float)

    @staticmethod
    def _four_to_sixd_slopes(px, py, pz, p0_GeV):
        """
        Convert the momentum (px, py, pz) [GeV] to (x', y', dp).

        Returns None if pz <= 0: the ring coordinate system assumes forward
        motion. (This never happens for scattered ultra-relativistic electrons,
        whose deflection is ~1e-7 rad.)
        """
        pz = float(pz)
        if pz <= 0.0:
            return None

        px = float(px)
        py = float(py)

        xp = px / pz
        yp = py / pz

        p = float(np.sqrt(px * px + py * py + pz * pz))
        dp = (p / float(p0_GeV)) - 1.0

        return float(xp), float(yp), float(dp)

    @staticmethod
    def _safe_slope(num, den, huge=1e12):
        """
        Ratio num/den for slopes (px/pz, py/pz); returns sign(num)*huge if den == 0.
        """
        num = float(num)
        den = float(den)
        if abs(den) > 0.0:
            return float(num / den)
        return float(np.sign(num) * huge)

    @staticmethod
    def _lorentz_boost(E, p, beta, gamma=None):
        """
        Lorentz boost of the four-vector (E, p_vec) into the frame moving with
        velocity beta_vec (|beta| < 1, c = 1):

          E' = gamma*(E - beta.p)
          p' = p + [ (gamma - 1)*(beta.p)/beta^2 - gamma*E ] * beta

        A boost with -beta_vec (same gamma) is the inverse transformation.

        gamma : float, optional
            Lorentz factor of beta_vec. If None it is computed as
            1/sqrt(1 - beta^2), which loses about log10(gamma^2) significant
            digits for ultra-relativistic velocities (1 - beta^2 = 1/gamma^2 is
            a tiny difference of numbers close to 1). Pass gamma = E/m when the
            energy of the boosting particle is known, to keep full precision.
        """
        beta = np.asarray(beta, dtype=float)
        p = np.asarray(p, dtype=float)

        b2 = float(np.dot(beta, beta))
        if b2 == 0.0:
            return float(E), p.copy()

        gamma = 1.0 / np.sqrt(1.0 - b2) if gamma is None else float(gamma)
        bp = float(np.dot(beta, p))

        E_prime = gamma * (float(E) - bp)
        coeff = ((gamma - 1.0) * bp / b2) - gamma * float(E)
        p_prime = p + coeff * beta

        return float(E_prime), p_prime

    @staticmethod
    def _rotation_matrix_from_vec_to_z(v):
        """
        Rotation matrix R such that R @ v_hat = z_hat (Rodrigues formula):

            R = I + sin(t) K + (1 - cos(t)) K^2,

        where t is the angle between v and z, and K is the cross-product matrix
        of the unit rotation axis (v_hat x z_hat)/sin(t) = (v_y, -v_x, 0)/sin(t).
        sin(t) = sqrt(v_x^2 + v_y^2) and cos(t) = v_z (for unit v) are used
        directly, without arccos, which keeps full precision near t = 0 and
        t = pi (in the ERF the incoming photon is almost always close to -z).

        Special cases: v = 0 or v along +z -> identity; v along -z -> rotation
        by pi about the x axis.
        """
        v = np.asarray(v, dtype=float)
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            return np.eye(3)

        vx, vy, vz = (float(c) for c in v / norm)
        sin_t = math.hypot(vx, vy)
        cos_t = vz
        if sin_t == 0.0:
            return np.eye(3) if cos_t > 0.0 else np.diag([1.0, -1.0, -1.0])

        # Unit rotation axis (ax, ay, 0)
        ax = vy / sin_t
        ay = -vx / sin_t
        K = np.array([[0.0, 0.0, ay],
                      [0.0, 0.0, -ax],
                      [-ay, ax, 0.0]], dtype=float)

        # 1 - cos(t), written as sin^2/(1 + cos) for cos > 0 to avoid cancellation
        one_minus_cos = sin_t * sin_t / (1.0 + cos_t) if cos_t > 0.0 else 1.0 - cos_t

        return np.eye(3) + sin_t * K + one_minus_cos * (K @ K)

    @staticmethod
    def _sample_isotropic_direction(rng):
        """
        Isotropic unit vector: cos(theta) uniform in [-1, 1], phi uniform in [0, 2*pi).
        """
        u = rng.random()
        v = rng.random()
        cos_theta = 2.0 * u - 1.0
        sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
        phi = 2.0 * np.pi * v
        return np.array([sin_theta * np.cos(phi),
                         sin_theta * np.sin(phi),
                         cos_theta], dtype=float)

    @staticmethod
    def _sample_flux_weighted_direction(rng, beta_vec):
        """
        Lab direction (unit vector) of a thermal photon that collides with an
        electron of velocity beta_vec (c = 1), for an isotropic photon gas.

        The collision rate is proportional to the flux factor (1 - beta*cos(a)),
        a being the angle between the photon direction and beta_vec. Its
        isotropic average is exactly 1, so cos(a) has the normalized pdf

            f(c) = (1 - beta*c) / 2,     -1 <= c <= 1     (mean -beta/3),

        and the azimuth around beta_vec is uniform: head-on photons (c = -1) are
        favoured. c is drawn by inverting the CDF F(c) = [(1 + c) - beta*(c^2 - 1)/2] / 2,
        using the cancellation-free root

            c = (4u - 2 - beta) / (1 + sqrt((1 + beta)^2 - 4*beta*u)),   u ~ U[0, 1),

        which reduces to the isotropic c = 2u - 1 for beta = 0.
        """
        bx, by, bz = (float(b) for b in beta_vec)
        beta = math.sqrt(bx * bx + by * by + bz * bz)

        # cos(a) by inverse CDF and uniform azimuth (two uniforms, like the isotropic sampler)
        u = rng.random()
        v = rng.random()
        disc = max(0.0, (1.0 + beta) ** 2 - 4.0 * beta * u)
        cos_a = (4.0 * u - 2.0 - beta) / (1.0 + math.sqrt(disc))
        cos_a = min(1.0, max(-1.0, cos_a))
        sin_a = math.sqrt(max(0.0, 1.0 - cos_a * cos_a))
        phi = 2.0 * math.pi * v
        cp = sin_a * math.cos(phi)
        sp = sin_a * math.sin(phi)

        if beta == 0.0:
            # Electron at rest: isotropic (here f(c) = 1/2); any polar axis will do
            return np.array([cp, sp, cos_a], dtype=float)

        # Orthonormal basis (e1, e2, e3) with e3 along the electron velocity
        # (branchless construction of Duff et al., J. Comput. Graph. Tech. 6 (2017) 1)
        e3x, e3y, e3z = bx / beta, by / beta, bz / beta
        sign = math.copysign(1.0, e3z)
        a = -1.0 / (sign + e3z)
        b = e3x * e3y * a
        e1x, e1y, e1z = 1.0 + sign * e3x * e3x * a, sign * b, -sign * e3x
        e2x, e2y, e2z = b, sign + e3y * e3y * a, -e3y

        return np.array([cp * e1x + sp * e2x + cos_a * e3x,
                         cp * e1y + sp * e2y + cos_a * e3y,
                         cp * e1z + sp * e2z + cos_a * e3z], dtype=float)

    @staticmethod
    def _sample_thomson_angles(rng):
        """
        Sample (cos(theta), sin(theta), phi) from the Thomson distribution
            dsigma_T/dOmega ∝ 1 + cos^2(theta).

        Exact mixture (the pdf of c = cos(theta) is 3/4 * 1/2 + 1/4 * 3c^2/2):
          - phi uniform in [0, 2*pi)
          - 75 %: c uniform in [-1, 1]
          - 25 %: |c| = max(r1, r2, r3) (pdf 3u^2 on [0, 1]) with a random sign
        """
        phi = 2.0 * np.pi * rng.random()
        if rng.random() < 0.75:
            cos_th = 2.0 * rng.random() - 1.0
        else:
            u = max(rng.random(), rng.random(), rng.random())
            cos_th = u if rng.random() < 0.5 else -u
        sin_th = np.sqrt(max(0.0, 1.0 - cos_th * cos_th))
        return float(cos_th), float(sin_th), float(phi)

    @staticmethod
    def _compton_over_thomson_ratio(k_in_star_GeV, cos_theta):
        """
        Acceptance probability R = (dsigma_KN/dOmega) / (dsigma_T/dOmega) in the
        ERF, and the Compton energy ratio x = k*_out / k*_in.

          x  = 1 / (1 + (k*/m_e)*(1 - cos(theta)))
          Thomson shape:       th = 1 + cos^2(theta)
          Klein-Nishina shape: kn = x^2 * [1 + cos^2(theta) + (x + 1/x - 2)]
          R  = kn / th  (always <= 1; equal to 1 in the Thomson limit k* -> 0)
        """
        k_in_star_GeV = float(k_in_star_GeV)
        cos_theta = float(cos_theta)

        x = 1.0 / (1.0 + (k_in_star_GeV / m_e_GeV) * (1.0 - cos_theta))
        th = 1.0 + cos_theta * cos_theta
        kn = (1.0 + cos_theta * cos_theta + (x + 1.0 / x - 2.0)) * (x * x)
        R = kn / th

        # Clip to [0, 1] for numerical safety (it is a probability)
        return float(min(1.0, max(0.0, R))), float(x)
