#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrii Natochii natochii@bnl.gov
February 2026
"""
# ============================================================
#  Thermal Compton scattering on blackbody photons
#  Packaged as ONE class for use in tracking code.
#
#  - Generates accepted Compton events using Thomson-angle sampling
#    + Klein–Nishina (Compton/Thomson) rejection.
#  - Returns outgoing electron + photon in a "6D-like" form:
#       (x, y, z, x', y', dp)
#    where x' = px/pz, y' = py/pz, dp = (p/p0) - 1.
#  - Provides an absolute event rate weight in Hz suitable for
#    summing losses around the ring to obtain a lifetime.
# ============================================================

import numpy as np

# -----------------------------
# Physical constants (units matter!)
# -----------------------------

# Speed of light [m/s]
# Used in blackbody photon number density (hc factor).
c_SI = 299792458.0

# Boltzmann constant [J/K]
# Used in blackbody photon number density via k_B*T (J).
kB_SI = 1.380649e-23

# Planck constant [J*s]
# Used in blackbody photon number density via h*c (J*m).
h_SI  = 6.62607015e-34

# Elementary charge [C]
# Used to convert current I [C/s] -> particle flow rate I/e [1/s].
e_SI  = 1.602176634e-19

# Boltzmann constant in electron-volts [eV/K]
# Used to convert temperature to kT in eV for photon energy sampling.
kB_eV = 8.617333262145e-5  # eV/K

# Apery's constant ζ(3) (dimensionless)
# Appears in analytic integrals of the Planck distribution for photon number density.
zeta3 = 1.202056903159594

# Thomson cross section σ_T [m^2]
# This is used as the "trial" cross section.
# The actual Compton (Klein–Nishina) distribution is obtained by rejection.
sigma_T_m2 = 6.6524587321e-29  # m^2

# Electron rest mass energy m_e c^2 in GeV (c=1 units) [GeV]
# Used in relativistic relations and Compton energy-shift formula in the ERF.
m_e_GeV = 0.000510998950  # GeV

# Energy conversion [GeV/eV]
# Used to convert sampled photon energy in eV to GeV for 4-vector kinematics.
GeV_per_eV = 1e-9


class ThermalCompton:
    """
    Self-contained thermal Compton (blackbody) event generator for tracking studies.
    
    Initialize once with:
      - Eb_GeV : reference beam energy [GeV] (defines p0)
      - I_A    : beam current [A] (sets absolute rate normalization)
      - T_K    : temperature [K] (sets photon density and energy sampler)
    
    For each ring section of length L_m, call:
        generate_section_events_6d(macros_6d, L_m, section_start_s_m=...)
    
    Inputs:
      - macros_6d : iterable of macro-electrons (x, y, z, x', y', dp)
      - L_m       : section length [m]
      - section_start_s_m : optional s-offset tag [m]
    
    Returns a list of events, each a dict with:
      - s_m          : interaction position [m] (uniform in the section)
      - w_Hz         : absolute event weight [Hz]
      - e6d_out      : scattered electron (x, y, z, x', y', dp) (None if pz<=0)
      - g6d_out      : scattered photon "6D-like" (x, y, z, x', y', dp)
      - photon_E_GeV : scattered photon energy [GeV]
    
    Absolute weighting:
      Electron flow rate is Ndot = I/e. In a photon gas of density rho(T), the collision rate
      in a section of length L is R = (I/e) * rho(T) * sigma * L.
    
    Following Burkhardt’s proposal method, trial angles are sampled from Thomson
    (dsigma_T/dOmega ∝ 1+cos^2θ) and accepted with probability (dsigma_KN/dOmega)/(dsigma_T/dOmega),
    so accepted events follow Klein–Nishina.
    
    With Nmacro macro-electrons and K = n_trials_per_macro trials per macro per section, each
    accepted trial carries:
      w_Hz = (I/(e*Nmacro)) * rho(T) * sigma_T * (L/K).
    
    Summing w_Hz over accepted events gives the Compton rate [Hz] in that section; summing w_Hz
    over accepted events whose electron is subsequently lost gives the loss rate [Hz].
    """

    # =====================================================================
    # Blackbody photon energy sampler (photon-number spectrum)
    # =====================================================================
    class BlackbodyPhotonEnergySampler:
        """
        Samples thermal photon energies from the blackbody *photon-number* spectrum:
            p(E) dE ∝ E^2 / (exp(E/kT) - 1) dE

        Uses the identity:
            1/(exp(x) - 1) = Σ_{n=1..∞} exp(-n x),   x = E/(kT)

        This turns the PDF into a mixture of Gamma distributions:
          1) Choose integer n with probability ∝ 1/n^3
          2) Sample x ~ Gamma(shape=3, scale=1/n)
          3) Return E = x * kT

        This sampler returns E in eV.
        """
        def __init__(self, T_K: float, nmax: int = 2000, rng: np.random.Generator | None = None):
            # Store temperature [K]
            self.T_K = float(T_K)

            # Precompute kT in eV so that E = x*kT returns eV directly
            self.kT_eV = kB_eV * self.T_K

            # Use provided RNG (for reproducibility) or create a default RNG
            self.rng = np.random.default_rng() if rng is None else rng

            # Build discrete CDF for n=1..nmax with weights w_n ∝ 1/n^3
            n = np.arange(1, nmax + 1, dtype=np.float64)
            w = 1.0 / (n ** 3)
            cdf = np.cumsum(w)
            cdf /= cdf[-1]            # normalize to 1
            self._cdf = cdf

        def sample_eV(self, size: int = 1) -> np.ndarray:
            # Inverse-CDF sampling of n
            u = self.rng.random(size)
            idx = np.searchsorted(self._cdf, u)       # 0-based indices
            n_chosen = (idx + 1).astype(np.float64)   # convert to n in [1..nmax]

            # Sample x = E/(kT) from Gamma(k=3, θ=1/n)
            x = self.rng.gamma(shape=3.0, scale=1.0 / n_chosen, size=size)

            # Convert back to energies in eV: E = x*kT
            return x * self.kT_eV

        def __call__(self) -> float:
            # Return one sample as a Python float
            return float(self.sample_eV(size=1)[0])

    # =====================================================================
    # Constructor: set all "fixed" parameters only once
    # =====================================================================
    def __init__(
        self,
        Eb_GeV: float,                      # reference beam energy [GeV]
        I_A: float,                         # beam current [A]
        T_K: float,                         # temperature [K]
        n_trials_per_macro: int = 100,       # number of trial scatterings per macro per section call
        nmax_sampler: int = 2000,            # truncation for mixture sampler (controls tail accuracy)
        rng: np.random.Generator | None = None,
        photon_energy_sampler_eV=None,       # optional: user-supplied callable returning photon energy [eV]
        dp_threshold: float = 1e-3,
    ):
        self.dp_threshold = float(dp_threshold)
        self.max_tries_rep_event = 100_000
        
        # RNG used everywhere in this class (angles, accept/reject, blackbody sampling, etc.)
        self.rng = np.random.default_rng() if rng is None else rng

        # Store beam/environment scalars
        self.Eb_GeV = float(Eb_GeV)
        self.I_A = float(I_A)
        self.T_K = float(T_K)

        # Validate and store number of trials per macro
        if int(n_trials_per_macro) <= 0:
            raise ValueError("n_trials_per_macro must be a positive integer.")
        self.n_trials_per_macro = int(n_trials_per_macro)

        # Compute and store reference momentum p0 [GeV] from reference beam energy Eb [GeV]
        self.p0_GeV = self._p0_from_Eb(self.Eb_GeV)

        # Compute and store blackbody photon number density ρ(T) [1/m^3]
        self.rho_m3 = self._photon_density_blackbody(self.T_K)

        # Photon energy sampler (temperature is set once at initialization)
        if photon_energy_sampler_eV is None:
            # Default: build internal sampler for the specified temperature
            self.photon_energy_sampler_eV = self.BlackbodyPhotonEnergySampler(
                T_K=self.T_K, nmax=nmax_sampler, rng=self.rng
            )
        else:
            # User provides sampler that returns energy in eV; just store it
            self.photon_energy_sampler_eV = photon_energy_sampler_eV

    # =====================================================================
    # Public method: generate accepted events for one ring section
    # =====================================================================
    def generate_section_events_6d( 
            self, 
            macros_6d,                          # iterable of (x, y, z, xp, yp, dp) 
            L_m: float,                         # section length [m] 
            section_start_s_m: float = 0.0,     # section start coordinate s [m]
            dp_threshold: float | None = None,  # keep only events with |dp_e| >= dp_threshold 
            ):
        """
        Generate *tail-only* Compton scattering events in ONE section of length L_m using
        a Thomson proposal + KN accept/reject done ONCE per trial.
    
        Key goals:
          1) Return only scattered electrons with |dp| >= dp_threshold.
          2) Keep the absolute rate normalization correct (weights in Hz).
          3) Keep output particle count manageable: at most ~1 scattered electron per input macro
             (per section call), by collapsing multiple tail events into one representative.
    
        Inputs
        ------
        macros_6d : iterable
            Each element is (x, y, z, xp, yp, dp), representing one incoming macro-electron.
    
        L_m : float
            Section length [m]. Interaction point is assumed uniformly distributed along [0, L_m].
    
        section_start_s_m : float
            The section start s-coordinate [m] used only to tag events with:
                s_m = section_start_s_m + U(0, L_m).
    
        dp_threshold : float | None
            If provided, keep only events whose outgoing electron satisfies |dp_e| >= dp_threshold.
            If None, a default is taken from self.dp_threshold if present; otherwise ValueError.
    
        Returns
        -------
        events : list[dict]
            Each dict contains:
              - "s_m"       : float, interaction tag position [m]
              - "w_Hz"      : float, absolute rate weight [Hz] for this *representative* event
              - "e6d_out"   : tuple (x, y, z, xp, yp, dp) for the scattered electron
              - "g6d_out"   : tuple (x, y, z, xp, yp, dp) for the scattered photon ("6D-like")
              - "photon_E_GeV" : float, outgoing photon energy [GeV] (diagnostic)
        """
    
        # -----------------------------
        # Validate / normalize inputs
        # -----------------------------
        L_m = float(L_m)
        if L_m < 0.0:
            raise ValueError(f"L_m must be >= 0. Given: {L_m} [m]")
    
        # Decide dp threshold
        if dp_threshold is None: 
            dp_threshold = self.dp_threshold
        else:
            dp_threshold = float(dp_threshold)
    
        # Materialize macros_6d if it has no __len__ (e.g., a generator)
        if not hasattr(macros_6d, "__len__"):
            macros_6d = list(macros_6d)
    
        Nmacro = len(macros_6d)
        if Nmacro == 0:
            return []
    
        # -----------------------------
        # Local references (speed + clarity)
        # -----------------------------
        rng        = self.rng
        sampler_eV = self.photon_energy_sampler_eV
        p0         = self.p0_GeV
        s0         = float(section_start_s_m)
    
        # Number of independent TRIALS per incoming macro-electron in this section
        # Each trial represents a slice of length L_m / K along which a Thomson "attempt" is made
        K = int(self.n_trials_per_macro)
        if K <= 0:
            raise ValueError(f"self.n_trials_per_macro must be >= 1. Given: {K}")
    
        # -----------------------------
        # Weight per TRIAL (Hz)
        # -----------------------------
        # (I_A / e_SI)          : beam electron flow rate through a point [1/s]
        # divide by Nmacro      : each macro represents equal share of the beam flux
        # rho_m3                : blackbody photon number density [1/m^3]
        # sigma_T_m2            : Thomson cross section [m^2] used as the proposal ("trial") cross section
        # (L_m / K)             : path length represented by one trial [m]
        #
        # Units: (1/s)*(1/m^3)*(m^2)*(m) = 1/s (Hz)
        #
        # IMPORTANT:
        # Use _compton_scatter_one_trial(...) which:
        #   - samples a thermal photon,
        #   - proposes Thomson angles,
        #   - accepts/rejects ONCE using KN/Thomson ratio.
        # Therefore, "accepted" trials happen at the *correct* Compton rate automatically
        # via the acceptance fraction, and keep sigma_T in the trial weight.
        w_trial_Hz = (self.I_A / (e_SI * Nmacro)) * self.rho_m3 * sigma_T_m2 * (L_m / K)
    
        events = []
    
        # -----------------------------
        # Loop over incoming macro-electrons
        # -----------------------------
        for (x, y, z, xp, yp, dp) in macros_6d:
    
            # Convert incoming 6D macro to electron 4-vector (E, p⃗) in LAB [GeV]
            Ee_in, pe_in = self._sixd_to_four_electron(x, y, z, xp, yp, dp, p0)
            Pe_lab = (Ee_in, pe_in)
    
            # Collect only *tail* candidates produced by accepted trials.
            # Each candidate corresponds to a real physical tail event with weight w_trial_Hz.
            tail_candidates = []  # list of tuples: (s_m, xp_e, yp_e, dp_e, Eg_out, pg_out)
    
            # -----------------------------
            # Perform K independent trials for THIS macro-electron in this section
            # -----------------------------
            
            # ---- (A) Fixed K trials: this is what defines the PHYSICAL normalization for this section ----
            for _ in range(K):
                s_m = s0 + L_m * rng.random()
            
                accepted, e_out, g_out = self._compton_scatter_one_trial(Pe_lab, sampler_eV, rng)
                if not accepted:
                    continue
            
                Ee_out, pe_out = e_out
                Eg_out, pg_out = g_out
            
                e_slopes = self._four_to_sixd_slopes(pe_out[0], pe_out[1], pe_out[2], p0)
                if e_slopes is None:
                    continue
            
                xp_e, yp_e, dp_e = e_slopes
            
                if abs(dp_e) >= dp_threshold:
                    tail_candidates.append((s_m, xp_e, yp_e, dp_e, Eg_out, np.asarray(pg_out, float)))
            
            n_pass = len(tail_candidates)
            
            # ---- (B) Choose ONE output event (always) ----
            if n_pass > 0:
                # Choose a real tail event from the K physical trials
                s_m, xp_e, yp_e, dp_e, Eg_out, pg_out = tail_candidates[rng.integers(n_pass)]
                w_Hz = w_trial_Hz * n_pass  # correct: represents ALL tail events among the K trials
            else:
                # No tail event observed in these K physical trials.
                # Still produce ONE kinematics (possible to track exactly one particle per macro),
                # but give it ZERO weight so it does not bias the tail rate estimate.
                for _try in range(self.max_tries):
                    s_m = s0 + L_m * rng.random()
                
                    accepted, e_out, g_out = self._compton_scatter_one_trial(Pe_lab, sampler_eV, rng)
                    if not accepted:
                        continue
                
                    Ee_out, pe_out = e_out
                    Eg_out, pg_out = g_out
                
                    e_slopes = self._four_to_sixd_slopes(pe_out[0], pe_out[1], pe_out[2], p0)
                    if e_slopes is None:
                        continue
                
                    xp_e, yp_e, dp_e = e_slopes
                    pg_out = np.asarray(pg_out, float)
                    w_Hz = 0.0
                    break
                else:
                    raise RuntimeError(
                        f"ThermalCompton: failed to generate a representative accepted event "
                        f"after {self.max_tries} attempts (n_pass==0 branch). "
                        f"Check RNG, sampler, and kinematics inputs. "
                        f"Eb_GeV={self.Eb_GeV}, T_K={self.T_K}, L_m={L_m}, K={K}"
                    )
    
            # -----------------------------
            # Build output 6D for electron and "6D-like" for photon
            # -----------------------------
            # Keep (x, y, z) at the interaction point (instantaneous scatter at this location).
            e6d_out = (float(x), float(y), float(z), float(xp_e), float(yp_e), float(dp_e))
    
            # Photon "6D-like" at the same point:
            # slopes: x' = px/pz, y' = py/pz (safe against pz ~ 0)
            xp_g = self._safe_slope(pg_out[0], pg_out[2])
            yp_g = self._safe_slope(pg_out[1], pg_out[2])
    
            # dp_g uses reference momentum p0 as normalization: dp_g = (|p_g|/p0) - 1
            p_g = float(np.linalg.norm(pg_out))
            dp_g = (p_g / p0) - 1.0
            g6d_out = (float(x), float(y), float(z), float(xp_g), float(yp_g), float(dp_g))
    
            # Store the representative event
            events.append({
                "s_m": float(s_m),
                "w_Hz": float(w_Hz),
                "e6d_out": e6d_out,
                "g6d_out": g6d_out,
                "photon_E_GeV": float(Eg_out),
            })
    
        return events


    def events_to_macros_cols(self, events, particle="e", skip_none=True):
        """
        Convert events (list of dicts) into the structure:

            scat_Beam = [ w_arr, [x_arr, xp_arr, y_arr, yp_arr, z_arr, dp_arr] ]

        Where:
          - w_arr is the event weight array (from ev["w_Hz"])
          - (x_arr, xp_arr, y_arr, yp_arr, z_arr, dp_arr) come from:
              ev["e6d_out"] = (x, y, z, xp, yp, dp)   if particle="e"
              ev["g6d_out"] = (x, y, z, xp, yp, dp)   if particle="g"

        Parameters
        ----------
        events : list[dict]
            Output of generate_section_events_6d(...).
        particle : {"e","g"}
            "e" -> use "e6d_out", "g" -> use "g6d_out"
        skip_none : bool
            If True, skip events where sixD output is None (relevant for electron case).

        Returns
        -------
        [w_arr, [x_arr, xp_arr, y_arr, yp_arr, z_arr, dp_arr]]
        """
        key = "e6d_out" if particle == "e" else "g6d_out"

        # Collect per-event rows and weights
        rows = []   # will hold tuples (x,y,z,xp,yp,dp)
        w_list = [] # will hold corresponding weights

        for ev in events:
            six = ev.get(key, None)      # (x,y,z,xp,yp,dp) or None
            if six is None:
                if skip_none:
                    continue
                # If skip_none=False is requested, define a fill policy; currently None outputs are skipped.
                continue

            rows.append(six)
            w_list.append(ev["w_Hz"])

        # If no usable events, return empty arrays (consistent dtypes, consistent structure)
        if len(rows) == 0:
            empty = np.array([], dtype=np.float64)
            return [empty, [empty, empty, empty, empty, empty, empty]]

        # Convert weights to a 1D float array
        w = np.asarray(w_list, dtype=np.float64)

        # Unzip rows -> columns in generator order: (x, y, z, xp, yp, dp)
        x, y, z, xp, yp, dp = map(lambda a: np.asarray(a, dtype=np.float64), zip(*rows))

        # Reorder to the preferred structure: [x, x', y, y', z, dp]
        return [w, [x, xp, y, yp, z, dp]]

    # =====================================================================
    # Private helpers: physics and coordinate transforms
    # (kept as static/class methods to avoid accidental state bugs)
    # =====================================================================

    @staticmethod
    def _photon_density_blackbody(T_K: float) -> float:
        """
        Blackbody photon number density ρ(T) [1/m^3]:
            ρ(T) = 16 π ζ(3) * (k_B T / (h c))^3
        """
        factor = (kB_SI * float(T_K)) / (h_SI * c_SI)   # [1/m]
        return float(16.0 * np.pi * zeta3 * factor**3)  # [1/m^3]

    @staticmethod
    def _p0_from_Eb(Eb_GeV: float) -> float:
        """
        Reference momentum p0 [GeV] from reference beam energy Eb [GeV]:
            Eb^2 = p0^2 + m_e^2  =>  p0 = sqrt(Eb^2 - m_e^2)
        """
        Eb = float(Eb_GeV)
        return float(np.sqrt(max(0.0, Eb * Eb - m_e_GeV * m_e_GeV)))

    @staticmethod
    def _sixd_to_four_electron(x, y, z, xp, yp, dp, p0_GeV):
        """
        Convert (x,y,z,x',y',dp) to electron 4-vector (E, p⃗) in LAB [GeV].

        Conventions:
          x' = px/pz,  y' = py/pz
          dp = (p - p0)/p0  =>  p = p0*(1+dp)

        Then:
          pz = p / sqrt(1 + x'^2 + y'^2)
          px = x' * pz
          py = y' * pz
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
        Convert momentum (px,py,pz) to (x', y', dp).

        Returns None if pz <= 0 because the usual ring coordinate system assumes forward motion.
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
        Safe ratio num/den for slopes (px/pz, py/pz). If den==0 returns +/- huge.
        """
        num = float(num)
        den = float(den)
        if abs(den) > 0.0:
            return float(num / den)
        return float(np.sign(num) * huge)

    @staticmethod
    def _lorentz_boost(E, p, beta):
        """
        Lorentz boost of (E, p⃗) by velocity beta⃗ (|beta|<1), with c=1:

          E' = γ(E - β·p)
          p' = p + [ (γ-1)(β·p)/β^2 - γE ] β
        """
        beta = np.asarray(beta, dtype=float)
        p = np.asarray(p, dtype=float)

        b2 = float(np.dot(beta, beta))
        if b2 == 0.0:
            return float(E), p.copy()

        gamma = 1.0 / np.sqrt(1.0 - b2)
        bp = float(np.dot(beta, p))

        E_prime = gamma * (float(E) - bp)
        coeff = ((gamma - 1.0) * bp / b2) - gamma * float(E)
        p_prime = p + coeff * beta

        return float(E_prime), p_prime

    @staticmethod
    def _rotation_matrix_from_vec_to_z(v):
        """
        Build rotation matrix R such that:
            R @ v_hat = z_hat

        Uses Rodrigues formula. Handles special cases:
          - v=0  -> identity
          - already aligned with +z -> identity
          - anti-aligned (-z) -> rotate by pi about any axis perpendicular to v
        """
        v = np.asarray(v, dtype=float)
        norm = np.linalg.norm(v)
        if norm == 0.0:
            return np.eye(3)

        v_hat = v / norm
        z_hat = np.array([0.0, 0.0, 1.0])
        dot = np.clip(float(np.dot(v_hat, z_hat)), -1.0, 1.0)

        # If already aligned: no rotation
        if dot > 1.0 - 1e-15:
            return np.eye(3)

        # If opposite: choose a perpendicular axis and rotate by pi
        if dot < -1.0 + 1e-15:
            axis = np.array([1.0, 0.0, 0.0])
            if abs(v_hat[0]) > 0.9:
                axis = np.array([0.0, 1.0, 0.0])
            axis = axis - np.dot(axis, v_hat) * v_hat
            axis /= np.linalg.norm(axis)
            angle = np.pi
        else:
            # General case: axis = v_hat x z_hat, angle = arccos(dot)
            axis = np.cross(v_hat, z_hat)
            axis /= np.linalg.norm(axis)
            angle = np.arccos(dot)

        ax, ay, az = axis
        K = np.array([[0.0, -az,  ay],
                      [az,  0.0, -ax],
                      [-ay, ax,  0.0]], dtype=float)

        I = np.eye(3)
        R = I + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
        return R

    @staticmethod
    def _sample_isotropic_direction(rng):
        """
        Sample isotropic unit vector on sphere:
          cosθ uniform in [-1,1], φ uniform in [0,2π).
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
    def _sample_thomson_angles(rng):
        """
        Sample (θ, φ) from Thomson:
            dσ/dΩ ∝ 1 + cos^2θ

        Exact mixture:
          - φ uniform
          - 75%: cosθ uniform in [-1,1]
          - 25%: |cosθ| with pdf ∝ u^2 generated by u=max(r1,r2,r3), then random sign
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
        Compute acceptance probability R = (dσ_KN/dΩ)/(dσ_T/dΩ) in ERF,
        and the Compton energy ratio x = k_out*/k_in*.

        In ERF:
          x = 1 / (1 + (k/m_e)(1 - cosθ))

        Thomson (shape): th = 1 + cos^2θ
        Klein–Nishina (shape): kn = x^2 * [ 1 + cos^2θ + (x + 1/x - 2) ]
        Ratio: R = kn/th
        """
        k_in_star_GeV = float(k_in_star_GeV)
        cos_theta = float(cos_theta)

        x = 1.0 / (1.0 + (k_in_star_GeV / m_e_GeV) * (1.0 - cos_theta))
        th = 1.0 + cos_theta * cos_theta
        kn = (1.0 + cos_theta * cos_theta + (x + 1.0 / x - 2.0)) * (x * x)
        R = kn / th

        # Clip to [0,1] for numerical safety (probability)
        return float(min(1.0, max(0.0, R))), float(x)
    
    @classmethod
    def _compton_scatter_one_trial(cls, Pe_lab, sampler_eV, rng):
        """
        One Thomson-proposed trial with KN accept/reject ONCE.
        Returns:
          accepted (bool),
          (Ee_out, pe_out), (Eg_out, pg_out)  only valid if accepted=True
        """
        E_e, p_e = Pe_lab
        beta_e = p_e / E_e
    
        # thermal photon in LAB
        k_lab_GeV = float(sampler_eV()) * GeV_per_eV
        n_in = cls._sample_isotropic_direction(rng)
        Eg_in = k_lab_GeV
        pg_in = k_lab_GeV * n_in
    
        # boost to ERF
        Eg_star, pg_star = cls._lorentz_boost(Eg_in, pg_in, beta_e)
        k_star = Eg_star
    
        # rotate incoming photon to +z in ERF
        R_to_z = cls._rotation_matrix_from_vec_to_z(pg_star)
        # pg_star_rot = R_to_z @ pg_star
    
        # propose angles from Thomson (one shot)
        cos_th, sin_th, phi = cls._sample_thomson_angles(rng)
    
        # KN/Thomson acceptance probability (depends on k_star and cos_th)
        R, x = cls._compton_over_thomson_ratio(k_star, cos_th)
        if rng.random() >= R:
            return False, None, None  # rejected trial
    
        # accepted: build outgoing photon in ERF rotated frame
        k_star_out = x * k_star
        pg_star_out_rot = k_star_out * np.array([
            sin_th * np.cos(phi),
            sin_th * np.sin(phi),
            cos_th
        ], dtype=float)
    
        # rotate back
        pg_star_out = R_to_z.T @ pg_star_out_rot
        Eg_star_out = k_star_out
    
        # boost back to LAB
        Eg_out, pg_out = cls._lorentz_boost(Eg_star_out, pg_star_out, -beta_e)
    
        # electron from 4-momentum conservation in LAB
        E_e_out = E_e + Eg_in - Eg_out
        p_e_out = p_e + pg_in - pg_out
    
        return True, (float(E_e_out), p_e_out), (float(Eg_out), pg_out)