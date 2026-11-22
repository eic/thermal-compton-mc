#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(E) Absolute rate normalization and scaling checks, plus tau_min.

Runs ThermalCompton.generate_section_events_6d for a cold, on-axis beam with
dp_threshold = 0 (every accepted collision counted) and compares the summed
weights R_MC [Hz] with

  * the exact expectation R_ref = (I/e) rho(T) sigma_T L <sigma_KN/sigma_T>,
    where <sigma_KN/sigma_T> is computed by quadrature (reference_formulas.py),
  * the simple Thomson estimate R_Th = (I/e) rho(T) sigma_T L (a few % higher),

and checks the scaling with current (linear), section length (linear) and
temperature (rho ∝ T^3, times the small change of <sigma_KN/sigma_T> with T).
Also prints the minimum lifetime tau_min = 1/(rho sigma_T c) (every collision
lost). Prints a PASS/FAIL summary (exit status 1 on failure); --plot saves a
scaling plot.

Usage:
    python validate_absolute_rate.py [--Eb 18] [--T0 300] [--I0 2.5] [--L0 1.0] [--plot]

Runtime: about 2 min with the defaults (10 runs of n_macro*K = 400,000 trials).
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from main import ThermalCompton, e_SI, sigma_T_m2, c_SI
import reference_formulas as rf


def estimate_section_rate_Hz(Eb_GeV: float, I_A: float, T_K: float, L_m: float,
                             n_macro: int, n_trials_per_macro: int, seed: int,
                             flux_weighting: bool = True) -> dict:
    """
    Monte Carlo collision rate of one section and its references.

    Returns a dict with
      R_mc_Hz  : Monte Carlo estimate (sum of event weights) [Hz]
      err_Hz   : its statistical standard error [Hz]
      R_th_Hz  : Thomson estimate (I/e) rho sigma_T L [Hz]
      R_ref_Hz : exact expectation R_th * <sigma_KN/sigma_T> [Hz]
    dp_threshold = 0 so that ALL accepted collisions contribute to the rate.
    """
    rng = np.random.default_rng(seed)
    tc = ThermalCompton(
        Eb_GeV=Eb_GeV,
        I_A=I_A,
        T_K=T_K,
        n_trials_per_macro=n_trials_per_macro,
        rng=rng,
        flux_weighting=flux_weighting,
    )

    # Simple "cold" beam: on-axis, on-momentum macro-electrons
    macros = [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(n_macro)]

    events = tc.generate_section_events_6d(
        macros_6d=macros,
        L_m=L_m,
        section_start_s_m=0.0,
        dp_threshold=0.0,  # keep all accepted events for the total rate
    )

    w = np.array([ev["w_Hz"] for ev in events])
    R_mc_Hz = float(w.sum())
    # Macro-electrons are independent: Var(sum w) = n_macro * Var(w)
    err_Hz = float(np.sqrt(n_macro * w.var(ddof=1)))

    R_th_Hz = (I_A / e_SI) * tc.rho_m3 * sigma_T_m2 * L_m
    R_ref_Hz = R_th_Hz * rf.mean_cross_section_ratio(Eb_GeV, T_K, flux_weighting=flux_weighting)
    return {"R_mc_Hz": R_mc_Hz, "err_Hz": err_Hz, "R_th_Hz": float(R_th_Hz), "R_ref_Hz": float(R_ref_Hz)}


def tau_min_hours(T_K: float) -> float:
    """
    Rough minimum lifetime (f = 1) assuming that every collision leads to a loss:
        tau_min = 1 / (rho(T) * sigma_T * c)
    Independent of beam current and ring geometry (v ≈ c).
    """
    rho = ThermalCompton._photon_density_blackbody(T_K)  # [1/m^3]
    tau_s = 1.0 / (rho * sigma_T_m2 * c_SI)
    return tau_s / 3600.0


def main():
    ap = argparse.ArgumentParser(description="(E) Absolute rate normalization and scaling checks.")
    ap.add_argument("--Eb", type=float, default=18.0, help="Reference electron energy [GeV]")
    ap.add_argument("--T0", type=float, default=300.0, help="Baseline temperature [K]")
    ap.add_argument("--I0", type=float, default=2.5, help="Baseline current [A]")
    ap.add_argument("--L0", type=float, default=1.0, help="Baseline section length [m]")
    ap.add_argument("--n-macro", type=int, default=2000, help="Number of macro-electrons")
    ap.add_argument("--K", type=int, default=200, help="Trials per macro-electron")
    ap.add_argument("--seed", type=int, default=12345, help="Base RNG seed")
    ap.add_argument("--isotropic", action="store_true",
                    help="Isotropic photon directions (flux_weighting=False, Burkhardt 1993)")
    ap.add_argument("--plot", action="store_true", help="Save a scaling plot")
    ap.add_argument("--out", type=str, default="absolute_rate_scaling.png", help="Output plot filename")
    args = ap.parse_args()

    Eb = float(args.Eb)
    T0 = float(args.T0)
    I0 = float(args.I0)
    L0 = float(args.L0)
    flux = not args.isotropic

    def run(I_A, T_K, L_m, seed):
        return estimate_section_rate_Hz(Eb, I_A, T_K, L_m, args.n_macro, args.K, seed, flux)

    print("=== Absolute rate scaling checks ===")
    print(f"Baseline: Eb={Eb} GeV, T0={T0} K, I0={I0} A, L0={L0} m, "
          f"photon directions: {'flux-weighted' if flux else 'isotropic'}")
    print(f"Sampling: n_macro={args.n_macro}, K={args.K}")
    print("")

    checks = []

    # ---- baseline rate vs exact expectation ----
    r0 = run(I0, T0, L0, args.seed)
    R0_mc, R0_err = r0["R_mc_Hz"], r0["err_Hz"]
    pull0 = (R0_mc - r0["R_ref_Hz"]) / R0_err
    print("Baseline section rate:")
    print(f"  R_MC  = {R0_mc:.6e} ± {R0_err:.1e} Hz")
    print(f"  R_ref = {r0['R_ref_Hz']:.6e} Hz   (exact: Thomson x <sigma_KN/sigma_T>)")
    print(f"  R_Th  = {r0['R_th_Hz']:.6e} Hz   (Thomson estimate)")
    print(f"  Ratio R_MC/R_Th = {R0_mc / r0['R_th_Hz']:.6f}   (expected {r0['R_ref_Hz'] / r0['R_th_Hz']:.6f})")
    print("")
    checks.append(("baseline rate = exact expectation", abs(pull0) < 5.0,
                   f"R_MC/R_ref = {R0_mc / r0['R_ref_Hz']:.6f} (pull {pull0:+.2f} sigma)"))

    def scan(label, symbol, unit, base, factors, seed_offset, make_args, expected_ratio):
        """Run a scan of one parameter and check R_MC/R0 against the expectation."""
        rates = []
        worst = 0.0
        print(f"{label} scaling:")
        for j, f in enumerate(factors):
            r = run(*make_args(f), args.seed + seed_offset + j)
            ratio = r["R_mc_Hz"] / R0_mc
            # Independent runs: combine the relative errors of both rates
            ratio_err = ratio * np.hypot(r["err_Hz"] / r["R_mc_Hz"], R0_err / R0_mc)
            expected = expected_ratio(f)
            pull = (ratio - expected) / ratio_err
            worst = max(worst, abs(pull))
            rates.append(r["R_mc_Hz"])
            print(f"  {symbol} = {base * f:.4g} {unit} (x{f:g}):  R_MC/R0 = {ratio:.6f}   "
                  f"expected = {expected:.6f}   (pull {pull:+.2f})")
        print("")
        checks.append((f"{label.lower()} scaling", worst < 5.0, f"max |pull| = {worst:.2f} sigma"))
        return np.array(rates)

    # ---- I scaling (linear) ----
    I_factors = np.array([0.5, 1.0, 2.0], dtype=float)
    R_I = scan("Current", "I", "A", I0, I_factors, 10_000,
               lambda f: (I0 * f, T0, L0), lambda f: f)

    # ---- L scaling (linear) ----
    L_factors = np.array([0.5, 1.0, 2.0], dtype=float)
    R_L = scan("Length", "L", "m", L0, L_factors, 20_000,
               lambda f: (I0, T0, L0 * f), lambda f: f)

    # ---- T scaling: rho ∝ T^3, times the (small) T dependence of <sigma_KN/sigma_T> ----
    T_factors = np.array([0.8, 1.0, 1.2], dtype=float)
    S0 = rf.mean_cross_section_ratio(Eb, T0, flux_weighting=flux)
    R_T = scan("Temperature", "T", "K", T0, T_factors, 30_000,
               lambda f: (I0, T0 * f, L0),
               lambda f: f ** 3 * rf.mean_cross_section_ratio(Eb, T0 * f, flux_weighting=flux) / S0)

    # ---- tau_min ----
    tau_h = tau_min_hours(T0)
    print(f"Rough minimum lifetime at T={T0:.1f} K (f=1):")
    print(f"  tau_min = {tau_h:.3f} hours  from  1/(n_ph * sigma_T * c)")
    print("==================================")

    ok = rf.print_checks(checks, title="(E) Absolute rate checks")

    # ---- optional plot ----
    if args.plot:
        plt.figure()
        # Normalize all to baseline
        plt.plot(I_factors, R_I / R0_mc, marker="o", linestyle="-", label="I scaling (measured)")
        plt.plot(I_factors, I_factors, linestyle="--", label="I scaling (expected)")

        plt.plot(L_factors, R_L / R0_mc, marker="o", linestyle="-", label="L scaling (measured)")
        plt.plot(L_factors, L_factors, linestyle="--", label="L scaling (expected)")

        plt.plot(T_factors, R_T / R0_mc, marker="o", linestyle="-", label="T scaling (measured)")
        plt.plot(T_factors, T_factors ** 3, linestyle="--", label=r"T scaling (expected $\propto T^3$)")

        plt.xlabel("Scale factor")
        plt.ylabel(r"Normalized rate $R_{\rm MC}/R_0$")
        plt.title("Absolute rate scaling checks")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out, dpi=200)
        print(f"Saved: {args.out}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
