#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(D) Kinematics and physics consistency of complete scattering events.

Generates N accepted collisions of one electron (by default on-axis and
on-momentum) with thermal photons, using the generator's own photon sampling
and scattering routines, and checks:

  * scattered photon is massless                 |E^2 - p^2| / E^2
  * scattered electron is on its mass shell      |E^2 - p^2 - m_e^2|, in units of the
                                                 double-precision floor eps_mach*E^2
  * four-momentum conservation                   P_e + P_gamma = P_e' + P_gamma'
  * Compton edge                                 E_gamma' <= k (E+p) / (m_e^2/(E+p) + 2k)
  * direction of the colliding photons           <cos a> ≈ -beta/3 (flux factor) or ≈ 0
  * acceptance fraction                          = <sigma_KN/sigma_T>
  * mean scattered photon energy                 = exact quadrature value

The last three compare with reference_formulas.py, an independent deterministic
evaluation of the same physics. Exit status 1 if a check fails.

Usage:
    python validate_kinematics.py [--Eb 18] [--T 300] [-N 200000] [--isotropic]
    python validate_kinematics.py --Eb 275 --xp 2e-3 --yp=-1e-3 --dp=-5e-3
(write negative values in scientific notation as --yp=-1e-3, an argparse requirement)
"""

import argparse
import sys

import numpy as np

from main import ThermalCompton, m_e_GeV, GeV_per_eV
import reference_formulas as rf


def main():
    ap = argparse.ArgumentParser(description="(D) Kinematics consistency checks.")
    ap.add_argument("--Eb", type=float, default=18.0, help="Electron beam energy [GeV]")
    ap.add_argument("--T", type=float, default=300.0, help="Temperature [K]")
    ap.add_argument("-N", "--N", type=int, default=200_000, help="Number of accepted events to test")
    ap.add_argument("--seed", type=int, default=12345, help="RNG seed")
    ap.add_argument("--max-proposals", type=int, default=5_000_000,
                    help="Max Thomson proposals to attempt (safety)")
    ap.add_argument("--xp", type=float, default=0.0, help="Incoming electron x' = px/pz [rad]")
    ap.add_argument("--yp", type=float, default=0.0, help="Incoming electron y' = py/pz [rad]")
    ap.add_argument("--dp", type=float, default=0.0, help="Incoming electron dp = p/p0 - 1")
    ap.add_argument("--isotropic", action="store_true",
                    help="Isotropic photon directions (flux_weighting=False, Burkhardt 1993)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    flux = not args.isotropic

    # I_A does not matter for kinematics checks
    tc = ThermalCompton(Eb_GeV=args.Eb, I_A=2.5, T_K=args.T, rng=rng, flux_weighting=flux)
    sampler = tc.photon_energy_sampler_eV

    # Incoming electron
    Pe_in = tc._sixd_to_four_electron(0.0, 0.0, 0.0, args.xp, args.yp, args.dp, tc.p0_GeV)
    Ee_in, pe_in = Pe_in
    me2 = m_e_GeV * m_e_GeV
    e_hat = pe_in / np.linalg.norm(pe_in)   # electron direction
    eps_mach = np.finfo(float).eps

    # Accumulators
    n_trials = 0
    photon_mass_rel = []   # |E^2 - p^2| / E^2 of the scattered photon
    electron_shell = []    # |E^2 - p^2 - m_e^2| / (eps_mach * E^2) of the scattered electron
    conservation = []      # max |(P_e + P_g) - (P_e' + P_g')| / E_e
    edge_ratio = []        # E_gamma' / Compton edge for the actual incoming photon
    cos_a = []             # cos(angle) between colliding photon and electron (lab)
    Eg_out_list = []       # scattered photon energy [GeV]
    k_in_eV = []           # incoming (colliding) photon energy [eV]
    dxp, dyp = [], []      # change of the electron slopes [rad]

    for _ in range(args.max_proposals):
        # Same two steps as ThermalCompton._compton_scatter_one_trial, kept
        # separate here so that the incoming photon is known
        Eg_in, pg_in = tc._sample_thermal_photon(Pe_in, sampler, rng, flux_weighted=flux)
        accepted, e_out, g_out = tc._scatter_photon(Pe_in, (Eg_in, pg_in), rng)
        n_trials += 1
        if not accepted:
            continue

        Ee_out, pe_out = e_out
        Eg_out, pg_out = g_out
        pe_out = np.asarray(pe_out, dtype=np.float64)
        pg_out = np.asarray(pg_out, dtype=np.float64)

        photon_mass_rel.append(abs(Eg_out * Eg_out - pg_out @ pg_out) / (Eg_out * Eg_out))
        electron_shell.append(abs(Ee_out * Ee_out - pe_out @ pe_out - me2) / (eps_mach * Ee_out * Ee_out))
        conservation.append(max(abs(Ee_in + Eg_in - Ee_out - Eg_out),
                                float(np.max(np.abs(pe_in + pg_in - pe_out - pg_out)))) / Ee_in)
        edge_ratio.append(Eg_out / float(rf.compton_edge_GeV(Ee_in, Eg_in)))
        cos_a.append(float(pg_in @ e_hat) / Eg_in)
        Eg_out_list.append(Eg_out)
        k_in_eV.append(Eg_in / GeV_per_eV)
        dxp.append(pe_out[0] / pe_out[2] - args.xp)
        dyp.append(pe_out[1] / pe_out[2] - args.yp)

        if len(Eg_out_list) >= args.N:
            break

    n_acc = len(Eg_out_list)
    if n_acc < args.N:
        raise RuntimeError(
            f"Only collected {n_acc} accepted events out of requested {args.N}. "
            f"Increase --max-proposals."
        )

    photon_mass_rel = np.asarray(photon_mass_rel)
    electron_shell = np.asarray(electron_shell)
    conservation = np.asarray(conservation)
    edge_ratio = np.asarray(edge_ratio)
    cos_a = np.asarray(cos_a)
    Eg_out_arr = np.asarray(Eg_out_list)
    k_in_eV = np.asarray(k_in_eV)
    dxp = np.abs(np.asarray(dxp))
    dyp = np.abs(np.asarray(dyp))

    # ---- statistical comparisons with the quadrature references ----
    acc_frac = n_acc / n_trials
    S_ref = rf.mean_cross_section_ratio(Ee_in, args.T, flux_weighting=flux)
    pull_acc = (acc_frac - S_ref) / np.sqrt(S_ref * (1.0 - S_ref) / n_trials)

    Eg_ref, _ = rf.mean_scattered_photon_energy_GeV(Ee_in, args.T, flux_weighting=flux)
    Eg_mean = float(Eg_out_arr.mean())
    pull_E = (Eg_mean - Eg_ref) / (float(Eg_out_arr.std(ddof=1)) / np.sqrt(n_acc))

    # cos(a) of the colliding photons: pdf (1 - beta c)/2 for the flux factor
    # (mean -beta/3), uniform for isotropic sampling (mean 0); the Klein-Nishina
    # acceptance shifts both by O(gamma kT/m_e), included in the reference
    cos_ref = rf.mean_cos_alpha_of_collisions(Ee_in, args.T, flux_weighting=flux)
    cos_mean = float(cos_a.mean())
    pull_cos = (cos_mean - cos_ref) / (float(cos_a.std(ddof=1)) / np.sqrt(n_acc))

    def stats(name, arr):
        return f"{name}: mean={arr.mean():.3e}, rms={arr.std():.3e}, max={arr.max():.3e}"

    print("=== Kinematics consistency checks ===")
    print(f"Beam energy Eb = {args.Eb} GeV, T = {args.T} K, "
          f"photon directions: {'flux-weighted' if flux else 'isotropic'}")
    print(f"Incoming electron: x' = {args.xp}, y' = {args.yp}, dp = {args.dp}, E = {Ee_in:.9f} GeV")
    print(f"Accepted events tested: {n_acc:,} (of {n_trials:,} trials)")
    print("")
    print(stats("Photon  |E^2-p^2|/E^2 (should be ~0)", photon_mass_rel))
    print(stats("Electron |E^2-p^2-m_e^2| / (eps_mach E^2) (O(1) = round-off)", electron_shell))
    print(stats("Four-momentum conservation residual / E_e", conservation))
    print(stats("E_gamma' / Compton edge (must be <= 1)", edge_ratio))
    print(stats("Electron deflection |dx'| [rad]", dxp))
    print(stats("Electron deflection |dy'| [rad]", dyp))
    print("")
    print("Diagnostic: energy of the colliding thermal photons (eV):")
    print(f"k_in [eV]: mean={k_in_eV.mean():.4f}, median={np.median(k_in_eV):.4f}, "
          f"99%={np.quantile(k_in_eV, 0.99):.4f}, max={k_in_eV.max():.4f}")
    print(f"Mean scattered photon energy: MC = {Eg_mean:.6f} GeV, quadrature = {Eg_ref:.6f} GeV, "
          f"Thomson limit = {rf.thomson_limit_mean_photon_energy_GeV(Ee_in, args.T, flux):.6f} GeV")
    print("=====================================")

    checks = [
        # Round-off floor: rare wide-angle photons (large p_perp/E) reach
        # ~eps_mach*N; a kinematic error would show up at >= 1e-6
        ("scattered photon massless", photon_mass_rel.max() < 1e-8,
         f"max |E^2-p^2|/E^2 = {photon_mass_rel.max():.1e}"),
        ("scattered electron on mass shell", electron_shell.max() < 100.0,
         f"max residual = {electron_shell.max():.1f} x eps_mach E^2"),
        ("four-momentum conservation", conservation.max() < 1e-14,
         f"max residual / E_e = {conservation.max():.1e}"),
        ("Compton edge respected", edge_ratio.max() <= 1.0 + 1e-10,
         f"max E_gamma'/edge = {edge_ratio.max():.12f}"),
        ("photon direction distribution <cos a>", abs(pull_cos) < 5.0,
         f"{cos_mean:+.5f} vs {cos_ref:+.5f} (pull {pull_cos:+.2f} sigma)"),
        ("acceptance = <sigma_KN/sigma_T>", abs(pull_acc) < 5.0,
         f"{acc_frac:.5f} vs {S_ref:.5f} (pull {pull_acc:+.2f} sigma)"),
        ("mean scattered photon energy", abs(pull_E) < 5.0,
         f"{Eg_mean:.6f} vs {Eg_ref:.6f} GeV (pull {pull_E:+.2f} sigma)"),
    ]
    ok = rf.print_checks(checks, title="(D) Kinematics checks")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
