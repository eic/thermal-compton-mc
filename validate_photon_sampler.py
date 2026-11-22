#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(A) Validation of the blackbody photon energy sampler.

Draws N photon energies from the built-in Planck sampler of ThermalCompton and
compares them with the exact photon-number spectrum p(E) ∝ E^2/(exp(E/kT) - 1):

  * mean energy      <E>   = pi^4/(30 zeta(3)) kT ≈ 2.701 kT
  * second moment    <E^2> = 12 zeta(5)/zeta(3) (kT)^2 ≈ 10.35 (kT)^2
  * histogram shape  chi^2 against the exact (bin-integrated) probabilities

Prints a PASS/FAIL summary (exit status 1 on failure) and saves a plot of the
histogram with the analytic spectrum.

Usage:
    python validate_photon_sampler.py [--T 300] [-N 300000] [--out photon_sampler_validation.pdf]
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

# Assumes this script lives next to main.py
from main import ThermalCompton, kB_eV
import reference_formulas as rf


def main():
    ap = argparse.ArgumentParser(description="(A) Validate the blackbody photon energy sampler.")
    ap.add_argument("--T", type=float, default=300.0, help="Temperature [K]")
    ap.add_argument("-N", "--N", type=int, default=300_000, help="Number of samples")
    ap.add_argument("--bins", type=int, default=120, help="Histogram bins")
    ap.add_argument("--Emax", type=float, default=0.5, help="Max energy on plot [eV]")
    ap.add_argument("--nmax", type=int, default=2000, help="Mixture truncation nmax for sampler")
    ap.add_argument("--seed", type=int, default=12345, help="RNG seed")
    ap.add_argument("--out", type=str, default="photon_sampler_validation.pdf", help="Output figure")
    args = ap.parse_args()

    T_K = float(args.T)
    N = int(args.N)
    rng = np.random.default_rng(args.seed)

    # Only the sampler is needed; Eb and I are irrelevant here
    tc = ThermalCompton(Eb_GeV=10.0, I_A=2.5, T_K=T_K, nmax_sampler=args.nmax, rng=rng)
    sampler = tc.photon_energy_sampler_eV

    # Sample energies [eV] and the dimensionless x = E/kT
    E = sampler.sample_eV(size=N)
    kT_eV = kB_eV * T_K
    x = E / kT_eV

    # ---- moments vs exact values (pull = deviation / standard error) ----
    mean_x = float(x.mean())
    se_mean_x = float(x.std(ddof=1)) / np.sqrt(N)
    pull_mean = (mean_x - rf.PLANCK_MEAN_X) / se_mean_x

    x2 = x * x
    mean_x2 = float(x2.mean())
    se_mean_x2 = float(x2.std(ddof=1)) / np.sqrt(N)
    pull_x2 = (mean_x2 - rf.PLANCK_MEAN_X2) / se_mean_x2

    mean_expected_eV = rf.PLANCK_MEAN_X * kT_eV
    print(f"T = {T_K:.1f} K, kT = {kT_eV:.6f} eV, N = {N:,}")
    print(f"Sample mean <E>         = {mean_x * kT_eV:.6f} eV")
    print(f"Expected <E> = 2.701 kT = {mean_expected_eV:.6f} eV")
    print(f"Ratio (sample/expected) = {mean_x / rf.PLANCK_MEAN_X:.6f}")
    print(f"<E^2>/(kT)^2: sample = {mean_x2:.5f}, expected = {rf.PLANCK_MEAN_X2:.5f}")

    # ---- histogram vs exact bin probabilities ----
    Emax = float(args.Emax)
    edges = np.linspace(0.0, Emax, args.bins + 1)
    counts, _ = np.histogram(E, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Expected counts: N times the exact probability of each bin (samples
    # beyond Emax are accounted for, since the probabilities are absolute)
    expected_counts = N * rf.planck_bin_probabilities(edges / kT_eV)
    chi2, ndf, z = rf.chi2_vs_expected(counts, expected_counts)
    print(f"Histogram: chi2/ndf = {chi2:.1f}/{ndf}")

    checks = [
        ("mean energy <E>/kT = 2.70118", abs(pull_mean) < 5.0,
         f"{mean_x:.5f} (pull {pull_mean:+.2f} sigma)"),
        ("second moment <E^2>/(kT)^2 = 10.3515", abs(pull_x2) < 5.0,
         f"{mean_x2:.4f} (pull {pull_x2:+.2f} sigma)"),
        ("spectrum shape (chi^2)", abs(z) < 5.0,
         f"chi2/ndf = {chi2:.1f}/{ndf} (z = {z:+.2f})"),
    ]
    ok = rf.print_checks(checks, title="(A) Photon sampler checks")

    # ---- plot: histogram + analytic spectrum ----
    plt.figure(figsize=(5, 2.5))
    plt.hist(
        E, bins=edges, histtype="stepfilled",
        linewidth=2, facecolor="black", edgecolor="black",
        alpha=0.3, label=f"Samples (N={N:,})"
    )
    plt.plot(centers, expected_counts, linewidth=2, color="red",
             label=r"Analytic $\propto E^2/(\exp(E/kT)-1)$")
    plt.xlabel("Photon energy E [eV]")
    plt.ylabel("Counts per bin")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved: {args.out}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
