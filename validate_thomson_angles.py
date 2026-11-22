#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(B) Validation of the Thomson angular sampler.

Samples N proposal angles with ThermalCompton._sample_thomson_angles (the exact
routine used by the generator) and checks them against the Thomson
distribution dsigma_T/dOmega ∝ 1 + cos^2(theta):

  * cos(theta): moments <cos> = 0, <cos^2> = 2/5, and histogram shape (chi^2
    against the exact CDF F(c) = (c + 1)(c^2 - c + 4)/8)
  * phi: uniform in [0, 2*pi) (mean pi, chi^2 against a flat histogram)
  * sin^2 + cos^2 = 1

Prints a PASS/FAIL summary (exit status 1 on failure) and saves a two-panel plot.

Usage:
    python validate_thomson_angles.py [-N 1000000] [--out thomson_angle_validation.pdf]
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

# Assumes this script lives next to main.py
from main import ThermalCompton
import reference_formulas as rf


def main():
    ap = argparse.ArgumentParser(description="(B) Validate the Thomson angular sampler.")
    ap.add_argument("-N", "--N", type=int, default=1_000_000, help="Number of samples")
    ap.add_argument("--bins", type=int, default=120, help="Histogram bins")
    ap.add_argument("--seed", type=int, default=12345, help="RNG seed")
    ap.add_argument("--out", type=str, default="thomson_angle_validation.pdf", help="Output figure")
    args = ap.parse_args()

    N = int(args.N)
    rng = np.random.default_rng(args.seed)

    cos_th = np.empty(N, dtype=np.float64)
    sin_th = np.empty(N, dtype=np.float64)
    phi = np.empty(N, dtype=np.float64)

    # Sample Thomson angles with the exact routine used by the generator
    for i in range(N):
        cos_th[i], sin_th[i], phi[i] = ThermalCompton._sample_thomson_angles(rng)

    bins = args.bins
    two_pi = 2.0 * np.pi

    # ---- cos(theta): expected counts from the exact CDF ----
    edges_c = np.linspace(-1.0, 1.0, bins + 1)
    counts_c, _ = np.histogram(cos_th, bins=edges_c)
    centers_c = 0.5 * (edges_c[:-1] + edges_c[1:])
    expected_counts_c = N * np.diff(rf.thomson_cos_cdf(edges_c))
    chi2_c, ndf_c, z_c = rf.chi2_vs_expected(counts_c, expected_counts_c)

    # ---- phi: uniform in [0, 2*pi) ----
    edges_p = np.linspace(0.0, two_pi, bins + 1)
    counts_p, _ = np.histogram(phi, bins=edges_p)
    centers_p = 0.5 * (edges_p[:-1] + edges_p[1:])
    expected_counts_p = np.full(bins, N / bins, dtype=np.float64)
    chi2_p, ndf_p, z_p = rf.chi2_vs_expected(counts_p, expected_counts_p)

    # ---- moments (pull = deviation / standard error) ----
    def pull(sample, expected):
        mean = float(sample.mean())
        return mean, (mean - expected) / (float(sample.std(ddof=1)) / np.sqrt(N))

    mean_c, pull_c = pull(cos_th, 0.0)
    mean_c2, pull_c2 = pull(cos_th ** 2, 0.4)
    mean_p, pull_p = pull(phi, np.pi)
    norm_err = float(np.max(np.abs(sin_th ** 2 + cos_th ** 2 - 1.0)))

    print(f"N = {N:,} Thomson proposals")
    checks = [
        ("<cos(theta)> = 0", abs(pull_c) < 5.0, f"{mean_c:+.5f} (pull {pull_c:+.2f} sigma)"),
        ("<cos^2(theta)> = 2/5", abs(pull_c2) < 5.0, f"{mean_c2:.5f} (pull {pull_c2:+.2f} sigma)"),
        ("cos(theta) shape ∝ 1 + cos^2 (chi^2)", abs(z_c) < 5.0,
         f"chi2/ndf = {chi2_c:.1f}/{ndf_c} (z = {z_c:+.2f})"),
        ("<phi> = pi", abs(pull_p) < 5.0, f"{mean_p:.5f} (pull {pull_p:+.2f} sigma)"),
        ("phi uniform (chi^2)", abs(z_p) < 5.0, f"chi2/ndf = {chi2_p:.1f}/{ndf_p} (z = {z_p:+.2f})"),
        ("sin^2 + cos^2 = 1", norm_err < 1e-12, f"max |deviation| = {norm_err:.1e}"),
    ]
    ok = rf.print_checks(checks, title="(B) Thomson angle checks")

    # ---- plot (one figure, two panels) ----
    plt.figure(figsize=(5, 5))

    # Top: cos(theta)
    ax1 = plt.subplot(2, 1, 1)
    ax1.hist(
        cos_th, bins=edges_c, histtype="stepfilled",
        linewidth=2, facecolor="black", edgecolor="black", alpha=0.3,
        label=f"Samples (N={N:,})"
    )
    ax1.plot(centers_c, expected_counts_c, linewidth=2, color="red",
             label=r"Expected $\propto 1+\cos^2\theta$")
    ax1.set_xlabel(r"$\cos\theta$")
    ax1.set_ylabel("Counts per bin")
    ax1.legend()

    # Bottom: phi
    ax2 = plt.subplot(2, 1, 2)
    ax2.hist(
        phi, bins=edges_p, histtype="stepfilled",
        linewidth=2, facecolor="black", edgecolor="black", alpha=0.3,
        label=f"Samples (N={N:,})"
    )
    ax2.plot(centers_p, expected_counts_p, linewidth=2, color="red", label=r"Expected uniform")
    ax2.set_xlabel(r"$\phi$ [rad]")
    ax2.set_ylabel("Counts per bin")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved: {args.out}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
