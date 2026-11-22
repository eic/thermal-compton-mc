#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(C) Validation of the Klein-Nishina accept/reject step at a fixed ERF photon energy k*.

Proposes N scattering angles from the Thomson distribution and accepts each with
probability (dsigma_KN/dOmega)/(dsigma_T/dOmega), using the exact routines of
ThermalCompton. The accepted sample must follow Klein-Nishina:

  * acceptance fraction = sigma_KN(k*)/sigma_T (analytic total cross section)
  * cos(theta*) histogram ∝ dsigma_KN/dcos (chi^2, exact bin integrals)
  * x = k*_out/k* histogram and mean <x> (chi^2 and pull)

Prints a PASS/FAIL summary (exit status 1 on failure) and saves a two-panel plot.

Usage:
    python validate_kn_acceptance.py [--kstar-keV 10] [-N 2000000] [--out kn_acceptance_validation.pdf]
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from main import ThermalCompton, m_e_GeV
import reference_formulas as rf


def main():
    ap = argparse.ArgumentParser(description="(C) Validate the Klein-Nishina acceptance at fixed k*.")
    ap.add_argument("-N", "--N", type=int, default=2_000_000, help="Number of Thomson proposals")
    ap.add_argument("--bins", type=int, default=120, help="Histogram bins")
    ap.add_argument("--seed", type=int, default=12345, help="RNG seed")
    ap.add_argument("--kstar-keV", type=float, default=10.0,
                    help="Representative ERF photon energy k* [keV]")
    ap.add_argument("--out", type=str, default="kn_acceptance_validation.pdf", help="Output figure")
    args = ap.parse_args()

    N = int(args.N)
    rng = np.random.default_rng(args.seed)

    # keV -> GeV (1 GeV = 1e6 keV) and eps = k*/(m_e c^2)
    kstar_GeV = float(args.kstar_keV) * 1e-6
    eps = kstar_GeV / m_e_GeV

    # ---- Thomson proposal + KN acceptance with the generator's routines ----
    accepted_costh = []
    accepted_x = []
    for _ in range(N):
        cos_th, _sin_th, _phi = ThermalCompton._sample_thomson_angles(rng)
        R, x = ThermalCompton._compton_over_thomson_ratio(kstar_GeV, cos_th)
        if rng.random() < R:
            accepted_costh.append(cos_th)
            accepted_x.append(x)

    accepted_costh = np.asarray(accepted_costh, dtype=np.float64)
    accepted_x = np.asarray(accepted_x, dtype=np.float64)
    n_acc = accepted_costh.size

    # ---- acceptance fraction vs analytic sigma_KN/sigma_T ----
    acc_frac = n_acc / N
    acc_expected = rf.kn_total_over_thomson(eps)
    acc_err = np.sqrt(acc_expected * (1.0 - acc_expected) / N)
    pull_acc = (acc_frac - acc_expected) / acc_err
    print(f"k* = {args.kstar_keV:.3f} keV = {kstar_GeV:.3e} GeV  (k*/m_e = {eps:.5f})")
    print(f"Accepted: {n_acc:,} / {N:,}  (acceptance fraction = {acc_frac:.6f}, "
          f"sigma_KN/sigma_T = {acc_expected:.6f})")

    bins = args.bins

    # ---- cos(theta*): exact KN bin probabilities ----
    c_edges = np.linspace(-1.0, 1.0, bins + 1)
    c_counts, _ = np.histogram(accepted_costh, bins=c_edges)
    c_centers = 0.5 * (c_edges[:-1] + c_edges[1:])
    expected_c_counts = n_acc * rf.kn_cos_bin_probabilities(c_edges, eps)
    chi2_c, ndf_c, z_c = rf.chi2_vs_expected(c_counts, expected_c_counts)

    # ---- x = k*_out/k*: x(c) = 1/(1 + eps (1 - c)) is monotonic, so an x-bin
    # [x_a, x_b] is the cos-bin [c(x_a), c(x_b)] with c(x) = 1 - (1/x - 1)/eps ----
    x_min = 1.0 / (1.0 + 2.0 * eps)   # back-scattering, cos(theta*) = -1
    x_max = 1.0                       # forward scattering, cos(theta*) = +1
    x_edges = np.linspace(x_min, x_max, bins + 1)
    x_counts, _ = np.histogram(accepted_x, bins=x_edges)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    c_of_x_edges = np.clip(1.0 - (1.0 / x_edges - 1.0) / eps, -1.0, 1.0)
    expected_x_counts = n_acc * rf.kn_cos_bin_probabilities(c_of_x_edges, eps)
    chi2_x, ndf_x, z_x = rf.chi2_vs_expected(x_counts, expected_x_counts)

    # ---- mean <x> vs KN quadrature ----
    ct, wt = rf.gauss_legendre(-1.0, 1.0, 128)
    dsig = rf.kn_dsigma_dcos_over_sigmaT(ct, eps)
    x_of_ct = 1.0 / (1.0 + eps * (1.0 - ct))
    mean_x_expected = float(np.sum(wt * dsig * x_of_ct) / np.sum(wt * dsig))
    mean_x = float(accepted_x.mean())
    pull_x = (mean_x - mean_x_expected) / (float(accepted_x.std(ddof=1)) / np.sqrt(n_acc))
    print(f"Sample mean <x> = {mean_x:.8f}  (expected {mean_x_expected:.8f}; x in [{x_min:.8f}, 1])")

    checks = [
        ("acceptance fraction = sigma_KN/sigma_T", abs(pull_acc) < 5.0,
         f"{acc_frac:.6f} vs {acc_expected:.6f} (pull {pull_acc:+.2f} sigma)"),
        ("cos(theta*) shape (chi^2)", abs(z_c) < 5.0,
         f"chi2/ndf = {chi2_c:.1f}/{ndf_c} (z = {z_c:+.2f})"),
        ("x = k*_out/k* shape (chi^2)", abs(z_x) < 5.0,
         f"chi2/ndf = {chi2_x:.1f}/{ndf_x} (z = {z_x:+.2f})"),
        ("mean <x>", abs(pull_x) < 5.0, f"pull {pull_x:+.2f} sigma"),
    ]
    ok = rf.print_checks(checks, title="(C) Klein-Nishina acceptance checks")

    # ---- plot: one figure, two panels ----
    plt.figure(figsize=(5, 5))

    ax1 = plt.subplot(2, 1, 1)
    ax1.hist(
        accepted_costh, bins=c_edges, histtype="stepfilled",
        linewidth=2, facecolor="black", edgecolor="black", alpha=0.3,
        label=f"Accepted samples (N={n_acc:,})"
    )
    ax1.plot(c_centers, expected_c_counts, linewidth=2, color="red", label="KN prediction")
    ax1.set_xlabel(r"$\cos\theta$")
    ax1.set_ylabel("Counts per bin")
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    ax2.hist(
        accepted_x, bins=x_edges, histtype="stepfilled",
        linewidth=2, facecolor="black", edgecolor="black", alpha=0.3,
        label=f"Accepted samples (N={n_acc:,})"
    )
    ax2.plot(x_centers, expected_x_counts, linewidth=2, color="red", label="KN-implied prediction")
    ax2.set_xlabel(r"$x=k^*_{\mathrm{out}}/k^*_{\mathrm{in}}$")
    ax2.set_ylabel("Counts per bin")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved: {args.out}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
