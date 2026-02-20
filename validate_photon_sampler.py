import argparse
import numpy as np
import matplotlib.pyplot as plt

# Assumes this script lives next to main.py
from main import ThermalCompton, kB_eV


def planck_photon_number_pdf(E_eV: np.ndarray, kT_eV: float) -> np.ndarray:
    """
    Photon-number spectrum (unnormalized):
        f(E) ∝ E^2 / (exp(E/kT) - 1)
    Returns an array with the same shape as E_eV.
    """
    x = E_eV / kT_eV
    # Use expm1 for numerical stability for small x
    denom = np.expm1(x)
    pdf = np.where(denom > 0.0, (E_eV**2) / denom, 0.0)
    return pdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=300.0, help="Temperature [K]")
    ap.add_argument("-N", type=int, default=300_000, help="Number of samples")
    ap.add_argument("--bins", type=int, default=120, help="Histogram bins")
    ap.add_argument("--Emax", type=float, default=0.5, help="Max energy on plot [eV]")
    ap.add_argument("--nmax", type=int, default=2000, help="Mixture truncation nmax for sampler")
    ap.add_argument("--seed", type=int, default=12345, help="RNG seed")
    ap.add_argument("--out", type=str, default="photon_sampler_validation.pdf", help="Output figure")
    args = ap.parse_args()

    T_K = float(args.T)
    rng = np.random.default_rng(args.seed)

    # We only need the sampler; Eb/I are irrelevant here
    tc = ThermalCompton(
        Eb_GeV=10.0,
        I_A=2.5,
        T_K=T_K,
        nmax_sampler=args.nmax,
        rng=rng,
    )
    sampler = tc.photon_energy_sampler_eV

    # Sample energies [eV]
    E = sampler.sample_eV(size=args.N)

    # Compute sample mean and expected mean
    kT_eV = kB_eV * T_K
    mean_sample = float(E.mean())
    mean_expected = 2.701178 * kT_eV  # π^4/(30 ζ(3)) * kT for photon-number spectrum

    print(f"T = {T_K:.1f} K, kT = {kT_eV:.6f} eV")
    print(f"Sample mean <E> = {mean_sample:.6f} eV")
    print(f"Expected <E> ≈ 2.701 kT = {mean_expected:.6f} eV")
    print(f"Ratio (sample/expected) = {mean_sample/mean_expected:.6f}")

    # Plot range
    Emax = float(args.Emax)
    # E_plot = E[(E >= 0.0) & (E <= Emax)]
    E_plot = E  # keep all samples; histogram range will ignore out-of-range values

    # Histogram (counts)
    counts, edges = np.histogram(E_plot, bins=args.bins, range=(0.0, Emax))
    centers = 0.5 * (edges[:-1] + edges[1:])
    binw = edges[1] - edges[0]

    # Analytic curve (scaled to histogram counts)
    pdf = planck_photon_number_pdf(centers, kT_eV)
    # Normalize pdf over the plotted range so area=1 within [0,Emax]
    area = np.trapz(pdf, centers)
    pdf_norm = pdf / area if area > 0 else pdf
    curve_counts = pdf_norm * counts.sum() * binw

    # Single plot: histogram + analytic shape
    plt.figure(figsize=(5, 2.5))
    plt.hist(
        E_plot, bins=args.bins, range=(0.0, Emax), histtype="stepfilled", 
        linewidth=2, facecolor="black", edgecolor="black",
        alpha=0.3, label=f"Samples (N={len(E_plot):,})"
        )
    plt.plot(centers, curve_counts, linewidth=2, color='red', label=r"Analytic $\propto E^2/(\exp(E/kT)-1)$")

    plt.xlabel("Photon energy E [eV]")
    plt.ylabel("Counts per bin")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()