import argparse
import numpy as np
import matplotlib.pyplot as plt

# Assumes this script lives next to main.py
from main import ThermalCompton


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", type=int, default=1_000_000, help="Number of samples")
    ap.add_argument("--bins", type=int, default=120, help="Histogram bins")
    ap.add_argument("--seed", type=int, default=12345, help="RNG seed")
    ap.add_argument("--out", type=str, default="thomson_angle_validation.pdf", help="Output figure")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    cos_th = np.empty(args.N, dtype=np.float64)
    phi = np.empty(args.N, dtype=np.float64)

    # Sample Thomson angles using the exact method in your class
    for i in range(args.N):
        c, s, p = ThermalCompton._sample_thomson_angles(rng)
        cos_th[i] = c
        phi[i] = p

    # ---- Expected shapes (scaled to histogram counts) ----
    # cos(theta) in [-1,1], Thomson shape: (1 + cos^2 theta)
    bins = args.bins

    # cos(theta) histogram
    counts_c, edges_c = np.histogram(cos_th, bins=bins, range=(-1.0, 1.0))
    centers_c = 0.5 * (edges_c[:-1] + edges_c[1:])
    binw_c = edges_c[1] - edges_c[0]

    shape_c = 1.0 + centers_c**2
    # normalize shape over [-1,1] so area=1
    area_c = np.trapz(shape_c, centers_c)
    shape_c /= area_c
    expected_counts_c = shape_c * counts_c.sum() * binw_c

    # phi histogram in [0, 2pi)
    two_pi = 2.0 * np.pi
    counts_p, edges_p = np.histogram(phi, bins=bins, range=(0.0, two_pi))
    centers_p = 0.5 * (edges_p[:-1] + edges_p[1:])
    # uniform expected counts per bin
    expected_counts_p = np.full_like(centers_p, counts_p.mean(), dtype=np.float64)

    # ---- Plot (single figure, two panels) ----
    plt.figure(figsize=(5, 5))

    # Left: cos(theta)
    ax1 = plt.subplot(2, 1, 1)
    ax1.hist(
        cos_th, bins=bins, range=(-1.0, 1.0), histtype="stepfilled", 
        linewidth=2, facecolor="black", edgecolor="black", alpha=0.3,
        label=f"Samples (N={args.N:,})"
        )
    ax1.plot(centers_c, expected_counts_c, linewidth=2, color='red', label=r"Expected $\propto 1+\cos^2\theta$")
    ax1.set_xlabel(r"$\cos\theta$")
    ax1.set_ylabel("Counts per bin")
    ax1.legend()

    # Right: phi
    ax2 = plt.subplot(2, 1, 2)
    ax2.hist(
        phi, bins=bins, range=(0.0, two_pi), histtype="stepfilled", 
        linewidth=2, facecolor="black", edgecolor="black", alpha=0.3, 
        label=f"Samples (N={args.N:,})"
        )
    ax2.plot(centers_p, expected_counts_p, linewidth=2, color='red', label=r"Expected uniform")
    ax2.set_xlabel(r"$\phi$ [rad]")
    ax2.set_ylabel("Counts per bin")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
