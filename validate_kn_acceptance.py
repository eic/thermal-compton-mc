import argparse
import numpy as np
import matplotlib.pyplot as plt

from main import ThermalCompton, m_e_GeV


def kn_shape_vs_costh(costh: np.ndarray, kstar_GeV: float) -> np.ndarray:
    """
    Klein–Nishina "shape" factor used in your code (up to an overall constant):
        th_shape = 1 + cos^2(theta)
        x = 1 / (1 + (k*/m_e)(1 - cosθ))
        kn_shape = x^2 * [ 1 + cos^2θ + (x + 1/x - 2) ]
    This corresponds to the angular dependence of dσ_KN/dΩ for unpolarized photons.
    """
    c = costh
    x = 1.0 / (1.0 + (kstar_GeV / m_e_GeV) * (1.0 - c))
    #th = 1.0 + c * c
    kn = (1.0 + c * c + (x + 1.0 / x - 2.0)) * (x * x)
    # kn is already the "shape"; no need to divide by th here because we compare to KN itself
    return kn


def expected_x_hist_from_kn(kstar_GeV: float, x_edges: np.ndarray, ngrid: int = 200_000) -> np.ndarray:
    """
    Build expected histogram of x = kout*/kin* implied by KN angular distribution at fixed k*.
    We do this numerically:
      - sample a dense cosθ grid
      - compute KN pdf(cosθ) ∝ kn_shape(cosθ)
      - compute probability mass in each cosθ bin ~ pdf(cosθ_mid)*dc
      - map cosθ -> x(cosθ)
      - histogram x with weights=probability mass
    Returns probability per x-bin (sums to 1 over the x-range covered).
    """
    c_edges = np.linspace(-1.0, 1.0, ngrid + 1)
    c_mid = 0.5 * (c_edges[:-1] + c_edges[1:])
    dc = c_edges[1] - c_edges[0]

    # KN pdf over cosθ
    kn = kn_shape_vs_costh(c_mid, kstar_GeV)
    # Normalize over cosθ in [-1,1]
    norm = np.trapz(kn, c_mid)
    pdf_c = kn / norm

    # probability mass for each small cosθ interval
    w = pdf_c * dc

    # map cosθ -> x
    x_mid = 1.0 / (1.0 + (kstar_GeV / m_e_GeV) * (1.0 - c_mid))

    # histogram x with weights
    prob_per_bin, _ = np.histogram(x_mid, bins=x_edges, weights=w)
    # Ensure numerical normalization
    if prob_per_bin.sum() > 0:
        prob_per_bin /= prob_per_bin.sum()
    return prob_per_bin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", type=int, default=2_000_000, help="Number of Thomson proposals")
    ap.add_argument("--bins", type=int, default=120, help="Histogram bins")
    ap.add_argument("--seed", type=int, default=12345, help="RNG seed")
    ap.add_argument("--kstar-keV", type=float, default=10.0, help="Representative ERF photon energy k* [keV]")
    ap.add_argument("--out", type=str, default="kn_acceptance_validation.pdf", help="Output figure")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Convert keV -> GeV (1 GeV = 1e6 keV)
    kstar_GeV = float(args.kstar_keV) * 1e-6

    # Acceptance sampling using your exact routines
    accepted_costh = []
    accepted_x = []

    for _ in range(args.N):
        cos_th, sin_th, phi = ThermalCompton._sample_thomson_angles(rng)
        R, x = ThermalCompton._compton_over_thomson_ratio(kstar_GeV, cos_th)
        if rng.random() < R:
            accepted_costh.append(cos_th)
            accepted_x.append(x)

    accepted_costh = np.asarray(accepted_costh, dtype=np.float64)
    accepted_x = np.asarray(accepted_x, dtype=np.float64)

    n_acc = accepted_costh.size
    acc_frac = n_acc / args.N
    print(f"k* = {args.kstar_keV:.3f} keV = {kstar_GeV:.3e} GeV")
    print(f"Accepted: {n_acc:,} / {args.N:,}  (acceptance fraction = {acc_frac:.6f})")

    # ---- cosθ expected curve from KN ----
    bins = args.bins
    c_counts, c_edges = np.histogram(accepted_costh, bins=bins, range=(-1.0, 1.0))
    c_centers = 0.5 * (c_edges[:-1] + c_edges[1:])
    dc = c_edges[1] - c_edges[0]

    kn_c = kn_shape_vs_costh(c_centers, kstar_GeV)
    kn_c_norm = kn_c / np.trapz(kn_c, c_centers)  # pdf over cosθ
    expected_c_counts = kn_c_norm * c_counts.sum() * dc

    # ---- x expected histogram from KN (numerical transform) ----
    # Determine x-range for plotting from physics:
    # x_min at cosθ=-1, x_max at cosθ=+1 (x_max=1)
    x_min = 1.0 / (1.0 + (kstar_GeV / m_e_GeV) * (1.0 - (-1.0)))  # cosθ=-1
    x_max = 1.0  # cosθ=+1
    x_edges = np.linspace(x_min, x_max, bins + 1)
    x_counts, _ = np.histogram(accepted_x, bins=x_edges)

    prob_x = expected_x_hist_from_kn(kstar_GeV, x_edges, ngrid=200_000)
    expected_x_counts = prob_x * x_counts.sum()

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

    # Print mean x as a quick check
    print(f"Sample mean <x> = {accepted_x.mean():.8f}  (x in [~{x_min:.8f}, 1])")

    # ---- Plot: one figure, two panels ----
    plt.figure(figsize=(5, 5))

    ax1 = plt.subplot(2, 1, 1)
    ax1.hist(
        accepted_costh, bins=bins, range=(-1.0, 1.0), histtype="stepfilled", 
        linewidth=2, facecolor="black", edgecolor="black", alpha=0.3, 
        label=f"Accepted samples (N={n_acc:,})"
        )
    ax1.plot(c_centers, expected_c_counts, linewidth=2, color='red', label="KN prediction")
    ax1.set_xlabel(r"$\cos\theta$")
    ax1.set_ylabel("Counts per bin")
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    ax2.hist(
        accepted_x, bins=x_edges, histtype="stepfilled", 
        linewidth=2, facecolor="black", edgecolor="black", alpha=0.3, 
        label=f"Accepted samples (N={n_acc:,})"
        )
    ax2.plot(x_centers, expected_x_counts, linewidth=2, color='red', label="KN-implied prediction")
    ax2.set_xlabel(r"$x=k^*_{\mathrm{out}}/k^*_{\mathrm{in}}$")
    ax2.set_ylabel("Counts per bin")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
