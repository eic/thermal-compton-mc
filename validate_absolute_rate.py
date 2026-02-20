import argparse
import numpy as np
import matplotlib.pyplot as plt

from main import ThermalCompton, e_SI, sigma_T_m2, c_SI


def estimate_section_rate_Hz(Eb_GeV: float, I_A: float, T_K: float, L_m: float,
                             n_macro: int, n_trials_per_macro: int, seed: int) -> tuple[float, float]:
    """
    Returns:
      R_mc_Hz   : Monte Carlo estimated Compton event rate in the section [Hz]
      R_th_Hz   : Simple analytic estimate using Thomson cross section [Hz]
                 R_th = (I/e) * rho(T) * sigma_T * L
    Note:
      Set dp_threshold=0.0 so ALL accepted scatters contribute to the rate.
    """
    rng = np.random.default_rng(seed)
    tc = ThermalCompton(
        Eb_GeV=Eb_GeV,
        I_A=I_A,
        T_K=T_K,
        n_trials_per_macro=n_trials_per_macro,
        rng=rng,
    )

    # Simple "cold" beam: on-axis, on-momentum macros (kinematics don't matter for total rate)
    macros = [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(n_macro)]

    events = tc.generate_section_events_6d(
        macros_6d=macros,
        L_m=L_m,
        section_start_s_m=0.0,
        dp_threshold=0.0,  # keep all accepted events for total rate estimate
    )

    R_mc_Hz = float(sum(ev["w_Hz"] for ev in events))
    rho = tc.rho_m3
    R_th_Hz = (I_A / e_SI) * rho * sigma_T_m2 * L_m
    return R_mc_Hz, float(R_th_Hz)


def tau_min_hours(T_K: float) -> float:
    """
    Rough minimum lifetime (f=1) assuming every Compton collision leads to loss:
        tau_min = 1 / (rho(T) * sigma_T * c)
    This is independent of beam current and ring geometry (using v~c).
    """
    rho = ThermalCompton._photon_density_blackbody(T_K)  # [1/m^3]
    tau_s = 1.0 / (rho * sigma_T_m2 * c_SI)
    return tau_s / 3600.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Eb", type=float, default=18.0, help="Reference electron energy [GeV] (only affects kinematics)")
    ap.add_argument("--T0", type=float, default=300.0, help="Baseline temperature [K]")
    ap.add_argument("--I0", type=float, default=2.5, help="Baseline current [A]")
    ap.add_argument("--L0", type=float, default=1.0, help="Baseline section length [m]")
    ap.add_argument("--n-macro", type=int, default=2000, help="Number of macro-electrons")
    ap.add_argument("--K", type=int, default=200, help="Trials per macro-electron")
    ap.add_argument("--seed", type=int, default=12345, help="Base RNG seed")
    ap.add_argument("--plot", action="store_true", help="Save a scaling plot")
    ap.add_argument("--out", type=str, default="absolute_rate_scaling.png", help="Output plot filename")
    args = ap.parse_args()

    Eb = float(args.Eb)
    T0 = float(args.T0)
    I0 = float(args.I0)
    L0 = float(args.L0)

    print("=== Absolute rate scaling checks ===")
    print(f"Baseline: Eb={Eb} GeV, T0={T0} K, I0={I0} A, L0={L0} m")
    print(f"Sampling: n_macro={args.n_macro}, K={args.K}")
    print("")

    # Baseline rate
    R0_mc, R0_th = estimate_section_rate_Hz(Eb, I0, T0, L0, args.n_macro, args.K, args.seed)
    print("Baseline section rate:")
    print(f"  R_MC = {R0_mc:.6e} Hz")
    print(f"  R_Th = {R0_th:.6e} Hz   (Thomson estimate)")
    print(f"  Ratio R_MC/R_Th = {R0_mc/R0_th:.6f}")
    print("")

    # --- I scaling (linear) ---
    I_factors = np.array([0.5, 1.0, 2.0], dtype=float)
    R_I = []
    print("Current scaling (expect linear):")
    for j, f in enumerate(I_factors):
        R_mc, R_th = estimate_section_rate_Hz(Eb, I0*f, T0, L0, args.n_macro, args.K, args.seed + 10_000 + j)
        R_I.append(R_mc)
        print(f"  I = {I0*f:.4g} A  (x{f:g}):  R_MC/R0 = {R_mc/R0_mc:.6f}   expected = {f:g}")
    print("")

    # --- L scaling (linear) ---
    L_factors = np.array([0.5, 1.0, 2.0], dtype=float)
    R_L = []
    print("Length scaling (expect linear):")
    for j, f in enumerate(L_factors):
        R_mc, R_th = estimate_section_rate_Hz(Eb, I0, T0, L0*f, args.n_macro, args.K, args.seed + 20_000 + j)
        R_L.append(R_mc)
        print(f"  L = {L0*f:.4g} m  (x{f:g}):  R_MC/R0 = {R_mc/R0_mc:.6f}   expected = {f:g}")
    print("")

    # --- T scaling (photon density ~ T^3) ---
    # Use modest variation around 300 K so KN corrections remain negligible.
    T_factors = np.array([0.8, 1.0, 1.2], dtype=float)
    R_T = []
    print("Temperature scaling (expect ~T^3 via photon density):")
    for j, f in enumerate(T_factors):
        R_mc, R_th = estimate_section_rate_Hz(Eb, I0, T0*f, L0, args.n_macro, args.K, args.seed + 30_000 + j)
        R_T.append(R_mc)
        print(f"  T = {T0*f:.4g} K (x{f:g}):  R_MC/R0 = {R_mc/R0_mc:.6f}   expected = {f**3:.6f}")
    print("")

    # --- tau_min check ---
    tau_h = tau_min_hours(T0)
    print(f"Rough minimum lifetime at T={T0:.1f} K (f=1):")
    print(f"  tau_min = {tau_h:.3f} hours  from  1/(n_ph * sigma_T * c)")
    print("==================================")

    # Optional plot
    if args.plot:
        plt.figure()
        # Normalize all to baseline
        plt.plot(I_factors, np.array(R_I)/R0_mc, marker="o", linestyle="-", label="I scaling (measured)")
        plt.plot(I_factors, I_factors, linestyle="--", label="I scaling (expected)")

        plt.plot(L_factors, np.array(R_L)/R0_mc, marker="o", linestyle="-", label="L scaling (measured)")
        plt.plot(L_factors, L_factors, linestyle="--", label="L scaling (expected)")

        plt.plot(T_factors, np.array(R_T)/R0_mc, marker="o", linestyle="-", label="T scaling (measured)")
        plt.plot(T_factors, T_factors**3, linestyle="--", label=r"T scaling (expected $\propto T^3$)")

        plt.xlabel("Scale factor")
        plt.ylabel(r"Normalized rate $R_{\rm MC}/R_0$")
        plt.title("Absolute rate scaling checks")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out, dpi=200)
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
