import argparse
import numpy as np

from main import ThermalCompton, m_e_GeV, GeV_per_eV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Eb", type=float, default=18.0, help="Electron beam energy [GeV]")
    ap.add_argument("--T", type=float, default=300.0, help="Temperature [K]")
    ap.add_argument("--N", type=int, default=200_000, help="Number of accepted events to test")
    ap.add_argument("--seed", type=int, default=12345, help="RNG seed")
    ap.add_argument("--max-proposals", type=int, default=5_000_000,
                    help="Max Thomson proposals to attempt (safety)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # I_A doesn't matter for kinematics checks.
    tc = ThermalCompton(Eb_GeV=args.Eb, I_A=2.5, T_K=args.T, rng=rng)

    # Incoming electron: on-axis, on-momentum
    x = y = z = 0.0
    xp = yp = 0.0
    dp = 0.0
    Ee_in, pe_in = tc._sixd_to_four_electron(x, y, z, xp, yp, dp, tc.p0_GeV)
    pe_in = np.asarray(pe_in, dtype=np.float64)
    me2 = m_e_GeV * m_e_GeV

    # Accumulators for diagnostics
    n_acc = 0

    # Outgoing photon checks
    rel_E_minus_p_out = []   # |E-|p||/E
    m2_out_photon = []       # E^2 - p^2 (GeV^2)

    # Outgoing electron checks
    dm2_out_e = []           # (E^2 - p^2) - me^2 (GeV^2)

    # Reconstructed incoming photon (from conservation) checks
    rel_E_minus_p_in = []
    m2_in_photon = []
    Ein_inferred_eV = []     # inferred incoming photon energy (eV), sanity

    # Generate accepted events (KN accepted) using your trial routine
    for _ in range(args.max_proposals):
        accepted, e_out, g_out = tc._compton_scatter_one_trial(
            (Ee_in, pe_in), tc.photon_energy_sampler_eV, rng
        )
        if not accepted:
            continue

        Ee_out, pe_out = e_out
        Eg_out, pg_out = g_out
        pe_out = np.asarray(pe_out, dtype=np.float64)
        pg_out = np.asarray(pg_out, dtype=np.float64)

        # ---- Outgoing photon masslessness ----
        p_out = float(np.linalg.norm(pg_out))
        m2g_out = float(Eg_out * Eg_out - np.dot(pg_out, pg_out))
        rel_out = abs(float(Eg_out) - p_out) / max(float(Eg_out), 1e-300)

        m2_out_photon.append(m2g_out)
        rel_E_minus_p_out.append(rel_out)

        # ---- Outgoing electron mass shell ----
        m2e_out = float(Ee_out * Ee_out - np.dot(pe_out, pe_out))
        dm2e = m2e_out - me2
        dm2_out_e.append(dm2e)

        # ---- Reconstruct incoming photon from 4-momentum conservation ----
        # From your construction: Pe_out = Pe_in + Pg_in - Pg_out
        # => Pg_in = Pe_out - Pe_in + Pg_out
        Eg_in = float(Ee_out - Ee_in + Eg_out)
        pg_in = pe_out - pe_in + pg_out

        p_in = float(np.linalg.norm(pg_in))
        m2g_in = float(Eg_in * Eg_in - np.dot(pg_in, pg_in))
        rel_in = abs(Eg_in - p_in) / max(Eg_in, 1e-300)

        m2_in_photon.append(m2g_in)
        rel_E_minus_p_in.append(rel_in)
        Ein_inferred_eV.append(Eg_in / GeV_per_eV)

        n_acc += 1
        if n_acc >= args.N:
            break

    if n_acc < args.N:
        raise RuntimeError(
            f"Only collected {n_acc} accepted events out of requested {args.N}. "
            f"Increase --max-proposals."
        )

    # Convert to arrays for statistics
    rel_E_minus_p_out = np.asarray(rel_E_minus_p_out)
    m2_out_photon = np.asarray(m2_out_photon)
    dm2_out_e = np.asarray(dm2_out_e)
    rel_E_minus_p_in = np.asarray(rel_E_minus_p_in)
    m2_in_photon = np.asarray(m2_in_photon)
    Ein_inferred_eV = np.asarray(Ein_inferred_eV)

    def stats(name, arr):
        return (f"{name}: mean={arr.mean():.3e}, rms={arr.std():.3e}, "
                f"max={arr.max():.3e}")

    print("=== Kinematics consistency checks ===")
    print(f"Beam energy Eb = {args.Eb} GeV, T = {args.T} K")
    print(f"Accepted events tested: {n_acc:,}")
    print("")
    print("Outgoing photon masslessness (should be ~0):")
    print(stats("|E-|p||/E (out)", rel_E_minus_p_out))
    print(stats("E^2-p^2 [GeV^2] (out)", np.abs(m2_out_photon)))
    print("")
    print("Outgoing electron mass-shell (should be ~0):")
    print(stats("|(E^2-p^2)-m_e^2| [GeV^2]", np.abs(dm2_out_e)))
    print("")
    print("Incoming photon reconstructed from conservation (should be massless):")
    print(stats("|E-|p||/E (in)", rel_E_minus_p_in))
    print(stats("E^2-p^2 [GeV^2] (in)", np.abs(m2_in_photon)))
    print("")
    print("Diagnostic: inferred incoming photon energy scale (eV):")
    print(f"E_in inferred [eV]: mean={Ein_inferred_eV.mean():.4f}, "
          f"median={np.median(Ein_inferred_eV):.4f}, "
          f"99%={np.quantile(Ein_inferred_eV, 0.99):.4f}, "
          f"max={Ein_inferred_eV.max():.4f}")
    print("=====================================")


if __name__ == "__main__":
    main()
