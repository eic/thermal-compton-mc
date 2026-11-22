# ThermalCompton: thermal (blackbody) Compton event generator for tracking

`ThermalCompton` is a compact Monte Carlo generator of **Compton scattering of
ultra-relativistic beam electrons on the thermal (blackbody) photons** of the
vacuum chamber. It is designed to be called from a multi-turn tracking code, ring
section by ring section: for every incoming macro-electron it returns the
scattered electron and photon in "6D-like" coordinates, together with an
**absolute event weight in Hz**. Summing the weights of the electrons that are
subsequently lost gives the loss rate, and hence the beam lifetime.

The algorithm follows H. Burkhardt (CERN SL/Note 93-73): Thomson proposal
sampling of the scattering angle in the electron rest frame with Klein–Nishina
acceptance, explicit Lorentz boosts and rotations, and four-momentum
conservation. It includes the relativistic flux factor in the sampling of the
thermal photon direction (see [Physics model](#physics-model)).

## Contents

- [Requirements](#requirements)
- [Repository contents](#repository-contents)
- [Quick start](#quick-start)
- [Physics model](#physics-model)
- [API reference](#api-reference)
- [Absolute weighting and lifetime](#absolute-weighting-and-lifetime)
- [Integration with a tracking code](#integration-with-a-tracking-code)
- [Validation and tests](#validation-and-tests)
- [Performance](#performance)
- [References](#references)
- [License](#license)
- [Citation](#citation)

## Requirements

- Python 3.10 or newer (the type hints use `X | None`)
- `numpy` (generator)
- `matplotlib` (validation plots only)
- `pytest` (tests only)

Tested with Python 3.10 / NumPy 1.26 / Matplotlib 3.8 and with Python 3.12 /
NumPy 2.3 / Matplotlib 3.10.

```bash
pip install -r requirements.txt
```

There is nothing to build or install: keep `main.py` next to your script (or on
your `PYTHONPATH`) and import the class with `from main import ThermalCompton`.

## Repository contents

| File | Content |
|---|---|
| `main.py` | The `ThermalCompton` event generator (sampling, boosts/rotations, weighting, 6D-like output) |
| `reference_formulas.py` | Independent analytic and quadrature reference values (Klein–Nishina cross section, collision averages, Compton edge) used by the validations and tests; also handy for quick rate estimates |
| `validate_photon_sampler.py` | (A) Planck photon energy sampler: moments and spectrum |
| `validate_thomson_angles.py` | (B) Thomson angular sampler: `cosθ` shape and `φ` uniformity |
| `validate_kn_acceptance.py` | (C) Klein–Nishina acceptance at a fixed rest-frame photon energy `k*` |
| `validate_kinematics.py` | (D) Complete scatterings: kinematics, photon direction, mean energy |
| `validate_absolute_rate.py` | (E) Absolute rate normalization, scaling with I, L, T, and `τ_min` |
| `tests/test_thermal_compton.py` | Automated test suite (pytest) |
| `requirements.txt` | Python dependencies |

## Quick start

Generate the events of one 1 m section for a cold, on-axis beam (no lattice
needed):

```python
import numpy as np
from main import ThermalCompton

tc = ThermalCompton(Eb_GeV=18.0, I_A=0.227, T_K=300.0, n_trials_per_macro=200,
                    rng=np.random.default_rng(12345))   # seeded => reproducible

# Macro-electron ensemble (x, y, z, x', y', dp): here a cold, on-axis beam
macros_6d = [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)] * 2000

events = tc.generate_section_events_6d(macros_6d, L_m=1.0, section_start_s_m=0.0,
                                       dp_threshold=1e-3)

R_tail_Hz = sum(ev["w_Hz"] for ev in events)
print(f"Collisions with |dp| >= 1e-3 in this 1 m section: {R_tail_Hz:.4e} Hz")
# -> 4.1055e+04 Hz (all collisions: 5.10e+04 Hz); about 10 s on a laptop
```

`dp_threshold` is a **numerical thinning threshold**: it decides which scattered
electrons are worth tracking. Set it well below the momentum acceptance of the
ring; it is not the acceptance itself.

## Physics model

One *trial collision* of a macro-electron proceeds in five steps:

1. **Photon energy.** The thermal photon energy `k` is drawn from the Planck
   photon-number spectrum `p(k) ∝ k² / (exp(k/kT) − 1)`, written as a series
   of Gamma distributions (truncated after `nmax_sampler` terms);
   `⟨k⟩ = 2.701 kT`.
2. **Photon direction.** In an isotropic photon gas, a photon moving at the
   angle `α` to the electron velocity collides at a rate proportional to the
   relativistic flux factor `1 − β cos α`: the electron runs into head-on
   photons, while photons moving along with it hardly ever reach it. Hence
   `cos α` is sampled from `(1 − β cos α)/2`, with a uniform azimuth. The
   isotropic average of the flux factor is exactly 1, so the normalization of
   the rate is unchanged.
3. **Electron rest frame (ERF).** The photon is boosted to the ERF (energy
   `k* = γ k (1 − β cos α)`) and rotated onto `+z`.
4. **Scattering angle.** `θ*` is proposed from the Thomson distribution
   `dσ_T/dΩ ∝ 1 + cos²θ*` and accepted with probability
   `(dσ_KN/dΩ)/(dσ_T/dΩ) ≤ 1`, so that accepted events follow Klein–Nishina.
   The accepted fraction is `σ_KN(k*)/σ_T`. The scattered photon energy is
   `k*' = k* / (1 + (k*/m_e)(1 − cos θ*))`.
5. **Back to the lab.** The photon is rotated back and boosted to the lab; the
   scattered electron follows from four-momentum conservation.

The boosts use `γ = E/m_e` directly instead of `1/sqrt(1 − β²)`, which would lose
about `log10(γ²)` significant digits (7 digits at 18 GeV) to cancellation, so
the scattered photon energies are accurate to double precision at any beam
energy.

### Flux factor and Burkhardt's 1993 note

The 1993 note generates the thermal photon directions isotropically in the lab
frame, without the flux factor `1 − β cos α`. Without it, the mean energy
transferred per collision is `γ²⟨k⟩` instead of the textbook value
`(4/3) γ²⟨k⟩` (Blumenthal & Gould 1970; Rybicki & Lightman, Ch. 7), and the
tail of large energy losses, which drives beam losses, is underestimated.

`flux_weighting=True` (the default) includes the flux factor.
`flux_weighting=False` reproduces the isotropic sampling of the note, and its
numbers, for benchmarking:

| LEP benchmark, T = 297 K | Burkhardt 1993 | `flux_weighting=False` | `flux_weighting=True` |
|---|---|---|---|
| 45.6 GeV: `σ_C/σ_T` | 0.9767 | 0.9772 | 0.9698 |
| 45.6 GeV: mean photon energy | 0.52 GeV | 0.515 GeV | 0.684 GeV |
| 45.6 GeV: fraction of collisions with `ΔE/E > 1 %` | 0.372 | 0.368 | 0.488 |
| 90 GeV: `σ_C/σ_T` | 0.9558 | 0.9571 | 0.9436 |
| 90 GeV: mean photon energy | 1.9 GeV | 1.894 GeV | 2.509 GeV |
| 90 GeV: fraction of collisions with `ΔE/E > 1 %` | 0.544 | 0.535 | 0.660 |

For a given energy acceptance, the flux factor raises the fraction of
collisions that lead to a loss (by 20–30 % in this example) and shortens the
computed lifetime accordingly. Results produced with versions of this code
before the flux factor was introduced correspond to `flux_weighting=False`.

## API reference

### Constructor

```python
ThermalCompton(
    Eb_GeV,                         # reference beam energy [GeV] (defines p0)
    I_A,                            # beam current [A] (absolute normalization)
    T_K,                            # photon gas temperature [K]
    n_trials_per_macro=100,         # trial collisions K per macro and section call
    nmax_sampler=2000,              # series truncation of the Planck sampler
    rng=None,                       # numpy.random.Generator (seed it for reproducibility)
    photon_energy_sampler_eV=None,  # optional callable returning one photon energy [eV]
    dp_threshold=1e-3,              # default thinning threshold on |dp|
    flux_weighting=True,            # flux factor (1 - beta cos a) in photon directions
)
```

| Parameter | Meaning |
|---|---|
| `Eb_GeV` | Reference energy; sets `p0 = sqrt(Eb² − m_e²)` used in `dp = p/p0 − 1`. Must exceed `m_e c²`. |
| `I_A` | Beam current (≥ 0); converted to the electron flow `I/e` for the weights. |
| `T_K` | Temperature (> 0); sets the photon density `ρ(T)` and the photon spectrum. |
| `n_trials_per_macro` | Number `K` of trial collisions per macro-electron per section call. A larger `K` lowers the statistical error; the CPU time scales as `N_macro × K`. |
| `nmax_sampler` | Terms kept in the series behind the Planck sampler (neglected probability ≈ 1e-7 for 2000). |
| `rng` | Random generator used for all sampling. |
| `photon_energy_sampler_eV` | Replacement photon energy sampler (no arguments, returns eV). The weights still use the blackbody density `ρ(T_K)`. |
| `dp_threshold` | Used when `generate_section_events_6d` is called without `dp_threshold`. |
| `flux_weighting` | `True`: physical flux factor. `False`: isotropic directions, as in Burkhardt (1993), for benchmarking only. |

Derived attributes: `p0_GeV` (reference momentum [GeV]) and `rho_m3` (photon
number density [1/m³]; 5.477 × 10¹⁴ m⁻³ at 300 K).

### `generate_section_events_6d(macros_6d, L_m, section_start_s_m=0.0, dp_threshold=None)`

Generates the events of **one** ring section of length `L_m` [m].

- `macros_6d`: incoming macro-electrons `(x, y, z, x', y', dp)`, as a list of
  tuples, an `(N, 6)` NumPy array or a generator. Each macro represents `1/N` of
  the beam current.
- `section_start_s_m`: section start [m]; only used to tag the events with the
  interaction position `s_m = section_start_s_m + U(0, L_m)`.
- `dp_threshold`: scattered electrons with `|dp| ≥ dp_threshold` are "tail"
  events (`dp` of the outgoing electron, including its incoming `dp`). `None`
  uses the constructor value; `0.0` keeps every collision (total rate).

For every macro-electron, `K` trial collisions are simulated and **exactly one
representative event is returned**:

- if `n_pass ≥ 1` trials produced a tail event, one of them is chosen at random
  and receives the weight `w_Hz = n_pass × w_trial`, so it stands for all of
  them. Rates and any property of the scattered electron (such as "is it
  lost?") remain unbiased.
- otherwise an accepted scattering is still returned, so that exactly one
  particle per macro can be tracked, but with `w_Hz = 0` (its `|dp|` may be
  below the threshold).

Each event is a dictionary:

| Key | Content |
|---|---|
| `s_m` | interaction position [m] |
| `w_Hz` | absolute rate weight [Hz] |
| `e6d_out` | scattered electron `(x, y, z, x', y', dp)` |
| `g6d_out` | scattered photon `(x, y, z, x', y', dp)`, "6D-like" (see below) |
| `photon_E_GeV` | scattered photon energy [GeV] |

The positions `(x, y, z)` are those of the incoming macro-electron (point-like
interaction).

### `events_to_macros_cols(events, particle="e", skip_none=True)`

Converts the event list into columns

```python
w, (x, xp, y, yp, z, dp) = tc.events_to_macros_cols(events, particle="e")
```

with `particle="e"` (scattered electrons) or `"g"` (scattered photons). Note the
column order `(x, x', y, y', z, dp)`. Records without coordinates are skipped,
or kept as rows of NaN with `skip_none=False`.

### Coordinates and conventions

- `x' = p_x/p_z` and `y' = p_y/p_z` are geometric slopes [rad]; `dp = p/p0 − 1`
  with `p0` set by `Eb_GeV`.
- Xsuite uses the canonical momenta `px = P_x/P0`, `py = P_y/P0` and
  `delta = (P − P0)/P0 = dp` (MAD-X uses the same `px`, `py` but `PT` instead
  of `delta`). Convert with `x' = px / sqrt((1 + delta)² − px² − py²)` (same for
  `y'`), and back with `px = x' (1 + delta) / sqrt(1 + x'² + y'²)`.
- `z` (or Xsuite's `zeta`) is passed through unchanged.
- Photon "6D-like" coordinates: slopes `p_x/p_z`, `p_y/p_z` of the photon
  direction and `dp = |p_γ|/p0 − 1`, i.e. the photon momentum relative to the
  electron reference momentum.

## Absolute weighting and lifetime

A beam current `I` corresponds to `I/e` electrons per second passing any point
of the ring. In a photon gas of density
`ρ(T) = 16π ζ(3) (k_B T / hc)³` (5.477 × 10¹⁴ m⁻³ at 300 K), an
ultra-relativistic electron undergoes `ρ(T)⟨σ⟩` collisions per metre, where
`⟨σ⟩` is the Klein–Nishina cross section averaged over the photon spectrum and
the flux factor. The collision rate in a section of length `L` is

```
R = (I/e) · ρ(T) · ⟨σ⟩ · L                      [Hz]
```

With `N` macro-electrons and `K` trials per macro, each trial carries the
Thomson-normalized weight

```
w_trial = (I / (e N)) · ρ(T) · σ_T · (L / K)    [Hz]
```

and is accepted with probability `(dσ_KN/dΩ)/(dσ_T/dΩ)`, so the accepted trials
reproduce `R` on average (`⟨σ⟩/σ_T` = 0.9875 at 18 GeV and 300 K).

- Sum of `w_Hz` over all events: collision rate (with `dp_threshold=0`) or rate
  of collisions with `|dp| ≥ dp_threshold`.
- Sum of `w_Hz` over the events whose electron is later lost: loss rate `R_loss`.
- Lifetime: `τ = N_beam / R_loss`, with `N_beam = I T_rev / e` the number of
  stored electrons.
- Lower bound (every collision lost, `σ = σ_T`): `τ_min = 1/(ρ σ_T c)`, i.e.
  25.4 h at 300 K, independent of the beam.

`reference_formulas.mean_cross_section_ratio(Eb_GeV, T_K)` returns `⟨σ⟩/σ_T`,
so the total collision rate can also be estimated without Monte Carlo as
`(I/e) ρ σ_T L × mean_cross_section_ratio(...)`.

## Integration with a tracking code

The generator is applied independently in each ring section. A typical loop
looks like this, where `sections` (start, length and beam macro-electrons of
each section), `track_to_loss` (your tracking and aperture model, returning a
boolean array) and `T_rev` (revolution period [s]) come from your own code:

```python
import numpy as np
from main import ThermalCompton, e_SI

I_A = 0.227
tc = ThermalCompton(Eb_GeV=18.0, I_A=I_A, T_K=300.0, n_trials_per_macro=100,
                    rng=np.random.default_rng(1))

R_loss_Hz = 0.0
for s_start, length, macros_6d in sections:
    events = tc.generate_section_events_6d(macros_6d, L_m=length,
                                           section_start_s_m=s_start)
    w, (x, xp, y, yp, z, dp) = tc.events_to_macros_cols(events, particle="e")
    lost = track_to_loss(x, xp, y, yp, z, dp, s_start)
    R_loss_Hz += w[lost].sum()

N_beam = I_A * T_rev / e_SI            # number of stored electrons
tau_hours = N_beam / R_loss_Hz / 3600.0
```

A conservative estimate of the statistical error of `R_loss_Hz` is
`sqrt(Σ w_i²)` over the lost events.

## Validation and tests

Run the validation scripts from the repository root. Each prints its results, a
PASS/FAIL summary against exact references (`reference_formulas.py`) with
5σ tolerances, and exits with status 1 if a check fails.

| Script | What is checked | Runtime |
|---|---|---|
| `python validate_photon_sampler.py` | (A) mean `⟨k⟩ = 2.701 kT`, `⟨k²⟩ = 10.35 (kT)²`, χ² of the spectrum; plot | 1 s |
| `python validate_thomson_angles.py` | (B) `⟨cos θ⟩ = 0`, `⟨cos² θ⟩ = 2/5`, uniform `φ`, χ² of both; plot | 3 s |
| `python validate_kn_acceptance.py --kstar-keV 10 -N 2000000` | (C) acceptance = `σ_KN/σ_T`, χ² of `cos θ*` and `x = k*'/k*`; plot | 6 s |
| `python validate_kinematics.py --Eb 18 --T 300 --N 200000` | (D) photon massless, electron on shell, four-momentum conservation, Compton edge, photon direction distribution, acceptance, mean photon energy | 7 s |
| `python validate_absolute_rate.py --Eb 18 --T0 300 --I0 2.5 --L0 1.0 --plot` | (E) absolute rate vs exact expectation, scaling with I, L and T, `τ_min` | 2 min |

Useful options: `--seed`, `--out` (plot file), `-N`; scripts D and E accept
`--isotropic` (`flux_weighting=False`), and D accepts an off-axis or
off-momentum electron, e.g. `--xp 2e-3 --yp=-1e-3 --dp=-5e-3` (write negative
values in scientific notation with `=`).

The automated tests cover the samplers, cross sections, transformations,
kinematics, flux factor, absolute normalization, event bookkeeping and input
validation (about 10 s):

```bash
python -m pytest -q
```

## Performance

One trial collision takes about 26 µs (Apple M2 Max, pure Python + NumPy). The
CPU time per section call scales as `N_macro × K`, e.g. 2000 macros × 200 trials
in about 10 s. `K` sets the statistical precision of the weights; `N_macro`
sets how well the beam distribution and the scattered-particle phase space are
sampled.

## References

* H. Burkhardt, "Monte Carlo Simulation of Scattering of Beam Particles and
  Thermal Photons", CERN SL/Note 93-73 (OP), 1993.
  [https://cds.cern.ch/record/703373](https://cds.cern.ch/record/703373)

* A. Di Domenico, "Inverse Compton Scattering of Thermal Radiation at LEP and
  LEP-200", Particle Accelerators 39 (1992) 137–146.
  [https://s3.cern.ch/inspire-prod-files-1/112d07f6d60840037a97affa44c6718e](https://s3.cern.ch/inspire-prod-files-1/112d07f6d60840037a97affa44c6718e)

* V. Telnov, Nucl. Instrum. Meth. A 260 (1987) 304–307.
  [https://doi.org/10.1016/0168-9002(87)90093-3](https://doi.org/10.1016/0168-9002(87)90093-3)

* G. R. Blumenthal and R. J. Gould, "Bremsstrahlung, Synchrotron Radiation, and
  Compton Scattering of High-Energy Electrons Traversing Dilute Gases",
  Rev. Mod. Phys. 42 (1970) 237.
  [https://doi.org/10.1103/RevModPhys.42.237](https://doi.org/10.1103/RevModPhys.42.237)

* G. B. Rybicki and A. P. Lightman, *Radiative Processes in Astrophysics*
  (Wiley, 1979), Chapter 7 (Compton scattering).

## License

Copyright © 2025, Brookhaven Science Associates, LLC, Brookhaven National
Laboratory. Distributed under the BSA open source license; see
[`LICENSE`](LICENSE).

## Citation

If you use this package in scientific work, please cite the associated
publication or preprint and the repository release URL.
