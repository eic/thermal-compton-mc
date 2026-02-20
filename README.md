# ThermalCompton - thermal (blackbody) Compton event generator for tracking

This repository provides a compact Monte Carlo event generator for **thermal Compton scattering** of ultra-relativistic electrons on **blackbody photons**, intended for integration into multi-turn tracking workflows (e.g., ring-section-by-section event generation). The implementation follows the algorithmic approach described by H. Burkhardt (Thomson-angle proposal sampling with Kleinâ€“Nishina acceptance) and uses an explicit boost/rotation kinematics convention consistent with standard inverse-Compton treatments (see references below).

The public release includes:
- the `ThermalCompton` class (Python) implementing the event generator (`main.py`),
- minimal driver/validation scripts that generate events without a lattice and reproduce benchmark plots (`validate_*.py`),
- an example integration pattern for generic tracking codes,
- documentation describing required inputs (beam energy, current, temperature, number of trials, Î´-threshold) and output conventions.


## Requirements

- Python 3.10+ (type hints use `X | None`)
- `numpy`
- `matplotlib` (for validation plots)

## Repository contents

- `main.py`  
  Implementation of `ThermalCompton` and helper routines (sampling, boosts/rotations, weighting, 6D-like output).

Validation scripts (no lattice required):
- `validate_photon_sampler.py` — (A) photon sampler benchmark at T = 300 K (spectrum + mean energy)
- `validate_thomson_angles.py` — (B) Thomson angular sampler benchmark (`cosθ` shape and `φ` uniformity)
- `validate_kn_acceptance.py` — (C) Klein–Nishina acceptance benchmark at representative fixed ERF photon energy `k*`
- `validate_kinematics.py` — (D) kinematics consistency checks (masslessness, mass shell, conservation)
- `validate_absolute_rate.py` — (E) absolute-rate scaling checks (I, L, T³) + τ_min estimate

## Quick start: generate events in a section (no lattice)

The generator is designed to be called per ring section. Example for a simple macro-beam (on-axis, on-momentum):

```python
import numpy as np
from main import ThermalCompton

tc = ThermalCompton(Eb_GeV=18.0, I_A=0.227, T_K=300.0, n_trials_per_macro=200)

# Macro-electron ensemble: (x, y, z, x', y', dp)
macros_6d = [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(2000)]

events = tc.generate_section_events_6d(
    macros_6d=macros_6d,
    L_m=1.0,
    section_start_s_m=0.0,
    dp_threshold=1e-3,   # thinning threshold in δ = Δp/p (not the ring acceptance)
)

R_section_Hz = sum(ev["w_Hz"] for ev in events)
print("Section rate [Hz]:", R_section_Hz)
````

### Notes on `dp_threshold`

`dp_threshold` is a **numerical thinning threshold** controlling which scattered electrons are retained for tracking/propagation. It is typically set well below the ring momentum acceptance and should not be interpreted as the machine acceptance itself.

## Inputs, outputs, and conventions

### Initialization

```python
ThermalCompton(
    Eb_GeV,              # reference beam energy [GeV] (defines p0)
    I_A,                 # beam current [A] (sets absolute rate normalization)
    T_K,                 # temperature [K] (sets photon density and energy sampler)
    n_trials_per_macro=100,
    nmax_sampler=2000,
    rng=None,
    photon_energy_sampler_eV=None,
)
```

Key parameters:

* `Eb_GeV`: defines reference momentum `p0` used in `dp = p/p0 - 1`.
* `I_A`: converts to electron flow rate `I/e` for absolute weighting.
* `T_K`: sets blackbody photon density ρ(T) and the photon energy sampler.
* `n_trials_per_macro`: number of Thomson proposals per macro-electron per section call.
* `rng`: optional `numpy.random.Generator` for reproducibility (seeded RNG recommended).

### Section call

```python
generate_section_events_6d(macros_6d, L_m, section_start_s_m=0.0, dp_threshold=...)
```

* `macros_6d`: iterable of macro-electrons `(x, y, z, x', y', dp)`
* `L_m`: section length [m]
* `section_start_s_m`: section start coordinate [m] used only as an event tag
* `dp_threshold`: keep only events with `|dp_e_out| >= dp_threshold` (optional thinning)

### Output event record

Each returned event is a dictionary with:

* `s_m`: interaction position [m] (uniformly distributed along the section)
* `w_Hz`: **absolute event weight** [Hz]
* `e6d_out`: scattered electron `(x, y, z, x', y', dp)` (None if `pz<=0`)
* `g6d_out`: scattered photon “6D-like” `(x, y, z, x', y', dp)`
* `photon_E_GeV`: scattered photon energy [GeV] (diagnostic)

6D-like convention:

* `x' = p_x / p_z`, `y' = p_y / p_z`
* `dp = (p/p0) - 1`, with `p0` set by `Eb_GeV`

## Absolute weighting (Hz)

Beam current corresponds to an electron flow rate:

* `Ndot = I / e`  [1/s]

In a photon gas of number density `rho(T)` [1/m³], the interaction rate per unit length for one electron is:

* `dP/ds = rho(T) * sigma`  [1/m]

Thus, the physical collision rate in a section of length `L` is:

* `R = (I/e) * rho(T) * sigma * L`  [1/s]

Following Burkhardt’s proposal method, trial angles are sampled from the Thomson distribution
`dσ_T/dΩ ∝ 1 + cos²θ` and accepted with probability `(dσ_KN/dΩ)/(dσ_T/dΩ)` so that accepted events follow the Klein–Nishina distribution.

With `Nmacro` macro-electrons and `K = n_trials_per_macro` trials per macro per section call, each accepted trial carries:

* `w_Hz = (I/(e*Nmacro)) * rho(T) * σ_T * (L/K)`

Summing `w_Hz` over accepted events yields the Compton collision rate [Hz] in that section; summing `w_Hz` only over accepted events whose scattered electron is subsequently lost yields the loss rate [Hz].

## Validation scripts (A–E)

Run each script from the repository root.

(A) Photon sampler validation (T = 300 K)

```bash
python validate_photon_sampler.py
```

(B) Thomson angular sampler

```bash
python validate_thomson_angles.py
```

(C) Klein–Nishina acceptance at representative `k*`

```bash
python validate_kn_acceptance.py --kstar-keV 10 -N 2000000
```

(D) Kinematics consistency checks

```bash
python validate_kinematics.py --Eb 18 --T 300 --N 200000
```

(E) Absolute rate scaling checks and τ_min

```bash
python validate_absolute_rate.py --Eb 18 --T0 300 --I0 2.5 --L0 1.0
python validate_absolute_rate.py --plot
```

## Integration pattern for tracking codes (generic)

A typical integration applies the generator independently in each ring section:

1. represent the stored beam by macro-electrons `(x,y,z,x',y',dp)`,
2. call `generate_section_events_6d(...)` per section,
3. propagate scattered particles through the lattice/aperture model,
4. accumulate absolute rates:

   * total scatter rate: sum of `w_Hz`,
   * loss rate: sum of `w_Hz` for particles that are lost,
   * lifetime estimate: `τ ≈ N_beam / loss_rate` (with the chosen normalization and macro-representation).

## References

* H. Burkhardt, “Monte Carlo Simulation of Scattering of Beam Particles and Thermal Photons”, SL/Note 93-73 (OP), 1993.
  [https://cds.cern.ch/record/703373/files/thermal.pdf](https://cds.cern.ch/record/703373/files/thermal.pdf)

* A. Di Domenico, “Inverse Compton Scattering of Thermal Radiation at LEP and LEP-200”, Particle Accelerators 39 (1992) 137–146.
  [https://s3.cern.ch/inspire-prod-files-1/112d07f6d60840037a97affa44c6718e](https://s3.cern.ch/inspire-prod-files-1/112d07f6d60840037a97affa44c6718e)

* V. Telnov, Nucl. Instrum. Meth. A 260 (1987) 304–307. DOI: 10.1016/0168-9002(87)90093-3 
  [https://doi.org/10.1016/0168-9002(87)90093-3](https://doi.org/10.1016/0168-9002(87)90093-3)

## License

MIT License (see `LICENSE`).

## Citation

If you use this package in scientific work, please cite the associated publication/preprint and the repository release URL.