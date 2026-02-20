# ThermalCompton — thermal (blackbody) Compton event generator for tracking

This repository provides a compact Monte Carlo event generator for **thermal Compton scattering** of ultra-relativistic electrons on **blackbody photons**, intended for integration into multi-turn tracking workflows (e.g., ring-section-by-section event generation). The implementation follows the algorithmic approach described by H. Burkhardt (Thomson-angle proposal sampling with Klein–Nishina acceptance) and uses an explicit boost/rotation kinematics convention consistent with standard inverse-Compton treatments (see references below).

The public release includes:
- the `ThermalCompton` class (Python) implementing the event generator (`main.py`),
- minimal driver/validation scripts that generate events without a lattice and reproduce benchmark plots (`validate_*.py`),
- an example integration pattern for generic tracking codes,
- documentation describing required inputs (beam energy, current, temperature, number of trials, δ-threshold) and output conventions.

