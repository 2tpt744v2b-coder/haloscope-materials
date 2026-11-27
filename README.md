# Haloscope Materials

Open-source Python tool that ranks cavity wall materials for axion haloscope
experiments, with honest uncertainty.

**Status: pre-alpha. Nothing here is validated yet. Do not use for real design decisions.**

## Goal
Given operating conditions (frequency, temperature, magnetic field, cavity size),
rank candidate materials by their expected effect on detector sensitivity.

## Install (development)
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .

## Run the tests
    pytest

## Principles
- Every material value must have a cited source.
- SI units internally.
- Uncertainty is reported, not hidden.

## License
MIT (see LICENSE).
