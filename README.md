# Fractional-Order SEIR/SIR Endemic Model — Figure Code

This repository contains the Python and MATLAB implementations used to
generate every numerical figure in:

> A. Aldawoud, "A Unified Fractional-Order SEIR Framework for Endemic
> Disease Dynamics: Vertical Transmission, Disease-Induced Mortality, and
> Stability" (in preparation / submitted).

It reproduces the classical (integer-order) SIR epidemic and endemic models
of Section 2, and the general Caputo fractional-order SEIR model (and its
SIR reduction) of Section 4, simulated with an explicit Grünwald–Letnikov
fractional-difference scheme as described in Section 5 of the article.

## What it produces

Running the script writes five PNG files (300 dpi) to the current directory,
and prints Table 3 (R0 and the endemic infective level I* for all eight
model variants) to stdout:

| File | Figure | Model |
|---|---|---|
| `fig1_epidemic_timeseries.png` | Fig. 1 | Classical SIR epidemic model |
| `fig2_bifurcation.png` | Fig. 2 | Transcritical bifurcation, endemic model |
| `fig3_endemic_timeseries.png` | Fig. 3 | Classical SIR endemic model |
| `fig4_seir_alpha_comparison.png` | Fig. 4 | General fractional-order SEIR model, $\alpha=1,0.95,0.90$ |
| `fig5_sir_alpha_comparison.png` | Fig. 5 | General fractional-order SIR model, $\alpha=1,0.95,0.90$ |

Table 3's R0 and I* values (all eight model variants) are computed
directly from the closed-form formulas of Section 4 — `seir_R0_Istar`,
`sir_R0_Istar`, and `compute_table3` — rather than simulated, so the whole
numerical comparison in the article can be checked from this one script.

## Requirements

Python 3.9+ with:

```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
```

Install with:

```bash
pip install -r requirements.txt
```

Or MATLAB R2018b+ (or GNU Octave 6.1+) with no additional toolboxes.

## Usage

```bash
python3 generate_figures.py
```

or, in MATLAB/Octave, from the `matlab/` directory:

```matlab
generate_figures
```

Both produce the same five PNG files and the same Table 3 console output;
see `matlab/generate_figures.m` for the MATLAB-specific version and
compatibility notes. Total run time is on the order of ten seconds
(Python) to one or two minutes (MATLAB/Octave) on an ordinary laptop.

## Method

The fractional-order simulations (Figures 4–5) use the explicit
Grünwald–Letnikov (GL) fractional-difference scheme for a Caputo initial
value problem $D^\alpha x = f(t,x)$:

$$
w_0 = 1,\qquad w_m = \left(1 - \frac{\alpha+1}{m}\right) w_{m-1},\qquad
x_{k+1} = h^\alpha f(t_k, x_k) - \sum_{j=1}^{k+1} w_j\, x_{k+1-j}.
$$

The memory sum is $O(n^2)$ in the number of time steps $n = t_{\text{sim}}/h$
and is vectorized with NumPy. The classical (integer-order, $\alpha=1$)
models of Section 2 instead use `scipy.integrate.solve_ivp` (RK45).

Full derivation, parameter values, and the closed-form expressions this
code is checked against are given in the article itself (Sections 2, 4,
and 5); the docstring at the top of `generate_figures.py` gives further
implementation notes, including why the fractional-order simulations run
to $t=1000$ rather than a shorter window.

## Provenance

Both `generate_figures.py` and `matlab/generate_figures.m` consolidate and
generalize an earlier prototype of 11 separate MATLAB files, which
implemented the classical SIR model, an age-structured SIR variant (not
used in the article, see its Discussion), and a single fractional-order
SEIR case (Table 1, case (a), i.e. no disease-induced mortality and no
vertical transmission). Both consolidated versions are strict
generalizations in the sense that setting the disease-induced mortality
rate and vertical-transmission probabilities to zero reproduces the
original prototype's equations exactly; the Python script's module
docstring and the MATLAB script's header comment give the full
correspondence. The MATLAB port's Grünwald–Letnikov update rule was
cross-checked against the Python version to floating-point precision
before being added to this repository.

## Citation

If you use this code, please cite the article above. This repository is
archived on Zenodo with a citable DOI:
[10.5281/zenodo.22313974](https://doi.org/10.5281/zenodo.22313974).

## License

MIT — see [LICENSE](LICENSE).
