# Acoustic-substepper instability — investigation report

Date: 2026-04-27
Branch: `~/Breeze` `glw/hevi-imex-docs` HEAD `de99960` plus the user's
uncommitted rewrite of `src/CompressibleEquations/acoustic_substepping.jl`.

## TL;DR

The split-explicit substepper has an unstable mode that fires for *rest
atmospheres at machine zero* whenever the **per-substep vertical acoustic
CFL** `cs · Δτ / Δz` is large enough — observed unstable at ratio ≈ 2.5 and
above, observed stable at ratio ≈ 0.37. Growth rate is roughly factor ~2 per
outer Δt, so seeds at machine epsilon saturate the field within ~30 outer
steps. The mode is horizontally uniform (purely 1-D vertical), which is the
signature of a bug in either the implicit vertical solve or the
slow-vertical-momentum-tendency assembly. The implicit half of the
off-centered Crank–Nicolson scheme (ω = 0.55) *should* be unconditionally
stable for arbitrary `cs Δτ / Δz`; observation says otherwise. This is the
mechanism behind the moist baroclinic-wave failure.

## Symptom on the moist BW

Verbatim re-run of `Breeze/examples/moist_baroclinic_wave.jl` (1°,
`latitude=(-80,80)`, Lz=30km, isothermal-T₀=250K reference state, Δt=20s):

```
step  1   max|ρw|=0.030
step  3   max|ρw|=0.042
step  5   max|ρw|=0.108
step  9   max|ρw|=1.96
step 12   max|ρw|=19.4
step 14   max|ρw|=91
step 15   NaN
```

Doubling per outer step. e-fold ~30 s. Float32 → Float64 changes nothing.
Same on the dry baroclinic-wave example. Same with
`KlempDivergenceDamping(coefficient=0.5)` enabled.

## What ruled out

1. **Float precision** — Float64 produces identical growth rates.
2. **Microphysics / surface fluxes** — dry baroclinic wave (no moisture, no
   surface fluxes) blows up the same way.
3. **BBI-specific IC code** — verbatim copy of Breeze's example IC into the
   BBI environment reproduces the failure.
4. **`ρe` BC bug we found** — fixing `ρe_bcs` → `ρθ_bcs` doesn't change the
   instability (it does fix the previously-silently-dropped surface heat
   flux, which is a separate real bug).
5. **Damping choice** — `NoDivergenceDamping` (default) and
   `KlempDivergenceDamping(coefficient=0.5)` both fail.
6. **Spherical metric / curvilinear grid** — 3-D Cartesian rest atmosphere at
   Δt=20s blows up identically to the lat-lon test.
7. **Lz** — earlier I thought Lz mattered, but at fixed Δt=20s the bug
   appears at both Lz=10 km and Lz=30 km on Cartesian and lat-lon alike.
   The Lz-dependence I'd seen was a confound with Δt.
8. **Slow tendency** — running the same setup with `ExplicitTimeStepping()`
   (no substepper) keeps a rest atmosphere at machine zero. Confirms the
   bug is inside the substep loop, not in the dynamics kernels.

## Reduction to the simplest reproducer

Rest atmosphere = reference state, no flow, no perturbation. The substepper
should keep `max|w|` at floating-point zero indefinitely. Test matrix:

| grid                                | Lz   | Δt    | Δτ ≈ Δt/6 | `cs·Δτ/Δz_min` | result   |
|-------------------------------------|------|-------|-----------|----------------|----------|
| 2-D Cart `(Periodic,Flat,Bounded)`  | 10 km| **1 s** | 0.17 s   | 0.37           | stable ✓ |
| 3-D Cart `(Periodic,Periodic,Bounded)` | 10 km| 20 s | 3.33 s | 7.4            | grows ✗  |
| 3-D Cart `(Periodic,Bounded,Bounded)` | 10 km| 20 s | 3.33 s | 7.4            | grows ✗  |
| 3-D Cart `(Periodic,Periodic,Bounded)` | 30 km| 20 s | 3.33 s | 2.5            | grows ✗  |
| Lat-lon `(Periodic,Bounded,Bounded)`| 10 km| 20 s  | 3.33 s   | 7.4            | grows ✗  |
| Lat-lon `(Periodic,Bounded,Bounded)`| 30 km| 20 s  | 3.33 s   | 2.5            | grows ✗  |
| 2-D Cart, Lz=30 km                  | 30 km| 20 s  | 3.33 s   | 1.4            | grows ✗  |

All `Δt=20s` cases produce identical growth rates (factor ≈ 2.2 per outer
step). The `Δt=1s` case is stable forever. The discriminator is `Δt`, not
`Lz` or curvilinearity.

## Mode structure

After one outer step on the 3-D Cartesian rest test:
- `max|ρw|` at step 1 lives at level k=4, with **all 64×64 horizontal cells
  at exactly the same value** (4.192e-14, std ≈ 6e-30). Pure 1-D vertical
  mode.
- `max|u|`, `max|v|` stay at exactly 0.0 for every recorded iteration. The
  perturbation never leaks into horizontal momentum.

A horizontally-uniform mode that grows over time, with no horizontal
momentum coupling, is the signature of a bug in the **per-column vertical
solve** (`_build_predictors_and_vertical_rhs!` → tridiag → `_post_solve_recovery!`)
or in the assembly of `Gˢρw` that feeds the column (`assemble_slow_vertical_momentum_tendency!`).

## The seed

After `set!(model; θ=θᵇᵍ, ρ=ref.density)` and `update_state!`:
- `outer_step_density - ref.density` = exactly 0 (we set ρ from ref.density)
- `outer_step_pressure - ref.pressure` = **2.91 × 10⁻¹¹ Pa**
- `pressure_imbalance` = 2.91 × 10⁻¹¹
- `Gˢρw` after `assemble_slow_vertical_momentum_tendency!` = **6.2 × 10⁻¹⁴ N/m³**

The pressure mismatch is because `update_state!` derives p from `(ρ, ρθ)`
via the EoS, while `ref.pressure` was constructed by trapezoidal hydrostatic
integration in `_compute_exner_hydrostatic_reference!`. They agree
analytically but produce slightly different Float64 values. The mismatch
flows into `Gˢρw` via `-∂z(p⁰-p_ref)/Δz_face`. This is not the bug — it's
the *seed* (a few hundred ulp). The bug is that the substepper amplifies
this seed instead of damping it.

## Hypothesis

The implicit Schur tridiag for μw in `acoustic_substepping.jl` lines
509-561 is supposed to make the linearized vertical acoustic system
unconditionally stable for `forward_weight = ω = 0.55` (the canonical
ε = 2ω − 1 = 0.1 minimal off-centering). At rest, the system reduces to a
purely vertical acoustic-buoyancy problem on each (i, j) column.

The observed growth (factor ≈ 2.2 per outer step at Δτ=3.33 s, Δz=156 m,
cs ≈ 348 m/s, so per-substep ≈ 1.14, six substeps → factor ≈ 2.4) suggests
that the eigenvalue of the column matrix is *outside* the unit disk for
these parameters. That contradicts the unconditional-stability claim.

Most likely culprits, in order:

1. **Mismatch between predictor / matrix off-centering weights.** The
   predictor and post-solve recovery use `δτ_new = ω · Δτ` (lines 742-749,
   802-806). The matrix coefficients use `δτ_new²` (lines 519-520, 539-540,
   559-560). The eigenvalue analysis only closes if the predictor's
   σ-to-face averaging exactly matches the matrix's σ-elimination form. A
   factor-of-2 or a sign on a single ω weight would produce an unstable
   mode of exactly the kind observed (purely vertical, growth depending on
   Δτ²).

2. **Buoyancy-PGF cross-coupling in the matrix.** Diagonals (line 539-540)
   add `δτ_new² × γRᵈ Π × θ_face × (rdz_above + rdz_below)/Δz_face` (PGF) to
   `δτ_new² × g × (rdz_above − rdz_below)/2` (buoyancy). The buoyancy term
   has a *signed* Δz-asymmetry (a − b form), and on a non-uniform z-grid the
   sign of `rdz_above − rdz_below` flips when the vertical spacing changes
   monotonicity. Cancellations could leave a tiny matrix asymmetry that's
   benign for small `δτ_new²` but drives an unstable eigenvalue at larger
   `δτ_new²`.

3. **`Gˢρw` assembly using cell-centered ρ° and p° to compute a face-located
   tendency.** Lines 614-625: `gσ_face = (gσ_above + gσ_below)/2` is a
   simple mean from cell centers, while `∂z_p′ = (p′_above − p′_below)/Δz_face`
   is centered difference. For a hydrostatically-balanced column these *do*
   exactly cancel for an isothermal reference state with uniform Δz, but the
   cancellation is delicate (each operator must use bit-identical inputs and
   matched discretization). Any z-stretching breaks the exact cancellation,
   leaving a `Gˢρw` proportional to `O(Δz²)` per face. With our setup having
   uniform Δz this should be exact, but it's worth checking by hand against
   the predictor's mass-flux divergence form.

## What didn't help

Without changing damping or off-centering (per user's direction):
- Switching `WENO(order=5)` advection to no advection (rest state has no
  advection anyway) — irrelevant, growth rate unchanged.
- Cartesian topologies of every kind at Δt=20s — all unstable.

## Reproducers checked in

In `/teamspace/studios/this_studio/BreezyBaroclinicInstability.jl/experiments/`:

- `repro_substepper_moist_bw.jl` — full Breeze moist-BW example inlined. NaN
  step 15 at Δt=20s.
- `repro_substepper_dry_bw.jl` — same in dry mode. NaN step 11 at Δt=225s
  (Breeze's documented "stable" dry value), step 18 at Δt=20s.

In `/tmp/`:

- `rest_3d_cart.jl` — minimal 3-D Cartesian rest test. Edit Δt and Lz to
  scan. Δt=20s reproduces the bug; Δt=1s does not.
- `rest_2d_tall.jl` — 2-D Flat-y at Lz=30 km Δt=20s — reproduces the bug
  even with Flat-y topology, ruling out Bounded-y as the cause.
- `rest_3d_gns.jl` — instruments the seed: prints `outer_step_pressure -
  ref.pressure`, `pressure_imbalance`, and `Gˢρw` after one stage prep.
  Documents the 2.9e-11 Pa seed and 6.2e-14 N/m³ slow tendency.
- `rest_latlon.jl` — lat-lon rest atmosphere; isolates curvilinear vs
  Cartesian (no separate curvilinear bug; lat-lon at Δt=20s grows
  identically to Cartesian).

## Recommended next steps

1. **Eigenvalue scan of the column tridiag.** Build the matrix at a single
   (i, j) for the rest reference state, sweep `δτ_new ∈ [0, 5s]`, dump
   `eigvals` to see exactly where the spectrum leaves the unit disk. The
   off-centered CN proof says it shouldn't; observation says it does.
   Whichever coefficient term (PGF off-diagonal vs buoyancy diagonal) is
   responsible can be located by setting the others to zero and seeing the
   spectrum.
2. **Verify predictor/post-solve consistency** by hand: substitute the
   post-solve `σ_new`, `η_new` formulas into the matrix RHS and confirm the
   resulting system is exactly the centered-CN linearized acoustic system.
   The current code mixes `Δτ` (in predictor advection) and `δτ_new = ω Δτ`
   (in implicit half) — the algebra needs to close.
3. **Make `outer_step_pressure ≡ ref.pressure` at rest.** Even if the
   amplification mechanism is fixed, the 2.9e-11 Pa seed is non-physical.
   Either snapshot from `ref.pressure` directly, or apply the same EoS path
   that `update_state!` uses when constructing the reference state, so both
   produce bitwise-identical fields.
