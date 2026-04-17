# Known Issues

## Moist-BCI stability beyond t ≈ day 1 (1° Phase 1)

Status: open. Hit on 2026-04-17.

### Symptom

Phase 1 of `experiments/single_gpu_cascade.jl` (1° global grid, moist microphysics,
bulk surface fluxes, acoustic substepping + adaptive `TimeStepWizard`) blows up
with `NaN found in field ρ` at simulated time t ≈ day 1 regardless of the
`max_Δt` or `cloud_formation_τ` tried so far. The blow-up is preceded by
explosive ρw growth:

| iter | sim time | ρw range              |
|------|----------|-----------------------|
| 500  | 0.31 d   | ±5×10⁻³ (healthy)     |
| 1000 | 0.66 d   | ±2×10⁻²               |
| 1500 | 1.01 d   | ±2.2×10⁻¹ (unstable)  |
| 1634 | 1.10 d   | NaN                   |

The mini cascade (stop_time = 6 h on Phase 1) passes because it ends before
the instability develops.

### Sensitivity sweep

| config                                                      | `cfl` | `max_Δt` (s) | `cloud_formation_τ` (s) | blowup time |
|-------------------------------------------------------------|-------|--------------|-------------------------|-------------|
| flat 30 km, uniform z                                       | 0.7   | 150          | 120                     | 16 h        |
| flat 30 km, uniform z                                       | 0.7   | 100          | 120                     | 17.6 h      |
| flat 30 km, uniform z                                       | 0.7   | 60           | 120                     | 18.3 h      |
| flat 30 km, uniform z                                       | 0.7   | 60           | 600                     | 26.4 h      |
| **45 km, stretched z, top sponge (1/600s, 7km)**            | 0.5   | 120          | 600                     | **38.9 h**  |

Each configuration change pushes the blowup later; no single knob stabilises
the full 14-day spinup. Growth rate also slows — from ~25× per 500 iters in
the original uniform-z run to ~5× in the stretched+sponge run — but ρw still
runs away.

### Verified *not* causing the blowup

- **Acoustic substepping** — `acoustic_substepping.jl:1319-1320` enforces
  `N = max(6, 6·cld(N_raw, 6))`. At Δt=60s on the 1° grid (Δx_min≈9.7 km) the
  minimum is 6 substeps; Δτ=10 s gives acoustic CFL 0.36. Safe.
- **Default NaN-checker cadence** — once pinned to `IterationInterval(1)` it
  catches the ρ=NaN at its first appearance.

### Likely remaining causes

1. **Polar Δx_min metric with no filter** — at φ=80° Δx≈9.7 km and the
   advective CFL at the BCI peak (U≈60 m/s) crosses 0.7. WENO(5) alone isn't
   enough at that latitude band. Try wiring in Breeze's `PolarFilter` or a
   small `HorizontalScalarDiffusivity(ν ≈ 1e4 m²/s)`.
2. **Float32 near the top** — with H=45km, top-cell ρ drops to ~5×10⁻⁴ kg/m³;
   any Float32 round-off in ρw/ρ gets amplified. Try `FloatType = Float64` in
   the experiment scripts and re-check stability.
3. **Sponge too weak** — current `rate = 1/600 s`, `width = 7 km`. A gravity-
   wave packet traverses the sponge in ~1500 s so only e-folds once before
   reflecting. Doubling `rate` and widening `width` to 10 km is cheap.
4. **Bulk-flux stiffness** — `Cᴰ = 1e-3`, `Uᵍ = 1e-2`. If an implicit in-Δt
   flux integration isn't available, at least halving `Cᴰ` is a sanity test.

### What works today

- `experiments/smoketest_substepping.jl` — 10-minute adaptive 1° run (pass).
- `CASCADE_MINI=true experiments/single_gpu_cascade.jl` — 6h + 1h + 30min
  end-to-end cascade (pass, 23 min wall on a single H200).
- All three resolution levels interpolate correctly; checkpoint roundtrips
  pass; NaN checker fires at IterationInterval(1); `conjure_time_step_wizard!`
  adapts Δt from advective CFL cleanly.

### Next debug steps (suggested)

- Turn microphysics off entirely (e.g. `cloud_formation_τ = 1e9`) and see
  whether the dry run reaches day 14 stably. That isolates whether the issue
  is fundamental dynamics or moist coupling.
- Add `Oceananigans.HorizontalScalarDiffusivity(ν = 1e4)` to `build_model` and
  see whether it damps the ρw growth without suppressing the BCI signal.
- Reduce SST gradient or `Cᴰ` to soften surface flux forcing at Phase-1
  spin-up.
