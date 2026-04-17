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

| `max_Δt` (s) | `cloud_formation_τ` (s) | blowup time |
|--------------|-------------------------|-------------|
| 150          | 120                     | t ≈ 16 h    |
| 100          | 120                     | t ≈ 17.6 h  |
| 60           | 120                     | t ≈ 18.3 h  |
| 60           | 600                     | t ≈ 26.4 h  |

The ordering suggests both outer-step CFL and condensation-timescale matter,
but neither alone stabilises the run through the full 14-day spinup.

### Likely causes to investigate

1. **Bulk-flux stiffness** — `Cᴰ = 1e-3`, `Uᵍ = 1e-2` in `src/model.jl`. With
   cold mid-latitude jets over a warm SST gradient, the sensible/vapor flux
   update per outer Δt may be too large. Try halving `Cᴰ` or `Uᵍ`, or look at
   implicit-in-Δt flux integration.
2. **Missing horizontal diffusion/filter** — the cascade currently uses WENO(5)
   advection only. Breeze's `PolarFilter` is not wired in. At 1° near the pole
   Δx ≈ 10 km, so the advective CFL at the BCI peak (U ≈ 60 m/s) pushes 1
   easily; dealias or add a small `HorizontalScalarDiffusivity`.
3. **Moisture feedback** — non-equilibrium cloud formation couples to potential
   temperature through latent heat release. At `Δt/τ` ≈ 0.1 (Δt=60, τ=600)
   this is no longer the dominant feedback, but combined with (1) and (2) it
   may be amplifying small perturbations.
4. **Float32 precision** — Oceananigans is compiled here with `FloatType =
   Float32`. Breeze's published Δt sweep used Float32 for a dry run; the moist
   run has tighter thermodynamic tolerances and may benefit from Float64 at
   least for the thermodynamic update.

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
