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

| config                                                      | lat range    | `cfl` | `max_Δt` (s) | `cloud_formation_τ` (s) | blowup time |
|-------------------------------------------------------------|--------------|-------|--------------|-------------------------|-------------|
| flat 30 km, uniform z                                       | (-80, 80)    | 0.7   | 150          | 120                     | 16 h        |
| flat 30 km, uniform z                                       | (-80, 80)    | 0.7   | 100          | 120                     | 17.6 h      |
| flat 30 km, uniform z                                       | (-80, 80)    | 0.7   | 60           | 120                     | 18.3 h      |
| flat 30 km, uniform z                                       | (-80, 80)    | 0.7   | 60           | 600                     | 26.4 h      |
| 45 km, stretched z, top sponge (1/600s, 7km)                | (-80, 80)    | 0.5   | 120          | 600                     | 38.9 h      |
| + latitude → (-75, 75)                                      | (-75, 75)    | 0.7   | 120 (spinup) | 600                     | 39 h        |
| + stronger sponge (1/180s, 12km)                            | (-75, 75)    | 0.7   | 120 (spinup) | 600                     | 26 h (worse)|
| + smaller spinup Δt (30s)                                   | (-75, 75)    | 0.7   | 30 (spinup)  | 600                     | 46 h        |
| **+ polar filter (threshold_lat=60)**                       | (-75, 75)    | 0.7   | 30 (spinup)  | 600                     | **46 h**    |

Growth rate slowed from ~25× per 500 iters (original) to ~2× per 500 iters
(latest), but ρw eventually doubles every ~4 hours after day 1 regardless of
knob. The blowup happens in top cells where ρ≈5×10⁻⁴ kg/m³; there, |ρw|≈0.05
translates to |w|≈100 m/s, a clear upper-atmosphere amplification. Polar
filter did **not** change the trajectory — rules out polar convergence as the
driver. Stronger sponge made it **worse** (target-zero forcing creates
near-top velocity gradients).

### Verified *not* causing the blowup

- **Acoustic substepping** — `acoustic_substepping.jl:1319-1320` enforces
  `N = max(6, 6·cld(N_raw, 6))`. At Δt=60s on the 1° grid (Δx_min≈9.7 km) the
  minimum is 6 substeps; Δτ=10 s gives acoustic CFL 0.36. Safe.
- **Default NaN-checker cadence** — once pinned to `IterationInterval(1)` it
  catches the ρ=NaN at its first appearance.

### Likely remaining causes (after the 2026-04-17 evening sweep)

1. **Float32 round-off at the top cell** — the most likely remaining suspect.
   With H=45km, top-cell ρ≈5×10⁻⁴ kg/m³; Float32 has ≈7 digits of precision so
   relative errors in ρu/ρ, ρw/ρ, and thermodynamic inversions are order 1e-7.
   After thousands of iterations those drift. Try `Oceananigans.defaults.FloatType = Float64`
   in the experiment scripts; the cost is ~2× memory and ~1.5× compute, but
   it directly tests this hypothesis.
2. **Sponge design, not strength** — relaxing ρu, ρv, ρw toward **zero** at
   the top is Rayleigh friction, which creates a velocity gradient between
   the jet and the sponged top. Targeting the *balanced zonal wind* (i.e.
   `balanced_zonal_wind(φ, z)` from `src/balanced_state.jl`) rather than 0
   would keep the sponge consistent with the base state.
3. **Bulk-flux stiffness** — `Cᴰ = 1e-3`, `Uᵍ = 1e-2`. Not yet tested. Halve
   `Cᴰ` or gate fluxes off the first day to see whether surface forcing is
   driving the initial perturbation that then grows.
4. **Add a small horizontal viscosity** — a `HorizontalScalarDiffusivity(ν = 1e4 m²/s)`
   on momentum tendencies would damp small-scale features without hurting the
   BCI signal. Cheap to try.

### Ruled out (from the 2026-04-17 sweep)

- Acoustic substepping (forced min 6 substeps, CFL 0.36 at Δt=60s).
- Polar convergence (Breeze `add_polar_filter!(threshold_latitude=60)` enabled
  — identical ρw trajectory to runs without it).
- CFL-driven numerical instability (blowup time is the same at Δt=30 and Δt=120
  after accounting for iter count; smaller Δt only delays by ~20%).
- Cloud-formation timescale alone (`τ ∈ {120, 600}` both blow up).
- Polar metric (extending latitude to (-75, 75) gave 3× larger Δx_min with no
  meaningful change in blowup time).

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
