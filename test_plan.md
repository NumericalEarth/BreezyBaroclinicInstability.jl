# Test Plan: BreezyBaroclinicInstability.jl

## Overview

This plan validates the package from basic smoke tests through the full
resolution cascade (1° → 1/4° → 1/8°) on a single H200 GPU.

The primary validation script is `experiments/single_gpu_cascade.jl`, which
runs all three phases sequentially on one GPU with checkpoint-based handoff
and `Oceananigans.Fields.interpolate!` for resolution transitions.

---

## T1: Smoke Tests

Run interactively in a Julia REPL with `--project`.

### T1.1 Package loads

```julia
using CUDA, Oceananigans, BreezyBaroclinicInstability
Oceananigans.defaults.FloatType = Float32
@assert CUDA.functional()
```

### T1.2 Build model at 1-degree

```julia
model, snapshots = build_model(GPU(); Nλ=360, Nφ=160, Nz=64, Δt=4.0)
@assert snapshots === nothing
@assert size(model.grid) == (360, 160, 64)
```

### T1.3 Build model with relaxation + cloud damping

```julia
model, snapshots = build_model(GPU(); Nλ=360, Nφ=160, Nz=64, Δt=4.0,
    relaxation=(0.1, 1800), cloud_damping=(0.1, 1800))
@assert snapshots !== nothing
@assert haskey(snapshots, :ρu)
@assert haskey(snapshots, :ρθ)
```

### T1.4 Analytic IC

```julia
model, _ = build_model(GPU(); Nλ=360, Nφ=160, Nz=64, Δt=4.0)
set_analytic_ic!(model)
@assert !any_nan(model)
check_density_positivity(model)
report_state(model; label="analytic IC")
```

Expected ranges:
- ρ: [0.01, 1.3] kg/m³
- ρu: jet peak implies |ρu| up to ~40 kg/(m²s)
- ρqᵛ: [0, ~0.02] kg/m³

### T1.5 Checkpoint roundtrip

```julia
save_checkpoint(model, "/tmp/test_ckpt.jld2"; Δt=4.0)
using JLD2
JLD2.jldopen("/tmp/test_ckpt.jld2", "r") do f
    @assert f["Nλ"] == 360
    @assert f["Nφ"] == 160
    @assert f["Nz"] == 64
    @assert haskey(f, "ρ") && haskey(f, "ρθ") && haskey(f, "ρqᵛ")
end
```

### T1.6 Single time step

```julia
Oceananigans.TimeSteppers.update_state!(model)
Oceananigans.TimeSteppers.time_step!(model, 4.0)
@assert !any_nan(model)
```

---

## T2: Single-Phase Validation

### T2.1 Run 1-degree for 6 hours simulated (~5400 iters)

```julia
model, _ = build_model(GPU(); Nλ=360, Nφ=160, Nz=64, Δt=4.0)
set_analytic_ic!(model)
Oceananigans.TimeSteppers.update_state!(model)
Oceananigans.TimeSteppers.time_step!(model, 4.0)

sim = Simulation(model; Δt=4.0, stop_time=6*3600.0)
run!(sim)
```

Check after run:
- No NaN
- Density stays positive
- Zonal wind: westerly jets at midlatitudes, peak ~35-40 m/s
- ρw: small, O(10⁻³) kg/(m²s) — instability is still developing at 6h
- ρqᵛ: non-negative everywhere

### T2.2 Baroclinic wave onset (3 simulated days)

By day 3 at 1-degree:
- Meridional wind (ρv) grows from zero to O(1-5) m/s
- Vertical velocity (ρw) grows to O(0.01-0.1) m/s
- Perturbation visibly breaks zonal symmetry in surface θ

### T2.3 CFL violation check

Run 100 steps at Δt=5.0s (above acoustic CFL limit ~4.3s for 1-degree).
Should blow up with NaN, confirming the CFL constraint is real and the
NaN checker catches it.

---

## T3: Interpolation and Cascade Handoff

### T3.1 Same-resolution roundtrip

Save 1-degree checkpoint, load into a fresh 1-degree model via
`load_ic_interpolated!`. Fields should match to machine precision
(identity interpolation).

### T3.2 Upscale 1° → 1/4°

```julia
model1, _ = build_model(GPU(); Nλ=360, Nφ=160, Nz=64, Δt=4.0)
set_analytic_ic!(model1)
save_checkpoint(model1, "/tmp/ckpt_1deg.jld2"; Δt=4.0)

model2, snaps = build_model(GPU(); Nλ=1440, Nφ=640, Nz=64, Δt=1.0,
    relaxation=(0.1, 1800), cloud_damping=(0.1, 1800))
load_ic_interpolated!(model2, "/tmp/ckpt_1deg.jld2"; clamp_moisture=true)
```

Check:
- No NaN after load
- ρ positive everywhere
- Field extrema in same ballpark as source (no wild values)
- ρqᵛ non-negative (clamp_moisture=true)
- `copy_ic_snapshots!(snaps, model2)` succeeds
- Single time step: no NaN

### T3.3 GPU memory cleanup

After releasing a model, verify GPU memory is reclaimed:

```julia
model = nothing
GC.gc(true); GC.gc(false); GC.gc(true)
CUDA.reclaim()
```

Available memory should return to within ~500 MB of baseline.

---

## T4: Full Cascade Validation

### T4.1 Mini cascade (reduced duration)

Run `experiments/single_gpu_cascade.jl` with environment variable
to select short durations:

| Phase | Full duration | Mini duration | Iterations |
|-------|-------------|---------------|------------|
| 1°    | 14 days     | 6 hours       | 5,400      |
| 1/4°  | 2 days      | 1 hour        | 3,600      |
| 1/8°  | 1 day       | 30 min        | 3,000      |

This validates the full machinery (build → IC → interpolate → relax →
checkpoint → memory cleanup → next phase) in ~30 minutes of wall time.

Checks at each phase transition:
- Checkpoint file exists with correct keys and sizes
- Post-interpolation: no NaN, ρ positive, extrema consistent
- Snapshots match model fields after `copy_ic_snapshots!`
- Relaxation active: ρw extrema decay during first 30 min

### T4.2 Production cascade (full duration)

Run with production durations (14d + 2d + 1d). Checks:

**Phase 1 (1°, 14 days):**
- Days 4-5: wave-4 to wave-6 pattern in surface θ
- Days 10-14: mature cyclones with warm/cold sectors
- Midlatitude jet maintained throughout
- Boundary layer moistening from surface fluxes

**Phase 2 (1/4°, 2 days):**
- First 30 min: relaxation dissipates interpolation staircase
- Sharper frontal structures than 1°
- ρw intensifies (resolved ageostrophic motions)
- Cloud fields develop self-consistently after damping window

**Phase 3 (1/8°, 1 day):**
- Continued frontal sharpening
- Convective-scale features in ρw
- Cloud bands along fronts
- Total energy should not drift wildly

---

## T5: Robustness Checks

### T5.1 NaN detection

`any_nan(model)` is called:
- After every IC load (analytic or interpolated)
- After every first time step
- At every diagnostic callback during `run!`

Any NaN halts immediately with phase label, iteration, and simulated time.

### T5.2 Density positivity

`check_density_positivity(model)` after IC load and periodically during
simulation. Negative density signals CFL violation or severe imbalance.

### T5.3 ρw magnitude monitoring

Vertical momentum density (ρw) is a sensitive indicator of interpolation
artifacts. Expected envelope after cascade interpolation:
- Post-interpolation: |ρw| < 0.1 (inherited from coarser grid)
- After relaxation (t=30min): |ρw| < 0.01 (artifacts damped)
- Mature dynamics (hours): |ρw| grows organically, O(1-10) in convective regions

### T5.4 Checkpoint file sizes

Verify after each `save_checkpoint`:
- 1° checkpoint: ~120 MB
- 1/4° checkpoint: ~1.9 GB
- 1/8° checkpoint: ~7.5 GB

Significantly smaller → truncated write. Significantly larger → corruption.

---

## Memory Budget (single H200, 80 GB HBM3e)

| Phase | Grid | Model memory | Peak (with interp source) |
|-------|------|-------------|--------------------------|
| 1°    | 360×160×64    | ~1 GB  | ~1 GB  |
| 1/4°  | 1440×640×64   | ~6 GB  | ~7 GB  (+ 1° source) |
| 1/8°  | 2880×1280×64  | ~20 GB | ~25 GB (+ 1/4° source) |

All phases fit comfortably within 80 GB.
