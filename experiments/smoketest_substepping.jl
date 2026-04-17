# Smoke tests for the acoustic-substepping build of BreezyBaroclinicInstability.
#
# Exercises T1.1 – T1.6 from test_plan.md plus a 10-minute 1° run with the
# adaptive TimeStepWizard active. Any failure aborts with a descriptive error.
#
# Launch: julia --project experiments/smoketest_substepping.jl

using CUDA
using Dates
using Printf
using Oceananigans
using Breeze
using BreezyBaroclinicInstability

Oceananigans.defaults.FloatType = Float32

@info "T1.1 — Package loads"
@assert CUDA.functional() "CUDA.functional() == false"
@info "CUDA functional" device=CUDA.device() free_GB=round(CUDA.available_memory()/1e9, digits=2)

@info "T1.2 — Build model at 1° (substepping)"
model, snapshots = build_model(GPU(); Nλ=360, Nφ=160, Nz=64, Δt=30.0)
@assert snapshots === nothing
@assert size(model.grid) == (360, 160, 64)
@info "build_model OK" grid=size(model.grid) td=model.dynamics.time_discretization

@info "T1.3 — Build model with relaxation + cloud damping"
model2, snaps2 = build_model(GPU(); Nλ=360, Nφ=160, Nz=64, Δt=30.0,
    relaxation=(0.1, 1800), cloud_damping=(0.1, 1800))
@assert snaps2 !== nothing
@assert haskey(snaps2, :ρu)
@assert haskey(snaps2, :ρθ)
@info "relaxation path OK" keys=keys(snaps2)

model2 = nothing; snaps2 = nothing
GC.gc(true); GC.gc(false); GC.gc(true); CUDA.reclaim()

@info "T1.4 — Analytic IC"
set_analytic_ic!(model)
@assert !any_nan(model)
check_density_positivity(model)
report_state(model; label="analytic IC")

@info "T1.5 — Checkpoint roundtrip"
tmp_ckpt = "/tmp/bbi_smoke_ckpt.jld2"
save_checkpoint(model, tmp_ckpt; Δt=30.0)
using JLD2
JLD2.jldopen(tmp_ckpt, "r") do f
    @assert f["Nλ"] == 360
    @assert f["Nφ"] == 160
    @assert f["Nz"] == 64
    @assert haskey(f, "ρ") && haskey(f, "ρθ") && haskey(f, "ρqᵛ")
end
@info "checkpoint roundtrip OK"

@info "T1.6 — Single time step"
Oceananigans.TimeSteppers.update_state!(model)
Oceananigans.TimeSteppers.time_step!(model, 30.0)
@assert !any_nan(model)
report_state(model; label="after first step")
@info "first step OK" clock=model.clock

@info "T2.1 — Short adaptive 1° run (10 minutes sim, wizard active)"
simulation = Simulation(model; Δt=30.0, stop_time=600.0)
conjure_time_step_wizard!(simulation; cfl=0.7, max_Δt=300.0, max_change=1.1)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

wall_start = Ref(time_ns())
function diag(sim)
    m = sim.model
    wall = (time_ns() - wall_start[]) / 1e9
    ρ_min, ρ_max = field_extrema(BreezyBaroclinicInstability.dynamics_density(m.dynamics))
    ρw_min, ρw_max = field_extrema(m.momentum.ρw)
    @info @sprintf("iter=%6d  t=%8.1fs  Δt=%5.1fs  wall=%6.1fs  ρ=[%.3e,%.3e]  ρw=[%.2e,%.2e]",
                   m.clock.iteration, m.clock.time, sim.Δt, wall, ρ_min, ρ_max, ρw_min, ρw_max)
    flush(stderr); flush(stdout)
end
simulation.callbacks[:diag] = Callback(diag, IterationInterval(5))

wall_start[] = time_ns()
Oceananigans.run!(simulation)

report_state(model; label="after 10-min run")
@info "SMOKE TESTS PASSED" now=now(UTC) final=model.clock final_Δt=simulation.Δt
