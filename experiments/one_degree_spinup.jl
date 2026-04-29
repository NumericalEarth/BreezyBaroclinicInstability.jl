# 1-degree moist baroclinic wave spinup — single GPU.
#
# Generates the first checkpoint in the resolution cascade. Runs with the
# DCMIP analytic IC (set by the model builder) for a configurable duration,
# saving periodic field snapshots.
#
# Launch: julia --project experiments/one_degree_spinup.jl

using CUDA
using Dates
using Printf
using Oceananigans
using Oceananigans.Units
using BreezyBaroclinicInstability

Oceananigans.defaults.FloatType = Float32

output_dir = joinpath(@__DIR__, "output", "one_degree_spinup")
mkpath(output_dir)

@info "Building 1° model" now(UTC)
model = moist_baroclinic_instability_model(GPU(); Nλ = 360, Nφ = 160, Nz = 64)

@info "Built model" model

simulation = Simulation(model; Δt = 30, stop_time = 6hours)
conjure_time_step_wizard!(simulation; cfl = 0.7)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

simulation.output_writers[:fields] = JLD2Writer(model, Oceananigans.fields(model);
    filename = joinpath(output_dir, "fields"),
    schedule = TimeInterval(1hour),
    overwrite_existing = true)

wall_start = Ref(time_ns())

function progress(sim)
    m = sim.model
    wall = (time_ns() - wall_start[]) / 1e9
    @info @sprintf("iter %6d  t=%.0fs (%.2fh)  Δt=%.1fs  wall=%.0fs",
                   m.clock.iteration, m.clock.time, m.clock.time/3600, sim.Δt, wall)
    flush(stderr); flush(stdout)
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(500))

@info "Starting simulation" now(UTC)
wall_start[] = time_ns()
run!(simulation)

# Save final checkpoint in the format expected by cascade scripts
final_path = joinpath(output_dir, "one_degree_final.jld2")
save_checkpoint(model, final_path; Δt = simulation.Δt)

@info "Done!" now(UTC) model.clock
