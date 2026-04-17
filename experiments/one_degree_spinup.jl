# 1-degree moist baroclinic wave spinup — single GPU.
#
# Generates the first checkpoint in the resolution cascade.
# Runs with analytic IC for a configurable duration, saving periodic
# checkpoints via Oceananigans.JLD2OutputWriter.
#
# Launch: julia --project experiments/one_degree_spinup.jl

using CUDA
using Dates
using Oceananigans
using BreezyBaroclinicInstability

Oceananigans.defaults.FloatType = Float32

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

Nλ = 360
Nφ = 160
Nz = 64
Δt = 30.0                              # [s] initial Δt; wizard adapts up to max_Δt
max_Δt = 150.0                         # [s] advective-CFL ceiling at polar Δx_min
cfl = 0.7                              # advective CFL target
stop_time = 6 * 3600.0                 # [s] 6 hours
save_interval = 3600.0                 # [s] save every 1 hour

output_dir = joinpath(@__DIR__, "output", "one_degree_spinup")
mkpath(output_dir)

# ═══════════════════════════════════════════════════════════════════════════
# Build model + set analytic IC
# ═══════════════════════════════════════════════════════════════════════════

@info "Building 1° model (Nλ=$Nλ, Nφ=$Nφ, Nz=$Nz, Δt=$(Δt)s)..." now(UTC)
arch = GPU()
model, _ = build_model(arch; Nλ, Nφ, Nz, Δt)

@info "Setting analytic IC..."
set_analytic_ic!(model)

any_nan(model) && error("NaN after analytic IC")
report_state(model; label="post-IC")

# ═══════════════════════════════════════════════════════════════════════════
# Simulation setup with Oceananigans infrastructure
# ═══════════════════════════════════════════════════════════════════════════

simulation = Simulation(model; Δt, stop_time)

conjure_time_step_wizard!(simulation; cfl, max_Δt, max_change=1.1)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

# JLD2 output writer for checkpoint fields
output_fields = Oceananigans.fields(model)
output_prefix = joinpath(output_dir, "fields")

simulation.output_writers[:fields] = JLD2Writer(model, output_fields;
    filename = output_prefix,
    schedule = TimeInterval(save_interval),
    overwrite_existing = true)

# Progress callback
wall_start = Ref(time_ns())

function progress(sim)
    wall = (time_ns() - wall_start[]) / 1e9
    iter = sim.model.clock.iteration
    t = sim.model.clock.time
    @info @sprintf("iter %d, t=%.0fs (%.2f h), wall=%.0fs", iter, t, t / 3600, wall)
    flush(stderr); flush(stdout)
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(500))

# ═══════════════════════════════════════════════════════════════════════════
# First step + run
# ═══════════════════════════════════════════════════════════════════════════

@info "First time step..." now(UTC)
Oceananigans.TimeSteppers.update_state!(model)
Oceananigans.TimeSteppers.time_step!(model, Δt)

any_nan(model) && error("NaN after first time step")
@info "First step complete" model.clock

@info "Starting simulation (stop_time=$(stop_time)s)..." now(UTC)
wall_start[] = time_ns()
run!(simulation)

# Save final checkpoint in the format expected by cascade scripts
final_path = joinpath(output_dir, "one_degree_final.jld2")
save_checkpoint(model, final_path; Δt)

@info "Done!" now(UTC) model.clock
