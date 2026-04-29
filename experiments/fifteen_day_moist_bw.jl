# 15-day moist baroclinic wave on the new acoustic substepper.
#
# Goal: probe how large the outer Δt can grow under the freshly overhauled
# split-explicit substepper in Breeze (branch glw/hevi-imex-docs). Uses
# Breeze defaults end-to-end — no bespoke spinup ramps, no max_Δt clamps.
#
# The model setup mirrors Breeze's `examples/moist_baroclinic_wave.jl`
# reference: H = 30 km, latitude = (-80, 80), uniform z, no top sponge,
# isothermal-T₀=250K reference state, NonEquilibriumCloudFormation with
# τ_relax = 200 s, ρθ-formulation surface fluxes. The Breeze example
# documents Δt = 20 s as the moist-microphysics-stiffness limit on this
# grid (Δt = 60 s NaNs in ~25 outer steps; Δt = 30 s grows a 2Δt
# oscillation). We start from 20 s and let the wizard probe upward.
#
# Launch: julia --project experiments/fifteen_day_moist_bw.jl
#
# Notes
# ─────
# * Δt envelope (min/max seen across the run) is reported each diag callback
#   so we can read off the largest stable outer step the wizard achieves.
# * NaNChecker fires every iteration (Memory: with TimeStepWizard +
#   substepping, anything coarser lets a NaN Δt propagate into
#   `compute_acoustic_substeps` as `InexactError: Int64(NaN)`).

using CUDA
using Dates
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.Diagnostics: AdvectiveCFL
using BreezyBaroclinicInstability

Oceananigans.defaults.FloatType = Float32
# Match DCMIP-2016 / Breeze moist BW reference constants so the grid arc length,
# Coriolis rate, and gravity all agree with what BBI's analytic IC uses.
Oceananigans.defaults.gravitational_acceleration = 9.80616
Oceananigans.defaults.planet_radius = 6371220.0
Oceananigans.defaults.planet_rotation_rate = 7.29212e-5

output_dir = joinpath(@__DIR__, "output", "fifteen_day_moist_bw")
mkpath(output_dir)

# ── Build model with Breeze defaults ─────────────────────────────────────
# `moist_baroclinic_instability_model` already wires in the new substepper
# defaults (forward_weight = 0.55, NoDivergenceDamping, ProportionalSubsteps,
# adaptive substep count). Microphysics is SaturationAdjustment by default —
# no cloud_formation_τ to tune.

@info "Building 1° model" now(UTC)
model = moist_baroclinic_instability_model(GPU(); Nλ = 360, Nφ = 160, Nz = 64)
@info "Built model" model

# ── Simulation: 15 days, default-armed wizard ────────────────────────────

simulation = Simulation(model; Δt = 20.0, stop_time = 15days)

# Probe upward from 20 s. cfl=0.7 is the WS-RK3 advective ceiling. A modest
# `max_change` keeps the wizard from doubling Δt in a single hop into a
# regime where the microphysics-dynamics coupling stiffens.
conjure_time_step_wizard!(simulation; cfl = 0.7, max_change = 1.05)

# NaNChecker every step — required when the wizard feeds Δt into the
# adaptive substep count.
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)
let cb = simulation.callbacks[:nan_checker]
    simulation.callbacks[:nan_checker] = Callback(cb.func, IterationInterval(1);
                                                   parameters = cb.parameters,
                                                   callsite   = cb.callsite)
end

simulation.output_writers[:fields] = JLD2Writer(model, Oceananigans.fields(model);
    filename = joinpath(output_dir, "fields"),
    schedule = TimeInterval(6hours),
    overwrite_existing = true)

# ── Diagnostics ──────────────────────────────────────────────────────────

wall_start  = Ref(time_ns())
wall_clock  = Ref(time_ns())
sim_clock   = Ref(0.0)
Δt_min_seen = Ref(Inf)
Δt_max_seen = Ref(0.0)
max_absu_seen = Ref(0.0)
max_absv_seen = Ref(0.0)
max_absw_seen = Ref(0.0)

cfl_op = AdvectiveCFL(simulation.Δt)

function diagnostics(sim)
    m = sim.model
    iter = m.clock.iteration
    t    = m.clock.time
    Δt   = sim.Δt

    Δt_min_seen[] = min(Δt_min_seen[], Δt)
    Δt_max_seen[] = max(Δt_max_seen[], Δt)

    wall_now    = time_ns()
    wall_total  = (wall_now - wall_start[]) / 1e9
    wall_window = (wall_now - wall_clock[]) / 1e9
    sim_window  = t - sim_clock[]
    sdpd_window = wall_window > 0 ? (sim_window / 86400) / (wall_window / 86400) : 0.0

    cfl = cfl_op(m)
    max_absu = maximum(abs, Oceananigans.interior(m.velocities.u))
    max_absv = maximum(abs, Oceananigans.interior(m.velocities.v))
    max_absw = maximum(abs, Oceananigans.interior(m.velocities.w))
    max_absu_seen[] = max(max_absu_seen[], max_absu)
    max_absv_seen[] = max(max_absv_seen[], max_absv)
    max_absw_seen[] = max(max_absw_seen[], max_absw)

    ρ_min, ρ_max = field_extrema(BreezyBaroclinicInstability.dynamics_density(m.dynamics))

    @info @sprintf("iter=%6d  t=%8.1fs (%5.2fd)  Δt=%6.2fs (env=[%5.2f, %6.2f])  CFL=%4.2f  wall=%7.0fs  SDPD=%5.2f  ρ=[%.3e,%.3e]  max|u|=%5.2f max|v|=%5.2f max|w|=%.2e",
                   iter, t, t/86400, Δt, Δt_min_seen[], Δt_max_seen[],
                   cfl, wall_total, sdpd_window,
                   ρ_min, ρ_max, max_absu, max_absv, max_absw)

    wall_clock[] = wall_now
    sim_clock[]  = t

    flush(stderr); flush(stdout)
end

simulation.callbacks[:diag] = Callback(diagnostics, IterationInterval(100))

# ── Run ──────────────────────────────────────────────────────────────────

@info "Starting 15-day run" now(UTC)
wall_start[] = time_ns()
wall_clock[] = time_ns()
sim_clock[]  = 0.0

run!(simulation)

total_wall = (time_ns() - wall_start[]) / 1e9
@info @sprintf("RUN COMPLETE: sim_time=%.0fs (%.2fd), wall=%.0fs (%.2fh), final Δt=%.2fs, Δt envelope=[%.2f, %.2f]s, peaks: max|u|=%.2f max|v|=%.2f max|w|=%.2e",
               simulation.model.clock.time, simulation.model.clock.time/86400,
               total_wall, total_wall/3600,
               simulation.Δt, Δt_min_seen[], Δt_max_seen[],
               max_absu_seen[], max_absv_seen[], max_absw_seen[])

final_path = joinpath(output_dir, "fifteen_day_final.jld2")
save_checkpoint(model, final_path; Δt = simulation.Δt)
@info "Saved final checkpoint" final_path now(UTC)
