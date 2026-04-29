# Minimal 1° moist-BCI debug script.
#
# Strips away the cascade machinery (no interpolation, no cross-phase handoff,
# no relaxation forcing, no checkpoint loading). Just: build a 1° model, set
# the analytic IC, run with an adaptive time step, and log thoroughly every
# few iterations so we can see exactly when / where the instability develops.
#
# Run: julia --project experiments/debug_1deg.jl

using CUDA
using Dates
using Printf
using Oceananigans
using Oceananigans.Units
using BreezyBaroclinicInstability
using Oceananigans.Diagnostics: AdvectiveCFL
using CairoMakie

Oceananigans.defaults.FloatType = Float32

surface_fluxes = false
variant = surface_fluxes ? "sst_on" : "sst_off"
output_dir = joinpath(@__DIR__, "output", "debug_1deg_$variant")
mkpath(output_dir)

@info "Building 1° model" now(UTC) surface_fluxes variant
model = moist_baroclinic_instability_model(GPU();
    Nλ=360, Nφ=140, Nz=32, latitude=(-70, 70),      # DCMIP-2016 Test 161 reference: H=44 km, 30 levels
    sponge=(; rate=1/1000, width=8e3),               # Rayleigh sponge on ρw: top 8 km, τ=100 s
    viscous_sponge=(; z_bottom=34e3, ν_max=1e6),    # Horizontal-viscosity sponge: top 10 km, ν_max=1e5 m²/s
    surface_fluxes)

@info "Built model!" model
Δt = 225
simulation = Simulation(model; Δt, stop_time=2days) #stop_iteration=70)# top_time=14days)
conjure_time_step_wizard!(simulation, IterationInterval(10); cfl=0.1, max_Δt=Δt)

# Once the geostrophic-adjustment transient has radiated out (~1 min sim), release
# the wizard's Δt ceiling so the CFL constraint alone drives the step.
# NB: during `initialize!` Oceananigans invokes every TimeStep-callsite callback once,
# bypassing its schedule. Guard with an explicit time check so we only release once the
# clock has genuinely advanced past the target.
function double_max_Δt!(sim)
    sim.model.clock.iteration == 0 && return
    wiz = sim.callbacks[:time_step_wizard].func
    wiz.max_Δt *= 1.01
    @info "Increased wizard.max_Δt to $(prettytime(wiz.max_Δt)) \
            at t=$(sim.model.clock.time)s, iter=$(sim.model.clock.iteration)"
    return nothing
end

#simulation.callbacks[:release_Δt] = Callback(double_max_Δt!, SpecifiedTimes(1hours, 12hours))
#simulation.callbacks[:release_Δt] = Callback(double_max_Δt!, TimeInterval(2hours))

simulation.output_writers[:fields] = JLD2Writer(model, Oceananigans.fields(model);
    filename = joinpath(output_dir, "fields"),
    schedule = TimeInterval(6hours),
    overwrite_existing = true)

wall_start = Ref(time_ns())
diag_clock = Ref(time(simulation))
wall_clock = Ref(time_ns())

max_u_series = Float64[]
max_v_series = Float64[]
max_w_series = Float64[]
min_ρ_series = Float64[]
min_qᵛ_series = Float64[]

# Reports max(|f|) and also the (i, j, k) index where the max lives — useful
# to see whether the growth is at the top, near the pole, or along the jet.
function argmax_abs(field)
    interior_arr = Array(interior(field))
    idx = argmax(abs.(interior_arr))
    return abs(interior_arr[idx]), Tuple(idx)
end

function diagnostics(sim)
    m = sim.model
    iter = m.clock.iteration
    t = m.clock.time
    Δt = sim.Δt
    wall = (time_ns() - wall_clock[]) / 1e9
    simtime = time(sim) - diag_clock[]
    sdpd = wall > 0 ? simtime / wall : 0.0

    cfl = AdvectiveCFL(sim.Δt)(sim.model)
    max_u, idx_u = argmax_abs(m.velocities.u)
    max_v, idx_v = argmax_abs(m.velocities.v)
    max_w, idx_w = argmax_abs(m.velocities.w)
    min_ρ = minimum(model.dynamics.density)
    min_qᵛ = minimum(model.moisture_density)

    push!(max_u_series, max_u)
    push!(max_v_series, max_v)
    push!(max_w_series, max_w)
    push!(min_ρ_series, min_ρ)    
    push!(min_qᵛ_series, min_qᵛ)

    msg = @sprintf("iter=%6d, t=%s, Δt=%s, wall=%s, CFL=%3.2f",
                   iter, prettytime(t), prettytime(Δt), prettytime(wall), cfl)
    msg *= @sprintf(", min(ρ)=%9.3e, max|u|=%6.2f @ %s, max|v|=%6.2f @ %s, max|w|=%6.2f @ %s",
                   min_ρ, max_u, idx_u, max_v, idx_v, max_w, idx_w)

    @info msg

    wall_clock[] = time_ns()
    diag_clock[] = time(sim)

    any_nan(m) && error("NaN at iter $iter, t=$(t)s")
    flush(stderr); flush(stdout)
end

simulation.callbacks[:diag] = Callback(diagnostics, IterationInterval(10))

@info "Starting simulation" now(UTC)
wall_start[] = time_ns()
run!(simulation)

total_wall = (time_ns() - wall_start[]) / 1e9
@info @sprintf("DONE — wall=%.0fs (%.2fh), final Δt=%.1fs",
               total_wall, total_wall/3600, simulation.Δt)
@info @sprintf("peaks: max|u|=%.2f  max|v|=%.2f  max|w|=%.3e  min ρ=%.3e  min qᵛ=%.3e",
               maximum(max_u_series), maximum(max_v_series), maximum(max_w_series),
               minimum(min_ρ_series), minimum(min_qᵛ_series))

w5 = view(model.velocities.w, :, :, 5)
fig = Figure()
ax = Axis(fig[1, 1])
heatmap!(ax, w5)
save("w5.png", fig)