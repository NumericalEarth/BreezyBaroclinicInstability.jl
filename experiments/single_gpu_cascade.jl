# Single-GPU resolution cascade: 1° → 1/4° → 1/8°
#
# Runs the full moist baroclinic instability cascade on one H200 GPU.
# Each phase saves a checkpoint that the next phase loads and interpolates.
#
#   Phase 1: 1°   for 14 days  (analytic IC)
#   Phase 2: 1/4° for 2 days   (interpolated from 1° with relaxation)
#   Phase 3: 1/8° for 1 day    (interpolated from 1/4° with relaxation)
#
# Set CASCADE_MINI=true for a quick validation run with reduced durations.
#
# Launch: julia --project experiments/single_gpu_cascade.jl

using CUDA
using Dates
using Printf
using Oceananigans
using BreezyBaroclinicInstability

Oceananigans.defaults.FloatType = Float32

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

const MINI = get(ENV, "CASCADE_MINI", "false") == "true"

MINI && @info "MINI CASCADE MODE: reduced durations for validation"

struct PhaseConfig
    label             :: String
    Nλ                :: Int
    Nφ                :: Int
    Nz                :: Int
    Δt                :: Float64   # initial outer Δt; TimeStepWizard adapts up to max_Δt
    max_Δt            :: Float64   # upper clamp for adaptive outer Δt
    cfl               :: Float64   # target advective CFL for TimeStepWizard
    stop_time         :: Float64
    save_interval     :: Float64
    diag_interval     :: Int
    relaxation        :: Union{Nothing, Tuple{Float64, Float64}}
    cloud_damping     :: Union{Nothing, Tuple{Float64, Float64}}
    cloud_formation_τ :: Float64
    sst_anomaly       :: Float64
    clamp_moisture    :: Bool
end

# `Δt` is the *initial* outer step; `conjure_time_step_wizard!` then adapts it from the
# advective CFL, targeting `cfl` and clamping at `max_Δt`. Under the 2026-04-17 stability
# fixes (H=45km domain, stretched-in-z grid, top sponge layer, moist cloud_formation_τ=600)
# the wizard drives Δt from CFL alone — `max_Δt` is only a guardrail.

phases = [
    # Phase 1: 1-degree, analytic IC, 14 days (or 6h mini)
    PhaseConfig("1deg",
        360, 160, 64,                              # grid
        20.0,                                      # initial Δt
        120.0,                                     # max Δt guardrail (wizard drives from CFL)
        0.5,                                       # CFL (dropped from 0.7 — safety margin)
        MINI ? 6*3600.0 : 14*86400.0,              # stop_time
        MINI ? 3600.0 : 86400.0,                   # save every hour (mini) or day
        MINI ? 200 : 500,                          # diag interval
        nothing,                                   # no relaxation
        nothing,                                   # no cloud damping
        600.0,                                     # cloud_formation_τ
        0.0,                                       # no SST anomaly
        false),                                    # no moisture clamping

    # Phase 2: 1/4-degree, from 1° checkpoint, 2 days (or 1h mini)
    PhaseConfig("quarter_deg",
        1440, 640, 64,
        5.0,                                       # initial Δt
        30.0,                                      # max Δt guardrail
        0.5,                                       # CFL
        MINI ? 3600.0 : 2*86400.0,
        MINI ? 600.0 : 3600.0,
        MINI ? 200 : 300,
        (0.1, 1800.0),                             # relaxation: 0.1 s⁻¹ over 30 min
        (0.1, 1800.0),                             # cloud damping
        600.0,                                     # cloud_formation_τ (stable ratio vs Δt)
        0.0,
        true),                                     # clamp moisture after interpolation

    # Phase 3: 1/8-degree, from 1/4° checkpoint, 1 day (or 30min mini)
    PhaseConfig("eighth_deg",
        2880, 1280, 64,
        3.0,                                       # initial Δt
        15.0,                                      # max Δt guardrail
        0.5,                                       # CFL
        MINI ? 1800.0 : 86400.0,
        MINI ? 600.0 : 3600.0,
        MINI ? 200 : 300,
        (0.1, 1800.0),
        (0.1, 1800.0),
        600.0,
        0.0,
        true),
]

output_root = joinpath(@__DIR__, "output", MINI ? "cascade_mini" : "cascade")
mkpath(output_root)

# ═══════════════════════════════════════════════════════════════════════════
# Phase runner
# ═══════════════════════════════════════════════════════════════════════════

function run_phase(config::PhaseConfig, output_root::String;
                   ic_path::Union{Nothing, String} = nothing)

    label = config.label
    phase_dir = joinpath(output_root, label)
    mkpath(phase_dir)

    @info "━━━ Phase: $label ━━━" config.Nλ config.Nφ config.Nz config.Δt config.stop_time now(UTC)

    # ── Build model ──────────────────────────────────────────────────────

    arch = GPU()
    model, snapshots = build_model(arch;
        Nλ              = config.Nλ,
        Nφ              = config.Nφ,
        Nz              = config.Nz,
        Δt              = config.Δt,
        halo            = (4, 4, 4),
        latitude        = (-80, 80),
        cloud_formation_τ = config.cloud_formation_τ,
        sst_anomaly     = config.sst_anomaly,
        relaxation      = config.relaxation,
        cloud_damping   = config.cloud_damping)

    @info "[$label] Model built" size(model.grid) eltype(model.grid)

    # ── Initial conditions ───────────────────────────────────────────────

    if ic_path === nothing
        @info "[$label] Setting analytic IC..."
        set_analytic_ic!(model)
    else
        @info "[$label] Loading IC from checkpoint..." ic_path
        load_ic_interpolated!(model, ic_path; clamp_moisture=config.clamp_moisture)
    end

    any_nan(model) && error("[$label] NaN after IC load")
    check_density_positivity(model)
    report_state(model; label="$label post-IC")

    if snapshots !== nothing
        copy_ic_snapshots!(snapshots, model)
        @info "[$label] IC-relaxation snapshots frozen"
    end

    # ── First time step (required for AtmosphereModel before run!) ───────

    @info "[$label] First time step..."
    Oceananigans.TimeSteppers.update_state!(model)
    Oceananigans.TimeSteppers.time_step!(model, config.Δt)

    any_nan(model) && error("[$label] NaN after first time step")
    @info "[$label] First step OK" model.clock

    # ── Simulation with JLD2OutputWriter ─────────────────────────────────

    simulation = Simulation(model; Δt=config.Δt, stop_time=config.stop_time)

    # Adaptive outer Δt from advective CFL; acoustic substepper inside handles sound waves.
    conjure_time_step_wizard!(simulation;
                              cfl       = config.cfl,
                              max_Δt    = config.max_Δt,
                              max_change = 1.1)
    @info "[$label] TimeStepWizard armed" cfl=config.cfl max_Δt=config.max_Δt initial_Δt=config.Δt

    # NaN checker fires every iteration so we catch blowups at their source (rather than
    # letting the wizard propagate a NaN Δt into compute_acoustic_substeps → InexactError).
    Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)
    let cb = simulation.callbacks[:nan_checker]
        simulation.callbacks[:nan_checker] = Callback(cb.func, IterationInterval(1); parameters=cb.parameters, callsite=cb.callsite)
    end

    # Periodic field output via Oceananigans JLD2OutputWriter
    output_fields = Oceananigans.fields(model)
    simulation.output_writers[:fields] = JLD2Writer(model, output_fields;
        filename = joinpath(phase_dir, "fields"),
        schedule = TimeInterval(config.save_interval),
        overwrite_existing = true)

    # Diagnostics callback
    wall_start = Ref(time_ns())

    function diagnostics(sim)
        m = sim.model
        iter = m.clock.iteration
        t    = m.clock.time
        Δt_now = sim.Δt
        wall = (time_ns() - wall_start[]) / 1e9
        sdpd = wall > 0 ? (t / 86400) / (wall / 86400) : 0.0

        ρ_min, ρ_max   = field_extrema(BreezyBaroclinicInstability.dynamics_density(m.dynamics))
        ρw_min, ρw_max = field_extrema(m.momentum.ρw)

        @info @sprintf("[%s] iter=%6d  t=%9.1fs (%5.2fd)  Δt=%5.1fs  wall=%7.1fs  SDPD=%5.1f  ρ=[%.3e,%.3e]  ρw=[%.2e,%.2e]",
                       label, iter, t, t/86400, Δt_now, wall, sdpd, ρ_min, ρ_max, ρw_min, ρw_max)

        any_nan(m) && error("[$label] NaN at iter $iter, t=$(t)s")
        flush(stderr); flush(stdout)
    end

    simulation.callbacks[:diag] = Callback(diagnostics, IterationInterval(config.diag_interval))

    # ── Run ──────────────────────────────────────────────────────────────

    @info "[$label] Starting simulation..." config.stop_time now(UTC)
    wall_start[] = time_ns()
    Oceananigans.run!(simulation)

    total_wall = (time_ns() - wall_start[]) / 1e9
    @info @sprintf("[%s] COMPLETE: sim_time=%.0fs (%.1f days), wall_time=%.0fs (%.2f hours)",
                   label, config.stop_time, config.stop_time/86400, total_wall, total_wall/3600)

    # ── Save cascade handoff checkpoint ──────────────────────────────────

    checkpoint_path = joinpath(phase_dir, "$(label)_final.jld2")
    save_checkpoint(model, checkpoint_path; Δt=simulation.Δt)

    report_state(model; label="$label final")

    return checkpoint_path, model
end

function release_gpu!(model_ref)
    model_ref[] = nothing
    GC.gc(true)
    GC.gc(false)
    GC.gc(true)
    CUDA.reclaim()
    @info "GPU memory released" free_GB=round(CUDA.available_memory() / 1e9, digits=1)
end

# ═══════════════════════════════════════════════════════════════════════════
# Run cascade
# ═══════════════════════════════════════════════════════════════════════════

@info "Starting resolution cascade" MINI now(UTC)
cascade_start = time_ns()

# ── Phase 1: 1-degree spinup ────────────────────────────────────────────

checkpoint_1, model_1 = run_phase(phases[1], output_root)

model_ref = Ref{Any}(model_1)
model_1 = nothing
release_gpu!(model_ref)

# ── Phase 2: 1/4-degree cascade ─────────────────────────────────────────

checkpoint_2, model_2 = run_phase(phases[2], output_root; ic_path=checkpoint_1)

model_ref = Ref{Any}(model_2)
model_2 = nothing
release_gpu!(model_ref)

# ── Phase 3: 1/8-degree cascade ─────────────────────────────────────────

checkpoint_3, model_3 = run_phase(phases[3], output_root; ic_path=checkpoint_2)

cascade_wall = (time_ns() - cascade_start) / 1e9
@info @sprintf("CASCADE COMPLETE: total wall time = %.0fs (%.2f hours)", cascade_wall, cascade_wall/3600)
@info "Final checkpoint" checkpoint_3
@info "Done!" now(UTC)
