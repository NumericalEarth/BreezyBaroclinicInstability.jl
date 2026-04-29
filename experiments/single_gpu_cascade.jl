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
using Oceananigans.Units
using BreezyBaroclinicInstability

Oceananigans.defaults.FloatType = Float32
const MINI = get(ENV, "CASCADE_MINI", "false") == "true"
MINI && @info "MINI CASCADE MODE: reduced durations for validation"

struct PhaseConfig
    label             :: String
    Nλ                :: Int
    Nφ                :: Int
    Nz                :: Int
    Δt                :: Float64   # initial outer Δt; TimeStepWizard adapts up to max_Δt
    max_Δt            :: Float64   # upper clamp for adaptive outer Δt (post-spinup)
    spinup_max_Δt     :: Float64   # tighter guardrail during initial `spinup_duration`
    spinup_duration   :: Float64   # [s] time at which wizard.max_Δt is raised
    cfl               :: Float64   # target advective CFL for TimeStepWizard
    stop_time         :: Float64
    save_interval     :: Float64
    diag_interval     :: Int
    cloud_damping     :: Union{Nothing, Tuple{Float64, Float64}}
    sst_anomaly       :: Float64
    clamp_moisture    :: Bool
end

# `Δt` is the *initial* outer step; `conjure_time_step_wizard!` then adapts it from the
# advective CFL, targeting `cfl` and clamping at `max_Δt`. Under the 2026-04-17 stability
# fixes (H=45km domain, stretched-in-z grid, top sponge layer, moist cloud_formation_τ=600)
# the wizard drives Δt from CFL alone — `max_Δt` is only a guardrail.

phases = [
    # Phase 1: 1-degree, analytic IC, 14 days (or 6h mini).
    # Spinup (~7 days) runs with a conservative max_Δt guardrail; after the
    # BCI has developed the wizard is allowed to push Δt up without a cap
    # (CFL alone limits it).
    PhaseConfig("1deg",
        360, 160, 64,                              # grid
        15.0,                                      # initial Δt
        Inf,                                       # max Δt post-spinup (CFL-limited)
        30.0,                                      # spinup max Δt guardrail (aggressive — prev 120 still blew up at t≈39h)
        7*86400.0,                                 # spinup duration [s]
        0.7,                                       # CFL target
        MINI ? 6*3600.0 : 14*86400.0,              # stop_time
        MINI ? 3600.0 : 86400.0,                   # save every hour (mini) or day
        MINI ? 200 : 500,                          # diag interval
        nothing,                                   # no cloud damping
        0.0,                                       # no SST anomaly
        false),                                    # no moisture clamping

    # Phase 2: 1/4-degree, from 1° checkpoint, 2 days (or 1h mini).
    # Starts from a post-spinup state, so no Δt ramp — wizard runs at full CFL.
    PhaseConfig("quarter_deg",
        1440, 640, 64,
        10.0,                                      # initial Δt
        Inf,                                       # no max_Δt cap — wizard finds it
        Inf,                                       # no spinup ramp
        0.0,                                       # spinup duration
        0.7,                                       # CFL target
        MINI ? 3600.0 : 2*86400.0,
        MINI ? 600.0 : 3600.0,
        MINI ? 200 : 300,
        (0.1, 1800.0),                             # cloud damping
        0.0,
        true),                                     # clamp moisture after interpolation

    # Phase 3: 1/8-degree, from 1/4° checkpoint, 1 day (or 30min mini).
    PhaseConfig("eighth_deg",
        2880, 1280, 64,
        5.0,                                       # initial Δt
        Inf,                                       # no max_Δt cap
        Inf,                                       # no spinup ramp
        0.0,
        0.7,                                       # CFL target
        MINI ? 1800.0 : 86400.0,
        MINI ? 600.0 : 3600.0,
        MINI ? 200 : 300,
        (0.1, 1800.0),                             # cloud damping
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
    # Phase 1 uses the DCMIP analytic IC set by the builder (set_ic = true default).
    # Phases 2–3 load an interpolated IC from the previous phase's checkpoint.

    arch = GPU()
    model = moist_baroclinic_instability_model(arch;
        Nλ              = config.Nλ,
        Nφ              = config.Nφ,
        Nz              = config.Nz,
        sst_anomaly     = config.sst_anomaly,
        cloud_damping   = config.cloud_damping,
        set_ic          = ic_path === nothing)

    @info "[$label] Model built" size(model.grid) eltype(model.grid)

    if ic_path !== nothing
        @info "[$label] Loading IC from checkpoint..." ic_path
        load_ic_interpolated!(model, ic_path; clamp_moisture=config.clamp_moisture)
    end

    any_nan(model) && error("[$label] NaN after IC load")
    check_density_positivity(model)
    report_state(model; label="$label post-IC")

    # ── Simulation with JLD2Writer ───────────────────────────────────────

    simulation = Simulation(model; Δt=config.Δt, stop_time=config.stop_time)

    # Adaptive outer Δt from advective CFL. During the first `spinup_duration`
    # seconds, clamp wizard.max_Δt to the tighter `spinup_max_Δt` so the BCI
    # has a chance to develop smoothly; afterwards raise to `max_Δt` (typically Inf).
    initial_max_Δt = config.spinup_duration > 0 ? config.spinup_max_Δt : config.max_Δt
    conjure_time_step_wizard!(simulation;
                              cfl       = config.cfl,
                              max_Δt    = initial_max_Δt,
                              max_change = 1.1)
    @info "[$label] TimeStepWizard armed" cfl=config.cfl initial_max_Δt=initial_max_Δt post_spinup_max_Δt=config.max_Δt spinup_duration=config.spinup_duration

    # Raise wizard.max_Δt once the spinup phase is over.
    if config.spinup_duration > 0
        function bump_max_Δt(sim)
            if sim.model.clock.time >= config.spinup_duration
                wiz = sim.callbacks[:time_step_wizard].func
                if wiz.max_Δt != config.max_Δt
                    @info "[$label] Spinup complete at t=$(sim.model.clock.time)s — raising wizard.max_Δt from $(wiz.max_Δt) → $(config.max_Δt)"
                    wiz.max_Δt = config.max_Δt
                end
            end
        end
        simulation.callbacks[:bump_max_Δt] = Callback(bump_max_Δt, IterationInterval(10))
    end

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

    # Diagnostics callback. Tracks Δt min/max AND max(|u|), max(|v|), max(|w|)
    # envelopes across the whole phase so we can report adaptive-Δt behaviour
    # and flow magnitudes at the end.
    wall_start = Ref(time_ns())
    Δt_min_seen = Ref(Inf)
    Δt_max_seen = Ref(0.0)
    max_absu_seen = Ref(0.0)
    max_absv_seen = Ref(0.0)
    max_absw_seen = Ref(0.0)

    function diagnostics(sim)
        m = sim.model
        iter = m.clock.iteration
        t    = m.clock.time
        Δt_now = sim.Δt
        Δt_min_seen[] = min(Δt_min_seen[], Δt_now)
        Δt_max_seen[] = max(Δt_max_seen[], Δt_now)
        wall = (time_ns() - wall_start[]) / 1e9
        sdpd = wall > 0 ? (t / 86400) / (wall / 86400) : 0.0

        ρ_min, ρ_max   = field_extrema(BreezyBaroclinicInstability.dynamics_density(m.dynamics))
        max_absu = maximum(abs, Oceananigans.interior(m.velocities.u))
        max_absv = maximum(abs, Oceananigans.interior(m.velocities.v))
        max_absw = maximum(abs, Oceananigans.interior(m.velocities.w))
        max_absu_seen[] = max(max_absu_seen[], max_absu)
        max_absv_seen[] = max(max_absv_seen[], max_absv)
        max_absw_seen[] = max(max_absw_seen[], max_absw)

        @info @sprintf("[%s] iter=%6d  t=%9.1fs (%5.2fd)  Δt=%5.1fs (min=%.1f max=%.1f)  wall=%7.1fs  SDPD=%5.1f  ρ=[%.3e,%.3e]  max|u|=%.2f max|v|=%.2f max|w|=%.3e",
                       label, iter, t, t/86400, Δt_now, Δt_min_seen[], Δt_max_seen[],
                       wall, sdpd, ρ_min, ρ_max, max_absu, max_absv, max_absw)

        any_nan(m) && error("[$label] NaN at iter $iter, t=$(t)s")
        flush(stderr); flush(stdout)
    end

    simulation.callbacks[:diag] = Callback(diagnostics, IterationInterval(config.diag_interval))

    # ── Run ──────────────────────────────────────────────────────────────

    @info "[$label] Starting simulation..." config.stop_time now(UTC)
    wall_start[] = time_ns()
    Oceananigans.run!(simulation)

    total_wall = (time_ns() - wall_start[]) / 1e9
    @info @sprintf("[%s] COMPLETE: sim_time=%.0fs (%.1f days), wall_time=%.0fs (%.2f hours), final Δt=%.1fs, Δt envelope=[%.1f, %.1f]s, max|u|=%.2f, max|v|=%.2f, max|w|=%.3e",
                   label, config.stop_time, config.stop_time/86400, total_wall, total_wall/3600,
                   simulation.Δt, Δt_min_seen[], Δt_max_seen[],
                   max_absu_seen[], max_absv_seen[], max_absw_seen[])

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
