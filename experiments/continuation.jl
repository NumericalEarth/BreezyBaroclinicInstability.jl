# Same-resolution continuation — restarts from an assembled checkpoint.
#
# No interpolation, no relaxation. Direct tile copy from assembled JLD2.
# Useful for extending a run that was cut short.
#
# Launch: ~/.julia/bin/mpiexecjl -n 8 --project julia -O0 experiments/continuation.jl

using Dates, MPI, JLD2, Printf, CUDA, NCCL
MPI.Init()

rank   = MPI.Comm_rank(MPI.COMM_WORLD)
nprocs = MPI.Comm_size(MPI.COMM_WORLD)
CUDA.device!(rank % length(CUDA.devices()))

rank == 0 && @info "Starting continuation run" nprocs now(UTC)

using Oceananigans
using Oceananigans.DistributedComputations: Partition

const NCCLExt = Base.get_extension(Oceananigans, :OceananigansNCCLExt)
const NCCLDistributed = NCCLExt.NCCLDistributed

using BreezyBaroclinicInstability

Oceananigans.defaults.FloatType = Float32

# ═══════════════════════════════════════════════════════════════════════════
# Configuration — edit these for your specific run
# ═══════════════════════════════════════════════════════════════════════════

Rx, Ry = 4, 2
@assert nprocs == Rx * Ry "Expected $(Rx * Ry) MPI ranks, got $nprocs"

Nλ = 8640
Nφ = 3840
Nz = 64
Δt = 0.8
total_continuation_time = 6 * 3600.0  # 6 hours more
save_interval = 3600.0

sst_anomaly = 2.0
cloud_formation_τ = 120.0

ic_path = joinpath(@__DIR__, "output", "twentyfourth_degree_cascade", "twentyfourth_degree_assembled.jld2")
isfile(ic_path) || error("IC not found: $ic_path")

output_dir = joinpath(@__DIR__, "output", "continuation")
rank == 0 && mkpath(output_dir)
MPI.Barrier(MPI.COMM_WORLD)

# ═══════════════════════════════════════════════════════════════════════════
# Build model (no relaxation for same-resolution restart)
# ═══════════════════════════════════════════════════════════════════════════

arch = NCCLDistributed(GPU(); partition = Partition(Rx, Ry, 1))

rank == 0 && @info "Building model..." now(UTC)
model, _ = build_model(arch;
    Nλ, Nφ, Nz, Δt,
    halo = (4, 4, 4),
    latitude = (-75, 75),
    cloud_formation_τ,
    sst_anomaly,
)

# ═══════════════════════════════════════════════════════════════════════════
# Load IC (direct tile copy, no interpolation)
# ═══════════════════════════════════════════════════════════════════════════

rank == 0 && @info "Loading IC (direct copy)..." ic_path now(UTC)
load_ic_direct!(model, ic_path; Rx, Ry, rank)

any_nan(model) && error("NaN after IC load on rank $rank")
report_state(model; rank, label="post-IC")
check_density_positivity(model; rank)
MPI.Barrier(MPI.COMM_WORLD)

# ═══════════════════════════════════════════════════════════════════════════
# First step + simulation
# ═══════════════════════════════════════════════════════════════════════════

rank == 0 && @info "First time step..." now(UTC)
Oceananigans.TimeSteppers.update_state!(model)
Oceananigans.TimeSteppers.time_step!(model, Δt)
any_nan(model) && error("NaN after first step on rank $rank")
rank == 0 && @info "First step complete" model.clock

# Set stop iteration relative to current iteration
prod_iters = round(Int, total_continuation_time / Δt)
final_iter = model.clock.iteration + prod_iters
save_iter_interval = round(Int, save_interval / Δt)

output_prefix = joinpath(output_dir, "fields_rank$rank")

simulation = Simulation(model; Δt, stop_iteration=final_iter)

function save_output(sim)
    iter = sim.model.clock.iteration
    iter > 0 && mod(iter, save_iter_interval) == 0 || return
    filepath = output_prefix * "_iter$(lpad(iter, 6, '0')).jld2"
    save_checkpoint(sim.model, filepath; Δt)
end
simulation.callbacks[:save] = Callback(save_output, IterationInterval(save_iter_interval))

wall_start = Ref(time_ns())
function diagnostics(sim)
    m = sim.model
    ρ_min, ρ_max = field_extrema(BreezyBaroclinicInstability.dynamics_density(m.dynamics))
    ρw_min, ρw_max = field_extrema(m.momentum.ρw)
    wall = (time_ns() - wall_start[]) / 1e9
    @info @sprintf("[r%d] iter %5d  t=%8.1fs  wall=%6.1fs  ρ=[%.3e,%.3e] ρw=[%.2e,%.2e]",
                   rank, m.clock.iteration, m.clock.time, wall, ρ_min, ρ_max, ρw_min, ρw_max)
    any_nan(m) && error("NaN at iter $(m.clock.iteration) on rank $rank")
    flush(stderr); flush(stdout)
end
simulation.callbacks[:diag] = Callback(diagnostics, IterationInterval(100))

rank == 0 && @info "Continuation: $(prod_iters) iters to iter $final_iter, save every $save_iter_interval" now(UTC)
wall_start[] = time_ns()
Oceananigans.run!(simulation)

rank == 0 && @info "Done!" now(UTC) model.clock
MPI.Finalize()
