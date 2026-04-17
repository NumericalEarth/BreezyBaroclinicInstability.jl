# 1/24-degree moist baroclinic wave — cascade from 1/16-degree IC.
#
# Uses NCCL distributed across 8 GPUs.
# Δt=0.8s (acoustic CFL binding at Nz=64, H=30km, τ_cloud=120s).
#
# Launch: ~/.julia/bin/mpiexecjl -n 8 --project julia -O0 experiments/twentyfourth_degree_cascade.jl

using Dates, MPI, JLD2, Printf, CUDA, NCCL
MPI.Init()

rank   = MPI.Comm_rank(MPI.COMM_WORLD)
nprocs = MPI.Comm_size(MPI.COMM_WORLD)
CUDA.device!(rank % length(CUDA.devices()))

rank == 0 && @info "Starting 1/24° NCCL cascade simulation" nprocs now(UTC)

using Oceananigans
using Oceananigans.DistributedComputations: Partition

const NCCLExt = Base.get_extension(Oceananigans, :OceananigansNCCLExt)
const NCCLDistributed = NCCLExt.NCCLDistributed

using BreezyBaroclinicInstability

Oceananigans.defaults.FloatType = Float32

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

Rx, Ry = 4, 2
@assert nprocs == Rx * Ry "Expected $(Rx * Ry) MPI ranks, got $nprocs"

Nλ = 8640
Nφ = 3840
Nz = 64
Δt = 0.8                              # Binding: vertical acoustic CFL
stop_time = 6 * 3600.0
save_interval = 3600.0

sst_anomaly = 2.0
cloud_formation_τ = 120.0
relaxation = (0.1, 1800)
cloud_damping = (0.1, 1800)

ic_path = joinpath(@__DIR__, "output", "sixteenth_degree_cascade", "sixteenth_degree_assembled.jld2")
isfile(ic_path) || error("IC file not found: $ic_path")

output_dir = joinpath(@__DIR__, "output", "twentyfourth_degree_cascade")
rank == 0 && mkpath(output_dir)
MPI.Barrier(MPI.COMM_WORLD)

# ═══════════════════════════════════════════════════════════════════════════
# Build model
# ═══════════════════════════════════════════════════════════════════════════

arch = NCCLDistributed(GPU(); partition = Partition(Rx, Ry, 1))

rank == 0 && @info "Building model (Nλ=$Nλ, Nφ=$Nφ, Nz=$Nz, Δt=$(Δt)s)..." now(UTC)
model, snapshots = build_model(arch;
    Nλ, Nφ, Nz, Δt,
    halo = (4, 4, 4),
    latitude = (-75, 75),
    cloud_formation_τ,
    sst_anomaly,
    relaxation,
    cloud_damping,
)

# ═══════════════════════════════════════════════════════════════════════════
# Load IC (interpolated from 1/16-degree)
# ═══════════════════════════════════════════════════════════════════════════

rank == 0 && @info "Loading IC..." ic_path now(UTC)
load_ic_interpolated!(model, ic_path; clamp_moisture=true)

any_nan(model) && error("NaN after IC load on rank $rank")
report_state(model; rank, label="post-IC")
check_density_positivity(model; rank)

copy_ic_snapshots!(snapshots, model)
rank == 0 && @info "IC-relaxation + cloud damping active"

MPI.Barrier(MPI.COMM_WORLD)

# ═══════════════════════════════════════════════════════════════════════════
# First step + simulation
# ═══════════════════════════════════════════════════════════════════════════

rank == 0 && @info "First time step..." now(UTC)
Oceananigans.TimeSteppers.update_state!(model)
Oceananigans.TimeSteppers.time_step!(model, Δt)
any_nan(model) && error("NaN after first step on rank $rank")
rank == 0 && @info "First step complete" model.clock

output_prefix = joinpath(output_dir, "fields_rank$rank")
save_iter_interval = round(Int, save_interval / Δt)

simulation = Simulation(model; Δt, stop_time)

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
    @info @sprintf("[r%d] iter %5d  t=%8.1fs  Δt=%.2f  wall=%6.1fs  ρ=[%.3e,%.3e] ρw=[%.2e,%.2e]",
                   rank, m.clock.iteration, m.clock.time, Δt, wall, ρ_min, ρ_max, ρw_min, ρw_max)
    any_nan(m) && error("NaN at iter $(m.clock.iteration) on rank $rank")
    flush(stderr); flush(stdout)
end
simulation.callbacks[:diag] = Callback(diagnostics, IterationInterval(100))

rank == 0 && @info "Starting simulation (stop_time=$(stop_time)s, save every $(save_iter_interval) iters)..." now(UTC)
wall_start[] = time_ns()
Oceananigans.run!(simulation)

rank == 0 && @info "Done!" now(UTC) model.clock
MPI.Finalize()
