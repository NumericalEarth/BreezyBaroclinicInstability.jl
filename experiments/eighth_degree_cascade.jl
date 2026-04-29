# 1/8-degree moist baroclinic wave — cascade from 1/4-degree IC.
#
# Uses NCCL distributed across multiple GPUs.
# Interpolates from 1/4-degree assembled checkpoint.
#
# Launch: ~/.julia/bin/mpiexecjl -n 8 --project julia -O0 experiments/eighth_degree_cascade.jl

using Dates, MPI, JLD2, Printf, CUDA, NCCL
MPI.Init()

rank   = MPI.Comm_rank(MPI.COMM_WORLD)
nprocs = MPI.Comm_size(MPI.COMM_WORLD)
CUDA.device!(rank % length(CUDA.devices()))

rank == 0 && @info "Starting 1/8° NCCL cascade simulation" nprocs now(UTC)

using Oceananigans
using Oceananigans.Units
using Oceananigans.DistributedComputations: Partition

const NCCLExt = Base.get_extension(Oceananigans, :OceananigansNCCLExt)
const NCCLDistributed = NCCLExt.NCCLDistributed

using BreezyBaroclinicInstability

Oceananigans.defaults.FloatType = Float32

Rx, Ry = 4, 2
@assert nprocs == Rx * Ry "Expected $(Rx * Ry) MPI ranks, got $nprocs"

ic_path = joinpath(@__DIR__, "output", "quarter_degree_cascade", "quarter_degree_assembled.jld2")
isfile(ic_path) || error("IC file not found: $ic_path")

output_dir = joinpath(@__DIR__, "output", "eighth_degree_cascade")
rank == 0 && mkpath(output_dir)
MPI.Barrier(MPI.COMM_WORLD)

arch = NCCLDistributed(GPU(); partition = Partition(Rx, Ry, 1))

rank == 0 && @info "Building 1/8° model" now(UTC)
model = moist_baroclinic_instability_model(arch;
    Nλ = 2880, Nφ = 1280, Nz = 64,
    set_ic = false)

rank == 0 && @info "Loading IC..." ic_path now(UTC)
load_ic_interpolated!(model, ic_path)
any_nan(model) && error("NaN after IC load on rank $rank")
report_state(model; rank, label="post-IC")

MPI.Barrier(MPI.COMM_WORLD)

simulation = Simulation(model; Δt = 3, stop_time = 6hours)
conjure_time_step_wizard!(simulation; cfl = 0.7)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

output_prefix = joinpath(output_dir, "fields_rank$rank")
simulation.output_writers[:fields] = JLD2Writer(model, Oceananigans.fields(model);
    filename = output_prefix,
    schedule = TimeInterval(1hour),
    overwrite_existing = true)

wall_start = Ref(time_ns())
function diagnostics(sim)
    m = sim.model
    ρ_min, ρ_max = field_extrema(BreezyBaroclinicInstability.dynamics_density(m.dynamics))
    wall = (time_ns() - wall_start[]) / 1e9
    @info @sprintf("[r%d] iter %d, t=%.0fs, Δt=%.1fs, wall=%.0fs, ρ=[%.3e,%.3e]",
                   rank, m.clock.iteration, m.clock.time, sim.Δt, wall, ρ_min, ρ_max)
    flush(stderr); flush(stdout)
end
simulation.callbacks[:diag] = Callback(diagnostics, IterationInterval(500))

rank == 0 && @info "Starting simulation" now(UTC)
wall_start[] = time_ns()
run!(simulation)

rank == 0 && @info "Done!" now(UTC) model.clock
MPI.Finalize()
