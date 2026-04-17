module BreezyBaroclinicInstability

using Adapt: Adapt
using CUDA
using JLD2
using Printf
using KernelAbstractions: @kernel, @index

using Oceananigans
using Oceananigans.BoundaryConditions: FieldBoundaryConditions
using Oceananigans.Fields: CenterField, XFaceField, YFaceField, ZFaceField, Field
using Oceananigans.Grids: λnode, φnode, znode, Center, Face, LatitudeLongitudeGrid

using Breeze
using Breeze: AtmosphereModel, CompressibleDynamics,
               ExplicitTimeStepping, SplitExplicitTimeDiscretization
using Breeze: BulkDrag, BulkSensibleHeatFlux, BulkVaporFlux
using Breeze.AtmosphereModels: dynamics_density, specific_prognostic_moisture
using Breeze.Microphysics: NonEquilibriumCloudFormation
using CloudMicrophysics

# ── Exports ──────────────────────────────────────────────────────────────

# Model construction
export build_model

# Initial conditions
export set_analytic_ic!
export load_ic_interpolated!, load_ic_direct!
export copy_ic_snapshots!

# Forcing
export build_ic_relaxation_forcing, build_cloud_damping_forcing

# IC reference functions (needed by model constructor and experiment scripts)
export theta_reference, surface_temperature
export initial_theta, initial_density, initial_zonal_wind, initial_moisture

# Diagnostics
export any_nan, field_extrema, report_state, check_density_positivity

# Checkpoint I/O
export save_checkpoint

# ── Includes ─────────────────────────────────────────────────────────────

include("constants.jl")
include("balanced_state.jl")
include("initial_conditions.jl")
include("forcing.jl")
include("ic_loading.jl")
include("model.jl")
include("diagnostics.jl")
include("checkpoint.jl")

end # module BreezyBaroclinicInstability
