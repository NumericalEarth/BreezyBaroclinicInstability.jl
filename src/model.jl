#####
##### Model Constructor
#####

"""
    build_model(arch; kwargs...) → (model, ic_snapshots)

Build a Breeze `AtmosphereModel` on a global `LatitudeLongitudeGrid` configured
for the DCMIP-2016 moist baroclinic wave (Test 1-1).

Returns `(model, ic_snapshots)` where `ic_snapshots` is `nothing` if no
relaxation forcing is requested, or a NamedTuple of snapshot fields to be
filled after IC loading via `copy_ic_snapshots!(snapshots, model)`.

# Keyword arguments

- `Nλ = 360`: number of longitude points
- `Nφ = 160`: number of latitude points
- `Nz = 64`: number of vertical levels
- `H = 30e3`: column height [m]
- `Δt = nothing`: time step [s]; if `nothing`, auto-computed from acoustic CFL
- `halo = (4, 4, 4)`: halo size
- `latitude = (-80, 80)`: latitude range
- `cloud_formation_τ = 120.0`: cloud condensation/freezing timescale [s]
- `sst_anomaly = 0.0`: SST anomaly [K] added to the balanced surface temperature
- `relaxation = nothing`: `(α0, T_decay)` tuple for IC-relaxation forcing, or `nothing`
- `cloud_damping = nothing`: `(α0, T_decay)` tuple for cloud-condensate damping, or `nothing`

# Example

```julia
using BreezyBaroclinicInstability
using Oceananigans

model, snapshots = build_model(GPU();
    Nλ = 5760, Nφ = 2560, Nz = 64,
    Δt = 0.5,
    relaxation = (0.1, 1800),
    cloud_damping = (0.1, 1800))

load_ic_interpolated!(model, "eighth_degree_checkpoint.jld2")
copy_ic_snapshots!(snapshots, model)

sim = Simulation(model; Δt = 0.5, stop_time = 12 * 3600)
run!(sim)
```
"""
function build_model(arch;
                     Nλ = 360,
                     Nφ = 160,
                     Nz = 64,
                     H = 30e3,
                     Δt = nothing,
                     halo = (4, 4, 4),
                     latitude = (-80, 80),
                     cloud_formation_τ = 120.0,
                     sst_anomaly = 0.0,
                     relaxation = nothing,
                     cloud_damping = nothing)

    # Auto time step from acoustic CFL
    Δt_value = isnothing(Δt) ? 0.5 * (H / Nz) / 330.0 : Δt

    grid = LatitudeLongitudeGrid(arch;
                                 size = (Nλ, Nφ, Nz),
                                 halo,
                                 longitude = (0, 360),
                                 latitude,
                                 z = (0, H))

    coriolis = SphericalCoriolis()

    dynamics = CompressibleDynamics(
        ExplicitTimeStepping();
        surface_pressure = p_ref,
        reference_potential_temperature = theta_reference,
    )

    # Cloud microphysics
    ext = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
    FT = Oceananigans.defaults.FloatType
    rate = FT(1) / FT(cloud_formation_τ)
    cf = Breeze.Microphysics.ConstantRateCondensateFormation{FT}(rate)
    cloud_formation = NonEquilibriumCloudFormation(cf, cf)
    microphysics = ext.OneMomentCloudMicrophysics(; cloud_formation)

    # Advection: WENO(5) with bounds-preserving for moisture tracers
    weno     = WENO(order = 5)
    weno_pos = WENO(order = 5, bounds = (0, 1))
    momentum_advection = weno
    scalar_advection = (ρθ   = weno,
                        ρqᵛ  = weno_pos,
                        ρqᶜˡ = weno_pos,
                        ρqᶜⁱ = weno_pos,
                        ρqʳ  = weno_pos,
                        ρqˢ  = weno_pos)

    # Prescribed-SST bulk surface fluxes
    Cᴰ = 1e-3
    Uᵍ = 1e-2
    T₀ = (λ, φ) -> surface_temperature(λ, φ) + sst_anomaly

    ρu_bcs  = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))
    ρv_bcs  = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))
    ρe_bcs  = FieldBoundaryConditions(bottom = BulkSensibleHeatFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))
    ρqᵛ_bcs = FieldBoundaryConditions(bottom = BulkVaporFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))

    boundary_conditions = (; ρu=ρu_bcs, ρv=ρv_bcs, ρe=ρe_bcs, ρqᵛ=ρqᵛ_bcs)

    # IC-relaxation forcing
    ic_relax_forcing, ic_snapshots = if relaxation === nothing
        NamedTuple(), nothing
    else
        α0, T_decay = relaxation
        build_ic_relaxation_forcing(grid; α0, T_decay)
    end

    # Cloud-condensate damping
    cloud_damp_forcing = if cloud_damping === nothing
        NamedTuple()
    else
        α0, T_decay = cloud_damping
        build_cloud_damping_forcing(; α0, T_decay)
    end

    merged_forcing = merge(ic_relax_forcing, cloud_damp_forcing)

    model = AtmosphereModel(grid; dynamics, coriolis, momentum_advection,
                            scalar_advection, microphysics, boundary_conditions,
                            forcing = merged_forcing)

    FT_grid = eltype(grid)
    model.clock.last_Δt = FT_grid(Δt_value)

    return model, ic_snapshots
end
