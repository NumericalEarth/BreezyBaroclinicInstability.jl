#####
##### Model Constructor
#####

"""
    moist_baroclinic_instability_model(arch; kwargs...) → model

Build a Breeze `AtmosphereModel` on a global `LatitudeLongitudeGrid` configured
for the DCMIP-2016 moist baroclinic wave (Test 1-1) and set its DCMIP analytic
initial condition before returning.

# Keyword arguments

- `Nλ = 360`: number of longitude points
- `Nφ = 160`: number of latitude points
- `Nz = 30`: number of vertical levels (DCMIP-2016 Test 161 reference resolution)
- `H = 44e3`: column height [m] (DCMIP-2016 Test 161 spec: z_top = 44 km)
- `halo = (4, 4, 4)`: halo size
- `latitude = (-75, 75)`: latitude range
- `sst_anomaly = 0.0`: SST anomaly [K] added to the balanced surface temperature
- `cloud_damping = nothing`: `(α0, T_decay)` tuple for cloud-condensate damping, or `nothing`
- `time_discretization = SplitExplicitTimeDiscretization()`: time discretization for the
  compressible dynamics. Default is acoustic substepping (Wicker–Skamarock RK3 with an
  adaptive number of substeps, derived from the horizontal acoustic CFL each outer step).
  Pass `ExplicitTimeStepping()` to recover the fully explicit (acoustic-CFL-limited) path.
- `z_stretching = 3.0`: hyperbolic-tangent vertical-grid stretching parameter σ. Larger σ
  packs more cells near the surface; typical values 2–4. Set to `0` for uniform Δz.
- `sponge = (; rate=1/600, width=7e3)`: top-of-domain Rayleigh sponge on ρw — a discrete
  `Forcing` with a linear mask ramping from 0 at `z = H - width` to 1 at `z = H`. Pass
  `nothing` to disable.
- `viscous_sponge = (; z_bottom=34e3, ν_max=1e5)`: top-of-domain horizontal-viscosity
  sponge via `HorizontalScalarDiffusivity`. `ν` ramps linearly from 0 at `z = z_bottom`
  to `ν_max` [m²/s] at `z = H`. Diffusive-CFL check: at 1° and φ = 70°, Δx ≈ 38 km,
  Δy ≈ 111 km ⇒ `ν·Δt·(1/Δx² + 1/Δy²) ≤ 1/2` allows `ν_max·Δt ≲ 6.5e8`. With `ν_max =
  1e5` m²/s that leaves Δt ≲ 6500 s of headroom. Pass `nothing` to disable.
- `surface_fluxes = true`: attach prescribed-SST bulk drag + sensible heat + vapor flux
  BCs at the bottom. Pass `false` for a pure-dynamics DCMIP-style run with no-flux BCs.
- `set_ic = true`: call `set_analytic_ic!(model)` before returning. Pass `false` if you
  plan to load an IC from a checkpoint.

# Example

```julia
using BreezyBaroclinicInstability
using Oceananigans

model = moist_baroclinic_instability_model(GPU(); Nλ=360, Nφ=160, Nz=64)
sim = Simulation(model; Δt=30, stop_time=14days)
conjure_time_step_wizard!(sim; cfl=0.7)
run!(sim)
```
"""
function moist_baroclinic_instability_model(arch;
                                            Nλ = 360,
                                            Nφ = 160,
                                            Nz = 64,
                                            H = 30e3,
                                            halo = (5, 5, 5),
                                            latitude = (-80, 80),
                                            sst_anomaly = 0.0,
                                            cloud_damping = nothing,
                                            time_discretization = SplitExplicitTimeDiscretization(),
                                            z_stretching = 0,
                                            sponge = nothing,
                                            viscous_sponge = nothing,
                                            cloud_formation_τ = 200.0,
                                            surface_fluxes = true,
                                            set_ic = true)

    z_faces = _vertical_faces(Nz, H, z_stretching)

    grid = LatitudeLongitudeGrid(arch;
                                 size = (Nλ, Nφ, Nz),
                                 halo,
                                 longitude = (0, 360),
                                 latitude,
                                 z = z_faces)

    coriolis = HydrostaticSphericalCoriolis()

    # Custom thermodynamic constants matching the DCMIP-2016 analytic IC values
    # (cᵖ = 1004.5 J/kg/K, Rᵈ = 287 J/kg/K, g = 9.80616). The model's EoS must
    # use the SAME constants the IC was built with, otherwise the analytic
    # balanced density and the model-recomputed pressure disagree at the rounding
    # level — a hydrostatic imbalance that excites the acoustic mode and drives
    # exponential ρw growth.
    FT = Oceananigans.defaults.FloatType
    constants = ThermodynamicConstants(FT;
                                       gravitational_acceleration = FT(gravity),
                                       dry_air_heat_capacity      = FT(cp_dry),
                                       dry_air_molar_mass         = FT(8.314462618 / Rd_dry))

    # Isothermal hydrostatic reference state at T₀ = 250 K, matching the Breeze
    # `examples/moist_baroclinic_wave.jl` reference. The substepper subtracts this
    # before computing slow tendencies so a hydrostatic rest atmosphere has zero
    # vertical drive at machine zero.
    g_ref   = FT(gravity)
    cᵖᵈ_ref = FT(cp_dry)
    T₀_ref  = FT(250)
    θ_ref(z) = T₀_ref * exp(g_ref * z / (cᵖᵈ_ref * T₀_ref))

    dynamics = CompressibleDynamics(
        time_discretization;
        surface_pressure = FT(p_ref),
        reference_potential_temperature = θ_ref,
    )

    # Cloud microphysics: one-moment mixed-phase with non-equilibrium cloud
    # formation. `cloud_formation_τ` is the relaxation timescale for vapor → cloud
    # condensate (and analogously for ice). The Breeze example sets τ = 200 s on
    # this 1° grid: the CloudMicrophysics default τ ≈ 40 s is shorter than typical
    # outer Δt and would let the explicit per-stage update overshoot saturation.
    ext = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
    rate = FT(1) / FT(cloud_formation_τ)
    relaxation = Breeze.Microphysics.ConstantRateCondensateFormation{FT}(rate)
    cloud_formation = NonEquilibriumCloudFormation(relaxation, relaxation)
    microphysics = ext.OneMomentCloudMicrophysics(; cloud_formation)

    # Advection: WENO(5) with bounds-preserving for moisture tracers.
    weno     = WENO(order = 5)
    weno_pos = WENO(order = 5, bounds = (0, 1))
    momentum_advection = weno
    scalar_advection = (ρθ   = weno,
                        ρqᵛ  = weno_pos,
                        ρqᶜˡ = weno_pos,
                        ρqᶜⁱ = weno_pos,
                        ρqʳ  = weno_pos,
                        ρqˢ  = weno_pos)

    # Prescribed-SST bulk surface fluxes (optional). With `surface_fluxes = false`
    # all bottom BCs default to no-flux, giving a pure-dynamics DCMIP-style test.
    boundary_conditions = if surface_fluxes
        Cᴰ = 1e-3
        Uᵍ = 1e-2
        T₀ = (λ, φ) -> surface_temperature(λ, φ) + sst_anomaly

        ρu_bcs  = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))
        ρv_bcs  = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))
        ρθ_bcs  = FieldBoundaryConditions(bottom = BulkSensibleHeatFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))
        ρqᵛ_bcs = FieldBoundaryConditions(bottom = BulkVaporFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))

        (; ρu=ρu_bcs, ρv=ρv_bcs, ρθ=ρθ_bcs, ρqᵛ=ρqᵛ_bcs)
    else
        NamedTuple()
    end

    # Cloud-condensate damping
    cloud_damp_forcing = if cloud_damping === nothing
        NamedTuple()
    else
        α0, T_decay = cloud_damping
        build_cloud_damping_forcing(; α0, T_decay)
    end

    # Top-of-domain sponge layer — linear ramp mask on ρw:
    #   mask(z) = 0 below (H − width), ramps linearly to 1 at z = H.
    # Discrete-form forcing; parameters are passed via the `parameters` kwarg so the
    # kernel stays isbits (closures over H / width / rate would break GPU compile).
    sponge_forcing = if sponge === nothing
        NamedTuple()
    else
        FT_sponge = Oceananigans.defaults.FloatType
        sponge_params = (z_bottom = FT_sponge(H - sponge.width),
                         width    = FT_sponge(sponge.width),
                         rate     = FT_sponge(sponge.rate))
        (; ρw = Forcing(_sponge_ρw_tendency;
                        discrete_form = true,
                        parameters    = sponge_params))
    end

    merged_forcing = merge(cloud_damp_forcing, sponge_forcing)

    # Viscous sponge: horizontal Laplacian viscosity that ramps linearly from 0 at
    # z = z_bottom to ν_max at z = H. Kills horizontal gradients of momentum in the
    # upper damping layer, absorbing short-horizontal-wavelength gravity waves before
    # they reflect off the rigid lid.
    closure = if viscous_sponge === nothing
        nothing
    else
        FT_visc = Oceananigans.defaults.FloatType
        visc_params = (z_bottom = FT_visc(viscous_sponge.z_bottom),
                       width    = FT_visc(H - viscous_sponge.z_bottom),
                       ν_max    = FT_visc(viscous_sponge.ν_max))
        HorizontalScalarDiffusivity(FT_visc;
                                    ν = _sponge_viscosity,
                                    discrete_form = true,
                                    parameters = visc_params)
    end

    model = AtmosphereModel(grid; dynamics, coriolis, momentum_advection,
                            scalar_advection, microphysics, boundary_conditions,
                            closure, forcing = merged_forcing,
                            thermodynamic_constants = constants)

    if set_ic
        set_analytic_ic!(model)
    end

    return model
end

# Discrete-form sponge tendency on ρw. `p = (z_bottom, width, rate)` is passed via
# `Forcing(...; parameters = p)`. Mask is a linear ramp: 0 below `z_bottom`,
# 1 at z = z_bottom + width.
@inline function _sponge_ρw_tendency(i, j, k, grid, clock, model_fields, p)
    z = znode(i, j, k, grid, Center(), Center(), Face())
    mask = max(zero(z), (z - p.z_bottom) / p.width)
    @inbounds return -p.rate * mask * model_fields.ρw[i, j, k]
end

# Discrete-form viscosity for the top-of-domain viscous sponge. Passed to
# `HorizontalScalarDiffusivity(ν=...; discrete_form=true, parameters=p)` with
# `p = (z_bottom, width, ν_max)`. Linear ramp: 0 below `z_bottom`, `ν_max` at top.
@inline function _sponge_viscosity(i, j, k, grid, ℓx, ℓy, ℓz, clock, fields, p)
    z = znode(i, j, k, grid, ℓx, ℓy, ℓz)
    mask = max(zero(z), (z - p.z_bottom) / p.width)
    return p.ν_max * mask
end

# Vertical face distribution: hyperbolic-tanh stretching that packs cells near
# z=0 when σ > 0, otherwise uniform. Returns a function suitable as the `z`
# argument of `LatitudeLongitudeGrid`.
#
# z(k) = H * (1 − tanh(σ * (1 − (k−1)/Nz)) / tanh(σ))
#
# At k=1: tanh(σ) / tanh(σ) = 1, so z=0.
# At k=Nz+1: tanh(0) / tanh(σ) = 0, so z=H.
# Δz is smallest near the surface (sech²(σ) ≪ 1) and largest near the top
# (sech²(0) = 1), giving fine boundary-layer resolution.
function _vertical_faces(Nz, H, σ)
    σ == 0 && return (0, H)
    return k -> H * (1 - tanh(σ * (1 - (k - 1) / Nz)) / tanh(σ))
end
