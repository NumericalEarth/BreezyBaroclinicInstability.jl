# Direct reproduction of Breeze's `examples/moist_baroclinic_wave.jl` setup,
# inlined here so it can be run from the BreezyBaroclinicInstability.jl
# project environment without going through any BBI helpers.
#
# Purpose: isolate whether the moist BW instability is a BBI issue or a Breeze
# substepper issue.
#
# Result on Breeze branch glw/hevi-imex-docs (incl. the local rewritten
# substepper, 2026-04-26): max|ρw| doubles every outer step starting at
# step 4, reaches NaN by step 15 at Δt=20s. Float64 doesn't help.
# `KlempDivergenceDamping(coefficient=0.5)` doesn't help. Smaller Δt (e.g. 5s)
# delays but does not prevent the instability. The instability is therefore
# in the Breeze split-explicit substepper itself, not BBI.
#
# Launch: julia --project experiments/repro_substepper_moist_bw.jl

using CUDA
using Oceananigans
using CloudMicrophysics
using Breeze

Oceananigans.defaults.FloatType = Float32
Oceananigans.defaults.gravitational_acceleration = 9.80616
Oceananigans.defaults.planet_radius = 6371220.0
Oceananigans.defaults.planet_rotation_rate = 7.29212e-5

constants = ThermodynamicConstants(;
    gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
    dry_air_heat_capacity = 1004.5,
    dry_air_molar_mass = 8.314462618 / 287.0)

g   = constants.gravitational_acceleration
Rᵈ  = Breeze.dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
κ   = Rᵈ / cᵖᵈ
p₀  = 1e5
a   = Oceananigans.defaults.planet_radius
Ω   = Oceananigans.defaults.planet_rotation_rate

Nλ = 360; Nφ = 160; Nz = 64; H = 30e3

grid = LatitudeLongitudeGrid(GPU();
                             size = (Nλ, Nφ, Nz),
                             halo = (5, 5, 5),
                             longitude = (0, 360),
                             latitude = (-80, 80),
                             z = (0, H))

Tᴱ=310.0; Tᴾ=240.0; Tₘ=(Tᴱ+Tᴾ)/2; Γ=0.005; K=3; b=2; ε_v=0.608

function τ_and_integrals(z)
    Hₛ = Rᵈ * Tₘ / g
    η  = z / (b * Hₛ)
    e  = exp(-η^2)
    A = (Tₘ - Tᴾ) / (Tₘ * Tᴾ)
    C = (K + 2) / 2 * (Tᴱ - Tᴾ) / (Tᴱ * Tᴾ)
    τ₁  = exp(Γ * z / Tₘ) / Tₘ + A * (1 - 2η^2) * e
    τ₂  = C * (1 - 2η^2) * e
    ∫τ₁ = (exp(Γ * z / Tₘ) - 1) / Γ + A * z * e
    ∫τ₂ = C * z * e
    return τ₁, τ₂, ∫τ₁, ∫τ₂
end

F(φ)  = cosd(φ)^K - K / (K + 2) * cosd(φ)^(K + 2)
dF(φ) = cosd(φ)^(K - 1) - cosd(φ)^(K + 1)

function virtual_temperature(λ, φ, z)
    τ₁, τ₂, _, _ = τ_and_integrals(z)
    return 1 / (τ₁ - τ₂ * F(φ))
end
function pressure(λ, φ, z)
    _, _, ∫τ₁, ∫τ₂ = τ_and_integrals(z)
    return p₀ * exp(-g / Rᵈ * (∫τ₁ - ∫τ₂ * F(φ)))
end
density(λ, φ, z) = pressure(λ, φ, z) / (Rᵈ * virtual_temperature(λ, φ, z))

function specific_humidity(λ, φ, z)
    q₀ = 0.018; qₜ = 1e-12; φʷ = 2π / 9; pʷ = 34000
    p = pressure(λ, φ, z); η = p / p₀; φʳ = deg2rad(φ)
    q_trop = q₀ * exp(-(φʳ / φʷ)^4) * exp(-((η - 1) * p₀ / pʷ)^2)
    return ifelse(η > 0.1, q_trop, qₜ)
end
function temperature(λ, φ, z)
    Tᵥ = virtual_temperature(λ, φ, z); q = specific_humidity(λ, φ, z)
    return Tᵥ / (1 + ε_v * q)
end
function potential_temperature(λ, φ, z)
    p = pressure(λ, φ, z); T = temperature(λ, φ, z)
    return T * (p₀ / p)^κ
end
function zonal_velocity(λ, φ, z)
    _, _, _, ∫τ₂ = τ_and_integrals(z)
    T = temperature(λ, φ, z)
    U = g / a * K * ∫τ₂ * dF(φ) * T
    rcosφ = a * cosd(φ); Ωrcosφ = Ω * rcosφ
    u_b = -Ωrcosφ + sqrt(Ωrcosφ^2 + rcosφ * U)
    uₚ=1.0; rₚ=0.1; λₚ=π/9; φₚ=2π/9; zₚ=15000.0
    φʳ=deg2rad(φ); λʳ=deg2rad(λ)
    gc = acos(sin(φₚ)*sin(φʳ) + cos(φₚ)*cos(φʳ)*cos(λʳ-λₚ)) / rₚ
    taper = ifelse(z < zₚ, 1 - 3*(z/zₚ)^2 + 2*(z/zₚ)^3, 0.0)
    u_p = ifelse(gc < 1, uₚ * taper * exp(-gc^2), 0.0)
    return u_b + u_p
end

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

coriolis = HydrostaticSphericalCoriolis(rotation_rate=Ω)
T₀_ref = 250.0
θ_ref(z) = T₀_ref * exp(g * z / (cᵖᵈ * T₀_ref))

td = SplitExplicitTimeDiscretization(damping = Breeze.KlempDivergenceDamping(coefficient=0.5))
dynamics = CompressibleDynamics(td;
    surface_pressure = p₀, reference_potential_temperature = θ_ref)

τ_relax = 200.0
relaxation = ConstantRateCondensateFormation(1 / τ_relax)
cloud_formation = NonEquilibriumCloudFormation(relaxation, relaxation)
microphysics = OneMomentCloudMicrophysics(; cloud_formation)

Cᴰ = 1e-3; Uᵍ = 1e-2
T_surface(λ, φ) = virtual_temperature(λ, φ, 0.0)
ρu_bcs  = FieldBoundaryConditions(bottom = Breeze.BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T_surface))
ρv_bcs  = FieldBoundaryConditions(bottom = Breeze.BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T_surface))
ρθ_bcs  = FieldBoundaryConditions(bottom = Breeze.BulkSensibleHeatFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T_surface))
ρqᵛ_bcs = FieldBoundaryConditions(bottom = Breeze.BulkVaporFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T_surface))
boundary_conditions = (; ρu=ρu_bcs, ρv=ρv_bcs, ρθ=ρθ_bcs, ρqᵛ=ρqᵛ_bcs)

weno = WENO()
bp_weno = WENO(order=5, bounds=(0, 1))
momentum_advection = weno
scalar_advection = (ρθ=weno, ρqᵛ=bp_weno, ρqᶜˡ=bp_weno, ρqᶜⁱ=bp_weno, ρqʳ=bp_weno, ρqˢ=bp_weno)

model = AtmosphereModel(grid; dynamics, coriolis, microphysics, boundary_conditions,
                        thermodynamic_constants = constants,
                        momentum_advection, scalar_advection,
                        timestepper = :AcousticRungeKutta3)

set!(model, θ=potential_temperature, u=zonal_velocity, ρ=density, qᵛ=specific_humidity)

Oceananigans.TimeSteppers.update_state!(model)

Δt = 20.0
for n in 1:60
    Oceananigans.TimeSteppers.time_step!(model, Δt)
    ρ = Breeze.AtmosphereModels.dynamics_density(model.dynamics)
    has_nan = any(isnan, Oceananigans.interior(ρ))
    ρ_min = minimum(ρ)
    ρw_max = maximum(abs, Oceananigans.interior(model.momentum.ρw))
    u_max = maximum(abs, Oceananigans.interior(model.velocities.u))
    println("step ", n, "  NaN=", has_nan, "  min(ρ)=", ρ_min, "  max|ρw|=", ρw_max, "  max|u|=", u_max)
    has_nan && break
end
