# Dry baroclinic wave repro — strip moisture/microphysics/surface fluxes,
# keep the same DCMIP-2016 setup as the moist BW.
#
# Purpose: confirm the substepper instability is not specific to moisture.
#
# Result on Breeze branch glw/hevi-imex-docs (2026-04-26): also NaNs.
# Δt=225s (the value documented as stable in Breeze/examples/baroclinic_wave.jl)
# blows up at step 11 (~41 minutes sim time). Δt=20s blows up at step 18
# (~6 minutes). The dry BW substepper is therefore also broken on this
# branch HEAD, with a slower-growing acoustic mode than the moist case.
#
# Launch: julia --project experiments/repro_substepper_dry_bw.jl

using CUDA
using Oceananigans
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
grid = LatitudeLongitudeGrid(GPU(); size=(Nλ,Nφ,Nz), halo=(5,5,5),
                             longitude=(0,360), latitude=(-80,80), z=(0,H))

Tᴱ=310.0; Tᴾ=240.0; Tₘ=(Tᴱ+Tᴾ)/2; Γ=0.005; K=3; b=2

function τ_and_integrals(z)
    Hₛ = Rᵈ * Tₘ / g; η = z / (b * Hₛ); e = exp(-η^2)
    A = (Tₘ-Tᴾ)/(Tₘ*Tᴾ); C = (K+2)/2 * (Tᴱ-Tᴾ)/(Tᴱ*Tᴾ)
    τ₁ = exp(Γ*z/Tₘ)/Tₘ + A*(1-2η^2)*e
    τ₂ = C*(1-2η^2)*e
    ∫τ₁ = (exp(Γ*z/Tₘ)-1)/Γ + A*z*e
    ∫τ₂ = C*z*e
    return τ₁, τ₂, ∫τ₁, ∫τ₂
end
F(φ)  = cosd(φ)^K - K/(K+2)*cosd(φ)^(K+2)
dF(φ) = cosd(φ)^(K-1) - cosd(φ)^(K+1)
function temperature(λ,φ,z)
    τ₁,τ₂,_,_ = τ_and_integrals(z); return 1/(τ₁ - τ₂*F(φ))
end
function pressure(λ,φ,z)
    _,_,∫τ₁,∫τ₂ = τ_and_integrals(z); return p₀*exp(-g/Rᵈ*(∫τ₁ - ∫τ₂*F(φ)))
end
density(λ,φ,z) = pressure(λ,φ,z)/(Rᵈ*temperature(λ,φ,z))
function potential_temperature(λ,φ,z)
    return temperature(λ,φ,z) * (p₀/pressure(λ,φ,z))^κ
end
function zonal_velocity(λ,φ,z)
    _,_,_,∫τ₂ = τ_and_integrals(z); T = temperature(λ,φ,z)
    U = g/a*K*∫τ₂*dF(φ)*T
    rcosφ = a*cosd(φ); Ωrcosφ = Ω*rcosφ
    u_b = -Ωrcosφ + sqrt(Ωrcosφ^2 + rcosφ*U)
    uₚ=1.0; rₚ=0.1; λₚ=π/9; φₚ=2π/9; zₚ=15000.0
    φʳ=deg2rad(φ); λʳ=deg2rad(λ)
    gc = acos(sin(φₚ)*sin(φʳ) + cos(φₚ)*cos(φʳ)*cos(λʳ-λₚ))/rₚ
    taper = ifelse(z<zₚ, 1-3*(z/zₚ)^2+2*(z/zₚ)^3, 0.0)
    u_p = ifelse(gc<1, uₚ*taper*exp(-gc^2), 0.0)
    return u_b + u_p
end

coriolis = HydrostaticSphericalCoriolis(rotation_rate=Ω)
T₀_ref = 250.0
θ_ref(z) = T₀_ref * exp(g*z/(cᵖᵈ*T₀_ref))

dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
    surface_pressure = p₀, reference_potential_temperature = θ_ref)

model = AtmosphereModel(grid; dynamics, coriolis,
    thermodynamic_constants = constants,
    advection = WENO(),
    timestepper = :AcousticRungeKutta3)

set!(model, θ=potential_temperature, u=zonal_velocity, ρ=density)

Oceananigans.TimeSteppers.update_state!(model)

# Try Δt=225s like the example claims is stable
Δt = 20.0
for n in 1:50
    Oceananigans.TimeSteppers.time_step!(model, Δt)
    ρ = Breeze.AtmosphereModels.dynamics_density(model.dynamics)
    has_nan = any(isnan, Oceananigans.interior(ρ))
    ρ_min = minimum(ρ)
    ρw_max = maximum(abs, Oceananigans.interior(model.momentum.ρw))
    u_max = maximum(abs, Oceananigans.interior(model.velocities.u))
    println("step ", n, " (t=", n*Δt, ")  NaN=", has_nan, "  min(ρ)=", ρ_min, "  max|ρw|=", ρw_max, "  max|u|=", u_max)
    has_nan && break
end
