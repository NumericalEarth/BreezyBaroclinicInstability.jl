#####
##### Initial Condition Functions and GPU Kernels
#####
# IC functions accept (λ_deg, φ_deg, z) — LatitudeLongitudeGrid coordinates in degrees.

"""
Actual potential temperature:  θ = T · (p_ref / p)^κ
where T = Tv / (1 + ε·q) is the actual (not virtual) temperature.
"""
function initial_theta(λ_deg, φ_deg, z)
    φ  = deg2rad(φ_deg)
    Tv = virtual_temperature(φ, z)
    p  = balanced_pressure(φ, z)
    q  = moisture_profile(φ, z)
    T  = Tv / (1 + ε_virtual * q)
    return T * (p_ref / p)^κ_exponent
end

"""Density  ρ = p / (Rd · Tv)  — uses virtual temperature (moist ideal-gas law)."""
function initial_density(λ_deg, φ_deg, z)
    φ = deg2rad(φ_deg)
    return balanced_pressure(φ, z) / (Rd_dry * virtual_temperature(φ, z))
end

"""Zonal wind: gradient-wind balance + exponential perturbation."""
function initial_zonal_wind(λ_deg, φ_deg, z)
    λ = deg2rad(λ_deg)
    φ = deg2rad(φ_deg)
    return balanced_zonal_wind(φ, z) + wind_perturbation(λ, φ, z)
end

"""Specific humidity from the DCMIP-2016 moist profile (Eq. 18)."""
initial_moisture(λ_deg, φ_deg, z) = moisture_profile(deg2rad(φ_deg), z)

"""Reference potential temperature at the equator for base-state subtraction."""
theta_reference(z) = initial_theta(0.0, 0.0, z)

"""
    surface_temperature(λ_deg, φ_deg)

Prescribed SST from the DCMIP-2016 balanced state evaluated at z = 0.
Returns virtual temperature at the surface, which for the bulk flux
formulas serves as the ocean skin temperature driving sensible/latent
heat exchange.
"""
surface_temperature(λ_deg, φ_deg) = virtual_temperature(deg2rad(φ_deg), 0.0)

# ═══════════════════════════════════════════════════════════════════════════
# GPU kernels for setting analytic initial conditions
# ═══════════════════════════════════════════════════════════════════════════

@kernel function _set_moist_baroclinic_wave_kernel!(θ_field, ρ_field, qv_field, grid)
    i, j, k = @index(Global, NTuple)
    λ_deg = λnode(i, j, k, grid, Center(), Center(), Center())
    φ_deg = φnode(i, j, k, grid, Center(), Center(), Center())
    z     = znode(i, j, k, grid, Center(), Center(), Center())
    @inbounds begin
        θ_field[i, j, k] = initial_theta(λ_deg, φ_deg, z)
        ρ_field[i, j, k] = initial_density(λ_deg, φ_deg, z)
        qv_field[i, j, k] = initial_moisture(λ_deg, φ_deg, z)
    end
end

@kernel function _set_zonal_wind_kernel!(u_field, grid)
    i, j, k = @index(Global, NTuple)
    λ_deg = λnode(i, j, k, grid, Face(), Center(), Center())
    φ_deg = φnode(i, j, k, grid, Face(), Center(), Center())
    z     = znode(i, j, k, grid, Face(), Center(), Center())
    @inbounds u_field[i, j, k] = initial_zonal_wind(λ_deg, φ_deg, z)
end

"""
    set_analytic_ic!(model)

Set the DCMIP-2016 moist baroclinic wave analytic initial conditions on `model`.
Computes θ, ρ, u, qᵛ from the balanced state + perturbation, then forms
the conserved densities ρθ, ρu, ρqᵛ.
"""
function set_analytic_ic!(model)
    grid = model.grid
    arch = grid.architecture

    ρ  = dynamics_density(model.dynamics)
    θ  = model.formulation.potential_temperature
    qv = specific_prognostic_moisture(model)
    u  = model.velocities.u

    Oceananigans.Utils.launch!(arch, grid, :xyz,
        _set_moist_baroclinic_wave_kernel!, θ, ρ, qv, grid)

    Oceananigans.Utils.launch!(arch, grid, :xyz,
        _set_zonal_wind_kernel!, u, grid)

    ρθ = model.formulation.potential_temperature_density
    parent(ρθ) .= parent(ρ) .* parent(θ)

    ρu = model.momentum.ρu
    parent(ρu) .= parent(ρ) .* parent(u)

    ρqv = model.moisture_density
    parent(ρqv) .= parent(ρ) .* parent(qv)

    return nothing
end
