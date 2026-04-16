#####
##### DCMIP-2016 Balanced State Functions
#####
# All angles in radians unless noted otherwise.

"""
    vertical_structure(z) → (; τ₁, τ₂, I₁, I₂)

Vertical profiles and their integrals that define the balanced state (Eqs. 5–8).
"""
function vertical_structure(z)
    ζ      = z / (vert_width * scale_height)
    exp_ζ² = exp(-ζ^2)

    τ₁ = coeff_A * lapse_rate / T_mean * exp(lapse_rate * z / T_mean) +
         coeff_B * (1 - 2 * ζ^2) * exp_ζ²

    τ₂ = coeff_C * (1 - 2 * ζ^2) * exp_ζ²

    I₁ = coeff_A * (exp(lapse_rate * z / T_mean) - 1) +
         coeff_B * z * exp_ζ²

    I₂ = coeff_C * z * exp_ζ²

    return (; τ₁, τ₂, I₁, I₂)
end

"""Horizontal temperature structure:  cos(φ)^K − K/(K+2) cos(φ)^(K+2)."""
F_temperature(cosφ) = cosφ^jet_width - jet_width / (jet_width + 2) * cosφ^(jet_width + 2)

"""Horizontal wind structure:  cos(φ)^(K−1) − cos(φ)^(K+1)."""
F_wind(cosφ) = cosφ^(jet_width - 1) - cosφ^(jet_width + 1)

"""
    virtual_temperature(φ, z)

Virtual temperature Tv from the balanced state (Eq. 9, shallow atmosphere).
In the dry case Tv = T; in the moist case Tv > T.
"""
function virtual_temperature(φ, z)
    vs = vertical_structure(z)
    return 1.0 / (vs.τ₁ - vs.τ₂ * F_temperature(cos(φ)))
end

"""
    balanced_pressure(φ, z)

Hydrostatic pressure from the balanced state (Eq. 10).
"""
function balanced_pressure(φ, z)
    vs = vertical_structure(z)
    return p_ref * exp(-gravity / Rd_dry * (vs.I₁ - vs.I₂ * F_temperature(cos(φ))))
end

"""
    moisture_profile(φ, z)

DCMIP-2016 specific humidity (Eq. 18).
Below the tropopause (η > 0.1): concentrated at midlatitudes in the lower troposphere,
peaking around 85 % relative humidity.  Above: q ≈ 10⁻¹² kg/kg.
"""
function moisture_profile(φ, z)
    p = balanced_pressure(φ, z)
    η = p / p_ref
    q_below = q0_surface * exp(-(φ / φ_width)^4) *
                           exp(-((η - 1) * p_ref / p_width)^2)
    return ifelse(η > η_tropopause, q_below, q_tropopause)
end

"""
    balanced_zonal_wind(φ, z)

Gradient-wind–balanced zonal wind (Eq. 12, shallow atmosphere).
"""
function balanced_zonal_wind(φ, z)
    vs   = vertical_structure(z)
    cosφ = cos(φ)
    Tv   = 1.0 / (vs.τ₁ - vs.τ₂ * F_temperature(cosφ))

    U_term   = gravity / earth_radius * jet_width * vs.I₂ * F_wind(cosφ) * Tv
    r_cosφ   = earth_radius * cosφ
    Ω_r_cosφ = earth_rotation * r_cosφ

    return -Ω_r_cosφ + sqrt(Ω_r_cosφ^2 + r_cosφ * U_term)
end

"""
    wind_perturbation(λ, φ, z)

Exponential perturbation to the zonal wind (Eq. 14).
Gaussian in great-circle distance from (λ_center, φ_center), tapered above z_perturb.
"""
function wind_perturbation(λ, φ, z)
    sin_dφ = sin((φ - φ_center) / 2)
    sin_dλ = sin((λ - λ_center) / 2)
    h = sin_dφ * sin_dφ + cos(φ) * cos(φ_center) * sin_dλ * sin_dλ
    gc_sq = 4.0 * h / (r_perturb * r_perturb)

    ẑ     = z / z_perturb
    taper = ifelse(z < z_perturb, 1 - 3 * ẑ^2 + 2 * ẑ^3, 0.0)

    return ifelse(gc_sq < 1.0, u_perturb * taper * exp(-gc_sq), 0.0)
end
