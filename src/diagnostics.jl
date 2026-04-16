#####
##### Diagnostic Utilities
#####

"""
    any_nan(model) → Bool

Check the 6 core prognostic fields for NaN values. Returns `true` if any NaN found.
"""
function any_nan(model)
    fields = (dynamics_density(model.dynamics),
              model.momentum.ρu, model.momentum.ρv, model.momentum.ρw,
              model.formulation.potential_temperature_density,
              model.moisture_density)
    for f in fields
        any(isnan, parent(f)) && return true
    end
    return false
end

"""
    field_extrema(f) → (Float64, Float64)

Return `(min, max)` of the interior of field `f` as Float64.
"""
function field_extrema(f)
    p = Oceananigans.interior(f)
    return Float64(minimum(p)), Float64(maximum(p))
end

"""
    report_state(model; rank=0, label="")

Print the extrema of all 6 core prognostic fields. Useful for diagnostics
after IC loading or time stepping.
"""
function report_state(model; rank=0, label="")
    prefix = isempty(label) ? "" : "$label "
    for (name, f) in [("ρ",   dynamics_density(model.dynamics)),
                       ("ρu",  model.momentum.ρu),
                       ("ρv",  model.momentum.ρv),
                       ("ρw",  model.momentum.ρw),
                       ("ρθ",  model.formulation.potential_temperature_density),
                       ("ρqᵛ", model.moisture_density)]
        mn, mx = field_extrema(f)
        fin = isfinite(mn) && isfinite(mx)
        @info @sprintf("[r%d] %s%s: [%.4e, %.4e] finite=%s", rank, prefix, name, mn, mx, fin)
    end
    flush(stderr); flush(stdout)
end

"""
    check_density_positivity(model; rank=0)

Check that density (ρ) is strictly positive in the interior. Warns if any cells ≤ 0.
"""
function check_density_positivity(model; rank=0)
    ρ_int = Array(Oceananigans.interior(dynamics_density(model.dynamics)))
    nzeros = count(x -> x <= 0, ρ_int)
    ρ_min = minimum(ρ_int)
    if nzeros > 0 || ρ_min <= 0
        @warn @sprintf("[r%d] ρ interior has %d cells ≤ 0 (min=%.4e)", rank, nzeros, ρ_min)
    else
        @info @sprintf("[r%d] ρ interior OK: min=%.4e", rank, ρ_min)
    end
    return nzeros == 0
end
