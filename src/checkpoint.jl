#####
##### Checkpoint Save
#####

"""
    save_checkpoint(model, filepath; Δt=nothing)

Save all prognostic and microphysical fields to a JLD2 file with metadata.
Each rank saves its own local portion of the fields (interior only).

The saved format is compatible with `load_ic_interpolated!` and `load_ic_direct!`.
"""
function save_checkpoint(model, filepath; Δt=nothing)
    mkpath(dirname(filepath))
    m = model
    grid = m.grid
    Nλ, Nφ, Nz = size(grid)

    JLD2.jldopen(filepath, "w") do file
        file["Nλ"] = Nλ
        file["Nφ"] = Nφ
        file["Nz"] = Nz
        file["iteration"] = m.clock.iteration
        file["time"] = m.clock.time
        if Δt !== nothing
            file["Δt"] = Δt
        end
        file["ρ"]   = Array(Oceananigans.interior(dynamics_density(m.dynamics)))
        file["ρu"]  = Array(Oceananigans.interior(m.momentum.ρu))
        file["ρv"]  = Array(Oceananigans.interior(m.momentum.ρv))
        file["ρw"]  = Array(Oceananigans.interior(m.momentum.ρw))
        file["ρθ"]  = Array(Oceananigans.interior(m.formulation.potential_temperature_density))
        file["ρqᵛ"] = Array(Oceananigans.interior(m.moisture_density))
        for name in keys(m.microphysical_fields)
            file[string(name)] = Array(Oceananigans.interior(m.microphysical_fields[name]))
        end
    end

    @info "Checkpoint saved" filepath
    return filepath
end
