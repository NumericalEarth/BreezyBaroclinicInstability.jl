#####
##### Initial Condition Loading from JLD2 Checkpoints
#####

"""
    load_ic_interpolated!(model, path::String;
                          H = 30e3,
                          source_latitude = (-75, 75),
                          source_longitude = (0, 360),
                          clamp_moisture = false)

Load the prognostic state from a JLD2 checkpoint and interpolate onto
the model fields using `Oceananigans.Fields.interpolate!`.

For each field:
1. Read from JLD2
2. Build a source grid on GPU at source resolution
3. Create source Field, copy data, fill halos
4. Call `interpolate!(target_field, src_field)`
5. Fill target halos after interpolation

Works identically for single-GPU (`GPU()`) and multi-GPU (`NCCLDistributed`)
architectures — `interpolate!` handles distributed target grids automatically.

Handles all field types: ρ, ρu, ρv, ρw, ρθ, ρqᵛ, and optionally
ρqᶜˡ, ρqᶜⁱ, ρqʳ, ρqˢ. Checks both `"micro_ρqᶜˡ"` and `"ρqᶜˡ"` naming.
"""
function load_ic_interpolated!(model, path::String;
                               H = nothing,
                               source_latitude = (-75, 75),
                               source_longitude = (0, 360),
                               source_z_stretching = 3.0,
                               clamp_moisture = false)
    Nλ_src, Nφ_src, Nz_src, H_saved, ρ_data, ρu_data, ρv_data, ρw_data, ρθ_data, ρqv_data =
        JLD2.jldopen(path, "r") do file
            H_src = haskey(file, "H") ? Float64(file["H"]) : 30e3  # legacy checkpoints
            (file["Nλ"], file["Nφ"], file["Nz"], H_src,
             file["ρ"], file["ρu"], file["ρv"], file["ρw"],
             file["ρθ"], file["ρqᵛ"])
        end
    H_src = isnothing(H) ? H_saved : Float64(H)

    ρqcl_data, ρqci_data, ρqr_data, ρqs_data = JLD2.jldopen(path, "r") do file
        ρqcl = haskey(file, "micro_ρqᶜˡ") ? file["micro_ρqᶜˡ"] :
               haskey(file, "ρqᶜˡ") ? file["ρqᶜˡ"] : nothing
        ρqci = haskey(file, "micro_ρqᶜⁱ") ? file["micro_ρqᶜⁱ"] :
               haskey(file, "ρqᶜⁱ") ? file["ρqᶜⁱ"] : nothing
        ρqr  = haskey(file, "ρqʳ") ? file["ρqʳ"] : nothing
        ρqs  = haskey(file, "ρqˢ") ? file["ρqˢ"] : nothing
        (ρqcl, ρqci, ρqr, ρqs)
    end

    grid = model.grid
    FT   = eltype(grid)

    pairs = Any[
        (ρ_data,   dynamics_density(model.dynamics)),
        (ρu_data,  model.momentum.ρu),
        (ρv_data,  model.momentum.ρv),
        (ρw_data,  model.momentum.ρw),
        (ρθ_data,  model.formulation.potential_temperature_density),
        (ρqv_data, model.moisture_density),
    ]

    for (data, name) in [(ρqcl_data, :ρqᶜˡ), (ρqci_data, :ρqᶜⁱ),
                          (ρqr_data, :ρqʳ), (ρqs_data, :ρqˢ)]
        if data !== nothing && haskey(model.microphysical_fields, name)
            push!(pairs, (data, model.microphysical_fields[name]))
            @info "Loading $name from IC file" extrema=extrema(data)
        end
    end

    halo = Oceananigans.halo_size(grid)

    src_z = if source_z_stretching == 0
        (0, H_src)
    else
        k -> H_src * tanh(source_z_stretching * (k - 1) / Nz_src) / tanh(source_z_stretching)
    end

    src_grid = LatitudeLongitudeGrid(GPU();
        size = (Nλ_src, Nφ_src, Nz_src),
        halo = halo,
        latitude  = source_latitude,
        longitude = source_longitude,
        z = src_z)

    for (src_array, target_field) in pairs
        loc = Oceananigans.location(target_field)
        iloc = map(L -> L(), loc)
        src_field = Field(iloc, src_grid)
        gpu_data = Oceananigans.on_architecture(GPU(), Array{FT}(src_array))
        copyto!(Oceananigans.interior(src_field), gpu_data)
        Oceananigans.BoundaryConditions.fill_halo_regions!(src_field)

        @info "interpolate!" field=nameof(typeof(target_field)) loc src=size(Oceananigans.interior(src_field)) dst=size(Oceananigans.interior(target_field))
        Oceananigans.Fields.interpolate!(target_field, src_field)
        Oceananigans.BoundaryConditions.fill_halo_regions!(target_field)
    end

    if clamp_moisture
        _clamp_moisture_fields!(model, FT)
    end

    return nothing
end

function _clamp_moisture_fields!(model, FT)
    function clamp_nonneg!(f, name)
        int = Oceananigans.interior(f)
        int .= max.(int, zero(FT))
        Oceananigans.BoundaryConditions.fill_halo_regions!(f)
        @info "clamped $name to ≥ 0"
    end
    clamp_nonneg!(model.moisture_density, "ρqᵛ")
    for name in (:ρqᶜˡ, :ρqᶜⁱ, :ρqʳ, :ρqˢ)
        if haskey(model.microphysical_fields, name)
            clamp_nonneg!(model.microphysical_fields[name], string(name))
        end
    end
end

"""
    load_ic_direct!(model, path::String; Rx::Int, Ry::Int, rank::Int)

Direct tile copy from an assembled checkpoint — no interpolation.
Source grid resolution must match the distributed target grid exactly.
Each rank reads only its slice from the file (no full-global GPU allocation).

`Rx`, `Ry` are the partition dimensions. `rank` is the MPI rank.
Rank-to-tile mapping: `ix = rank ÷ Ry`, `iy = rank % Ry`.
"""
function load_ic_direct!(model, path::String; Rx::Int, Ry::Int, rank::Int)
    ix = rank ÷ Ry
    iy = rank % Ry

    Nλ, Nφ, Nz = JLD2.jldopen(path, "r") do f
        Int(f["Nλ"]), Int(f["Nφ"]), Int(f["Nz"])
    end

    Nx_per = Nλ ÷ Rx
    Ny_per = Nφ ÷ Ry
    x_range = (ix * Nx_per + 1):((ix + 1) * Nx_per)
    y_range_c = (iy * Ny_per + 1):((iy + 1) * Ny_per)
    # ρv: top-y rank includes the wall face (Ny_per + 1 rows)
    y_range_v = (iy == Ry - 1) ? ((iy * Ny_per + 1):((iy + 1) * Ny_per + 1)) : y_range_c

    FT = eltype(model.grid)

    function load_one!(target, key, yr; z_range=Colon())
        tile = JLD2.jldopen(path, "r") do f
            Array{FT}(f[key][x_range, yr, z_range])
        end
        gpu_tile = Oceananigans.on_architecture(GPU(), tile)
        copyto!(Oceananigans.interior(target), gpu_tile)
        Oceananigans.BoundaryConditions.fill_halo_regions!(target)
        rank == 0 && @info "loaded" key=key tile=size(tile)
    end

    load_one!(dynamics_density(model.dynamics), "ρ", y_range_c)
    load_one!(model.momentum.ρu, "ρu", y_range_c)
    load_one!(model.momentum.ρv, "ρv", y_range_v)
    load_one!(model.momentum.ρw, "ρw", y_range_c)
    load_one!(model.formulation.potential_temperature_density, "ρθ", y_range_c)
    load_one!(model.moisture_density, "ρqᵛ", y_range_c)

    for (file_key, field_name) in [("micro_ρqᶜˡ", :ρqᶜˡ), ("micro_ρqᶜⁱ", :ρqᶜⁱ),
                                    ("ρqʳ", :ρqʳ), ("ρqˢ", :ρqˢ)]
        haskey(model.microphysical_fields, field_name) || continue
        JLD2.jldopen(path, "r") do f
            haskey(f, file_key) || return
        end
        load_one!(model.microphysical_fields[field_name], file_key, y_range_c)
    end

    rank == 0 && @info "IC loaded (direct-copy, no interpolation)"
    return nothing
end
