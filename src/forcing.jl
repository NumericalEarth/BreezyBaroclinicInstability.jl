#####
##### Relaxation-to-IC and Damp-to-Zero Forcing
#####
# GPU-compatible forcing structs using {Name} type parameter to read
# fields[Name] in tendency kernels.

# ═══════════════════════════════════════════════════════════════════════════
# RelaxToICForcing: damps prognostic fields toward an IC snapshot
#   F(i,j,k) = -α(t) * (current[i,j,k] - ic[i,j,k])
#   α(t) = max(0, α₀ * (1 - t/T_decay))
# ═══════════════════════════════════════════════════════════════════════════

struct RelaxToICForcing{Name, F, T}
    ic       :: F
    α0       :: T
    T_decay  :: T
end

RelaxToICForcing(name::Symbol, ic, α0::Real, T_decay::Real) =
    let Tpromoted = promote_type(typeof(α0), typeof(T_decay))
        RelaxToICForcing{name, typeof(ic), Tpromoted}(ic, Tpromoted(α0), Tpromoted(T_decay))
    end

@inline function (f::RelaxToICForcing{Name})(i, j, k, grid, clock, fields) where {Name}
    t = clock.time
    α = max(zero(t), f.α0 * (1 - t / f.T_decay))
    @inbounds ic_val  = f.ic[i, j, k]
    @inbounds cur_val = fields[Name][i, j, k]
    return -α * (cur_val - ic_val)
end

# When the kernel is adapted for the GPU, strip the Field stored in `ic`
# down to its underlying OffsetArray-of-CuDeviceArray so the whole forcing
# struct is isbits-compatible. Without this, the Field's grid carries the
# (non-isbits) NCCL communicator and CUDA stream handles into the kernel.
function Adapt.adapt_structure(to, f::RelaxToICForcing{Name}) where {Name}
    ic_a = Adapt.adapt(to, f.ic)
    return RelaxToICForcing{Name, typeof(ic_a), typeof(f.α0)}(ic_a, f.α0, f.T_decay)
end

# ═══════════════════════════════════════════════════════════════════════════
# DampToZeroForcing: damps fields toward zero
#   F(i,j,k) = -α(t) * fields[Name][i,j,k]
#   α(t) = max(0, α₀ * (1 - t/T_decay))
#
# Use on cloud condensate (ρqᶜˡ, ρqᶜⁱ) to suppress fast microphysics ↔
# dynamics feedback during spinup from interpolated IC.
# ═══════════════════════════════════════════════════════════════════════════

struct DampToZeroForcing{Name, T}
    α0      :: T
    T_decay :: T
end

DampToZeroForcing(name::Symbol, α0::Real, T_decay::Real) =
    let Tp = promote_type(typeof(α0), typeof(T_decay))
        DampToZeroForcing{name, Tp}(Tp(α0), Tp(T_decay))
    end

@inline function (f::DampToZeroForcing{Name})(i, j, k, grid, clock, fields) where {Name}
    t = clock.time
    α = max(zero(t), f.α0 * (1 - t / f.T_decay))
    @inbounds cur_val = fields[Name][i, j, k]
    return -α * cur_val
end

# ═══════════════════════════════════════════════════════════════════════════
# Builder functions
# ═══════════════════════════════════════════════════════════════════════════

"""
    build_ic_relaxation_forcing(grid; α0, T_decay)

Allocate IC-snapshot fields and build a NamedTuple of `RelaxToICForcing`s.
Returns `(forcing, snapshots)`. The caller must fill snapshots after loading ICs
by calling `copy_ic_snapshots!(snapshots, model)`.
"""
function build_ic_relaxation_forcing(grid; α0, T_decay)
    snapshots = (
        ρu   = XFaceField(grid),
        ρv   = YFaceField(grid),
        ρw   = ZFaceField(grid),
        ρθ   = CenterField(grid),
        ρqᵛ  = CenterField(grid),
    )
    forcing = (
        ρu  = RelaxToICForcing(:ρu,  snapshots.ρu,  α0, T_decay),
        ρv  = RelaxToICForcing(:ρv,  snapshots.ρv,  α0, T_decay),
        ρw  = RelaxToICForcing(:ρw,  snapshots.ρw,  α0, T_decay),
        ρθ  = RelaxToICForcing(:ρθ,  snapshots.ρθ,  α0, T_decay),
        ρqᵛ = RelaxToICForcing(:ρqᵛ, snapshots.ρqᵛ, α0, T_decay),
    )
    return forcing, snapshots
end

"""
    build_cloud_damping_forcing(; α0, T_decay)

Build damp-to-zero forcing for cloud condensate fields (ρqᶜˡ, ρqᶜⁱ).
"""
function build_cloud_damping_forcing(; α0, T_decay)
    return (
        ρqᶜˡ = DampToZeroForcing(:ρqᶜˡ, α0, T_decay),
        ρqᶜⁱ = DampToZeroForcing(:ρqᶜⁱ, α0, T_decay),
    )
end

"""
    copy_ic_snapshots!(snapshots, model)

Copy the current prognostic field values from `model` into the matching
`snapshots` fields, then fill halos. Must be called AFTER the IC has been
loaded (and interpolated) into the model's prognostic fields.

CRITICAL: Snapshot fields are freshly allocated, so their halo regions
contain uninitialized GPU memory (often NaN). Without filling halos,
NaN halos contaminate the momentum/thermo tendencies at iter 2.
"""
function copy_ic_snapshots!(snapshots::NamedTuple, model)
    Oceananigans.set!(snapshots.ρu,  model.momentum.ρu)
    Oceananigans.set!(snapshots.ρv,  model.momentum.ρv)
    Oceananigans.set!(snapshots.ρw,  model.momentum.ρw)
    Oceananigans.set!(snapshots.ρθ,  model.formulation.potential_temperature_density)
    Oceananigans.set!(snapshots.ρqᵛ, model.moisture_density)

    Oceananigans.BoundaryConditions.fill_halo_regions!(snapshots.ρu)
    Oceananigans.BoundaryConditions.fill_halo_regions!(snapshots.ρv)
    Oceananigans.BoundaryConditions.fill_halo_regions!(snapshots.ρw)
    Oceananigans.BoundaryConditions.fill_halo_regions!(snapshots.ρθ)
    Oceananigans.BoundaryConditions.fill_halo_regions!(snapshots.ρqᵛ)

    return nothing
end
