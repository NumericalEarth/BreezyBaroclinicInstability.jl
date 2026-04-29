# Substepper validation test plan

A concrete, implementable suite of 14 tests for the split-explicit acoustic
substepper. Each test is fully specified — grid, IC, reference state,
integration parameters, and a quantitative pass criterion. All tests use
defaults `forward_weight = 0.55` and `damping = NoDivergenceDamping()`
(no sweeps over those).

## Conventions used in every test

```julia
using Breeze, Oceananigans, CUDA
Oceananigans.defaults.FloatType = Float64
Oceananigans.defaults.gravitational_acceleration = 9.80665
g  = 9.80665
Rᵈ = 287.0
cᵖᵈ = 1005.0
constants = ThermodynamicConstants(eltype(grid))
```

Δt and Δz comparisons use `cs = sqrt(γᵈ Rᵈ × 300) ≈ 348 m/s` as the reference
sound speed. ε ≡ `eps(Float64)` ≈ 2.2e-16. All grids use `halo = (5, 5, 5)`
unless stated; reduce to `(5, 5)` on 2-D Flat-y. Default substepper
`SplitExplicitTimeDiscretization()` (adaptive N, ω=0.55, NoDamping,
ProportionalSubsteps).

A single helper:

```julia
function track_drift!(sim; field = :w, period = 10)
    extrema_history = Float64[]
    cb = sim -> push!(extrema_history,
                      maximum(abs, interior(getproperty(sim.model.velocities, field))))
    add_callback!(sim, cb, IterationInterval(period))
    run!(sim)
    return extrema_history
end
```

---

# Tier 1 — basis tests (per-PR CI; <10 min total on one GPU)

## T1: Reference-state discrete hydrostatic balance

**Failure modes:** D (reference state not in discrete hydrostatic balance).

**Setup**
```julia
grid = RectilinearGrid(GPU(); size = (16, 16, 64), halo = (5, 5, 5),
                       x = (0, 100e3), y = (0, 100e3), z = (0, 30e3),
                       topology = (Periodic, Periodic, Bounded))
T₀ = 250.0
θᵇᵍ(z) = T₀ * exp(g * z / (cᵖᵈ * T₀))
dyn = CompressibleDynamics(SplitExplicitTimeDiscretization();
                           reference_potential_temperature = θᵇᵍ)
model = AtmosphereModel(grid; dynamics = dyn,
                        thermodynamic_constants = constants,
                        timestepper = :AcousticRungeKutta3)
ref = model.dynamics.reference_state
```

**Procedure** — compute the discrete hydrostatic residual at every interior
z-face using the *exact* operators the substepper uses (Δzᶜᶜᶠ, simple
average for ρ-to-face):

```julia
using Oceananigans.Operators: Δzᶜᶜᶠ
ε = zeros(grid.Nx, grid.Ny, grid.Nz - 1)
for i in 1:grid.Nx, j in 1:grid.Ny, k in 2:grid.Nz
    Δz_face = Δzᶜᶜᶠ(i, j, k, grid)
    ∂z_p   = (ref.pressure[i,j,k] - ref.pressure[i,j,k-1]) / Δz_face
    gρ_face = g * (ref.density[i,j,k] + ref.density[i,j,k-1]) / 2
    ε[i,j,k-1] = ∂z_p + gρ_face
end
```

**Pass:** `maximum(abs, ε) ≤ 1e-9` (a few hundred ulp of `g × ρ_max ≈ 12 N/m³`).

**Why this number:** the reference state must produce a slow vertical-momentum
tendency that's bit-quiet at rest. A residual of 3e-11 Pa over Δz=470 m
becomes a ρw force of 3e-11/470 ≈ 6e-14 N/m³ — exactly the seed we observed.
Anything ≥ 1e-9 is a real bookkeeping bug.

---

## T2: Pressure consistency between EoS path and reference path

**Failure modes:** D (the EoS-pressure-vs-reference-pressure seed).

**Setup** — same `model` as T1.

**Procedure**
```julia
set!(model; θ = θᵇᵍ, ρ = ref.density)
Oceananigans.TimeSteppers.update_state!(model)
Δp = maximum(abs, interior(model.dynamics.pressure) .- interior(ref.pressure))
Δρ = maximum(abs, interior(BreezeBC.dynamics_density(model.dynamics)) .- interior(ref.density))
```

**Pass:** `Δp ≤ 100 * eps(Float64) * maximum(abs, interior(ref.pressure))`
(i.e., ≲ 2e-11 Pa on a ~1e5 Pa scale). `Δρ == 0` exactly.

**Why this number:** if `update_state!` and the reference-state constructor
agree, both produce p from the same equations of state at the same (ρ, ρθ),
so they must match within rounding. A sustained ~3e-11 Pa difference
indicates the two paths are *not* using bit-identical formulas (e.g.
trapezoidal hydrostatic integration vs direct EoS), and that residual will
seed the substepper's ρw at every outer step.

---

## T3: Slow vertical-momentum tendency at rest (3-D Cartesian)

**Failure modes:** B, D (substep loop must keep rest = ref state at machine
zero).

**Setup** — same `model` as T1.

**Procedure**
```julia
set!(model; θ = θᵇᵍ, ρ = ref.density)
sub = model.timestepper.substepper
Breeze.CompressibleEquations.freeze_outer_step_state!(sub, model)
Breeze.TimeSteppers.compute_slow_momentum_tendencies!(model)
Breeze.TimeSteppers.compute_slow_scalar_tendencies!(model)
Breeze.CompressibleEquations.assemble_slow_vertical_momentum_tendency!(sub, model)

max_slow_ρw = maximum(abs, interior(sub.slow_vertical_momentum_tendency))
```

**Pass:** `max_slow_ρw ≤ 1e-12` N/m³.

**Why this number:** at U⁰ = U_ref, every term in
`Gˢρw = Gⁿρw - ∂z(p⁰-p_ref) - g(ρ⁰-ρ_ref)` should vanish to within rounding.
A non-zero value here is the bug seed before any substep loop runs.

---

## T4: Rest-atmosphere drift, Δt sweep, isothermal reference

**Failure modes:** B (unconditional stability proof), D, H.

**Setup** — same grid as T1.

**Procedure** — run a rest atmosphere for many outer steps at five values of
Δt, recording max|w| every 5 outer steps:

```julia
results = Dict()
for Δt in (0.5, 2.0, 5.0, 10.0, 20.0)
    model = build_t1_model()
    set!(model; θ = θᵇᵍ, ρ = model.dynamics.reference_state.density)
    sim = Simulation(model; Δt, stop_iteration = 1000)
    drift = track_drift!(sim; period = 5)
    results[Δt] = (final_max_w = last(drift), envelope = maximum(drift))
end
```

**Pass for every Δt:** `envelope ≤ 1e-10` m/s.

**Why this number:** rest atmosphere has no physical signal. With Float64,
machine-zero drift settles around `eps(Float64) × cs ≈ 1e-13 m/s`. A drift
of 1e-10 already represents 1000× amplification — a flag that an unstable
mode exists, even if it hasn't saturated. If even one Δt fails the bound,
the substep loop has a Δt-dependent unstable mode (this is the class of bug
we currently have at Δt=20s).

---

## T5: Rest-atmosphere drift on LatitudeLongitudeGrid

**Failure modes:** B, D, F (curvilinear metric in horizontal operators).

**Setup**
```julia
grid = LatitudeLongitudeGrid(GPU(); size = (180, 90, 64), halo = (5, 5, 5),
                             longitude = (0, 360), latitude = (-80, 80),
                             z = (0, 30e3))
# Same θᵇᵍ, dynamics, model construction as T1
```

**Procedure** — same as T4 but only at `Δt = 20.0` for 1000 outer steps.

**Pass:** `envelope ≤ 1e-10` m/s.

**Why this number:** if T4 passes (Cartesian rest atmosphere is bit-quiet)
but T5 fails, the bug is in the lat-lon metric handling of one of the
substepper operators. If T4 also fails at Δt=20s, T5 result is redundant —
fix T4 first.

---

## T6: Bounded-y vs Periodic-y rest equivalence

**Failure modes:** E (boundary condition / halo handling).

**Setup** — two RectilinearGrids identical except topology:
```julia
grid_PP = RectilinearGrid(GPU(); size = (32, 32, 64), halo = (5, 5, 5),
                          x = (0, 100e3), y = (0, 100e3), z = (0, 10e3),
                          topology = (Periodic, Periodic, Bounded))
grid_PB = RectilinearGrid(GPU(); size = (32, 32, 64), halo = (5, 5, 5),
                          x = (0, 100e3), y = (-50e3, 50e3), z = (0, 10e3),
                          topology = (Periodic, Bounded, Bounded))
```

**Procedure** — build identical models on both grids, set rest atmosphere,
step Δt=2s for 200 outer steps. Record max|w| trajectories.

**Pass:** at every recorded step, `|max|w|_PP - max|w|_PB| ≤ 1e-12` m/s.

**Why this number:** for an exactly horizontally-uniform IC, the boundary
condition shouldn't matter — Periodic and Bounded both produce 0 horizontal
gradient. Any divergence between the two trajectories indicates a
boundary-handling bug in the substepper's halo fills or boundary masks.

---

## T7: Substep-count independence at fixed Δt

**Failure modes:** H (adaptive-N consistency).

**Setup** — T1 grid, isothermal reference.

**Procedure** — fix Δt=20s. Build four models with explicit
`SplitExplicitTimeDiscretization(substeps = N)` for `N ∈ (6, 12, 24, 48)`.
Set the same rest IC, step 100 outer steps each, record final max|w|.

**Pass:** if T4 passes (rest atmosphere is bit-quiet), this should give
`max|w| ≤ 1e-10` for every N. Additionally, `range / mean ≤ 0.1` across the
four N values (results should be N-independent past a coarseness threshold).

**Why this number:** the substepper is supposed to converge in N: doubling
N halves the truncation error. If the rest state amplifies *worse* at large
N, there's a per-substep accumulation bug.

---

## T8: WS-RK3 stage linearization at small Δt matches analytic

**Failure modes:** A (operator correctness), C (slow + fast = total), J (RK3
stage consistency).

**Setup** — T1 grid, isothermal reference, plus a known small perturbation:

```julia
σ_pert(x, y, z) = 1e-3 * exp(-((z - 5e3)/2e3)^2)  # tiny density bump
set!(model; θ = θᵇᵍ, ρ = ref.density + σ_pert)
```

**Procedure** — take a *single* WS-RK3 outer step at `Δt = 0.01s` (so the
linearization is essentially exact). Record (ρ, ρθ, μu, μv, μw)_after.
Independently compute the analytic linearized RHS at U⁰ and the expected
state `U⁰ + Δt × R(U⁰)`.

**Pass:** L2 difference between simulated and analytic ≤ `Δt²` * (typical
state magnitude). On Δt=0.01s with state ~ 1, that's ≤ 1e-4 — i.e. within
the linearization-truncation budget.

**Why this number:** at infinitesimal Δt the full nonlinear scheme reduces
to its linearized version, which has an explicit closed form. Failing this
test means an operator is plain wrong (not just unstable).

---

## T9: Acoustic pulse propagation in 3-D Cartesian

**Failure modes:** A (vertical / horizontal acoustic operators).

**Setup**
```julia
grid = RectilinearGrid(GPU(); size = (64, 64, 64), halo = (5, 5, 5),
                       x = (-10e3, 10e3), y = (-10e3, 10e3), z = (0, 20e3),
                       topology = (Periodic, Periodic, Bounded))
T₀ = 250.0; θᵇᵍ(z) = T₀ * exp(g * z / (cᵖᵈ * T₀))
# Pulse: small ρ perturbation centered at (0, 0, 10 km)
σ_pulse(x, y, z) = 1e-4 * exp(-((x^2 + y^2 + (z - 10e3)^2) / (1e3)^2))
set!(model; θ = θᵇᵍ, ρ = ref.density + σ_pulse)
```

**Procedure** — step at Δt = 0.5s (cs Δτ/Δz_min ≈ 0.27, comfortably stable)
for 50 outer steps. Track the radius of the pulse: time of arrival of the
pulse peak at a probe at (0, 0, 13 km) should be `3 km / cs ≈ 8.6 s`.

**Pass:** measured arrival time within ±5% of analytic `3 km / cs`.

**Why this number:** propagation speed errors larger than 5% indicate an
operator-coefficient bug in the acoustic core. (Stronger criterion: <1%
error compared to the 2-D acoustic-pulse reference test that the substepper
already passes.)

---

## T10: Mass and ρθ conservation under long substepping

**Failure modes:** G (hidden mass / energy loss).

**Setup** — T9 grid, T9 IC.

**Procedure** — step Δt=2s for 5000 outer steps (= 10000 sim seconds).
Track `M(t) = ∑_{i,j,k} ρ × V_cell` and `H(t) = ∑_{i,j,k} ρθ × V_cell`
every 100 outer steps.

**Pass:** `|M(t) - M(0)| / M(0) ≤ 1e-12` and same for H. (Periodic in x, y,
no fluxes through z-walls, no microphysics → both must be exactly conserved
to within rounding.)

**Why this number:** machine-precision mass conservation is a structural
property of finite-volume formulations. Any drift indicates a non-conservative
operator (e.g. divergence not in flux form, post-solve recovery losing mass).
1e-12 over 5000 steps allows ~2e-16 per-step rounding accumulation — that's
the right scale for Float64.

---

## T11: WENO-on-rest-state consistency

**Failure modes:** A (advection at rest), C.

**Setup** — T1 grid, isothermal rest atmosphere, plus a *constant* tracer
field added: `c(x, y, z) = 1.0 + 0.1 * exp(-((z-5e3)/1e3)^2)` (vertically
varying, horizontally constant).

**Procedure** — step Δt=20s for 500 outer steps. Track max horizontal
gradient of c at each step.

**Pass:** `max(|∂x c| + |∂y c|) ≤ 1e-12` at every step.

**Why this number:** WENO reconstruction of a horizontally-uniform field
should produce a horizontally-uniform result. If it doesn't, WENO's
nonlinear weights are introducing horizontal structure from rounding —
which then becomes a substep seed.

---

# Tier 2 — multi-feature integration tests (nightly; ~30 min on one GPU)

## T12: Inertia-gravity wave on the f-plane

**Failure modes:** Coupling between Coriolis (slow) and acoustic substep
(fast); J (RK3 with non-trivial linearization point).

**Setup**
```julia
f₀ = 1e-4
N₀² = 1e-4
grid = RectilinearGrid(GPU(); size = (128, 1, 64), halo = (5, 5, 5),
                       x = (-150e3, 150e3), y = (-1, 1), z = (0, 10e3),
                       topology = (Periodic, Flat, Bounded))
coriolis = ConstantCartesianCoriolis(f = f₀)
θᵇᵍ(z) = 300 * exp(N₀² * z / g)
# IC: Skamarock-Klemp 1994 IGW analytic perturbation
```

The Skamarock-Klemp 1994 inertia-gravity wave is a standard substepper
benchmark; analytic linear solution exists.

**Procedure** — step Δt=2s for 3000 outer steps (= 100 minutes sim). Compare
simulated `θ'` at each output time to the linear analytic solution.

**Pass:** L2 error in θ' < 5% relative to peak θ' amplitude over the run.

**Why:** standard reference; failure means Coriolis/buoyancy split is
inconsistent with the linearized substep.

---

## T13: Balanced jet drift (DCMIP IC, no perturbation)

**Failure modes:** F (curvilinear with flow), the production-realistic test
that catches what BCI runs would catch but at lower cost.

**Setup**
```julia
grid = LatitudeLongitudeGrid(GPU(); size = (180, 80, 32), halo = (5, 5, 5),
                             longitude = (0, 360), latitude = (-80, 80),
                             z = (0, 30e3))
# DCMIP-2016 balanced-jet IC, but DROP THE WIND PERTURBATION
# (set u_perturb = 0 in zonal_velocity).
# Isothermal reference state at T₀ = 250 K.
```

**Procedure** — step Δt=20s for 24 hours sim (= 4320 outer steps).

**Pass:** `max|w| < 1e-2 m/s` at every output. (The balanced jet has
no source of vertical motion; any growth larger than 1 cm/s indicates the
substepper is not preserving the analytical balanced state.)

**Why this number:** a balanced jet on a rotating sphere has u ~ 30 m/s
analytically and w ≡ 0. Discrete imbalance produces some w, but a healthy
substepper should keep |w| at the discretization-error level (~mm/s).
1 cm/s is a generous bound that any production-quality global model must
satisfy.

---

## T14: Anelastic-vs-substepper comparison on bubble (existing)

**Failure modes:** A, C (acoustic core correctness for the standard test).

This is the existing `validation/anelastic_compressible_comparison/` test.
Listed for completeness — it's already in place and is the strongest test
of the substepper's *acoustic core*. It does not catch bugs B, D, F, G that
T1-T13 above do; it should not be deleted, but it should not be the only
substepper validation either.

**Pass:** unchanged from current test.

---

# Test runner

Group all 14 tests into a single `test/substepper_validation/runtests.jl`
that returns a structured pass/fail summary. Tier 1 (T1-T11) runs on every
PR; Tier 2 (T12-T13) nightly; T14 is already in place.

Each test should fail-loud when its threshold is exceeded (informative log
line, not just a generic `@test`), so debugging the failing test points at
the failing failure-mode immediately:

```julia
@info "T4: rest-atmosphere drift Δt=$(Δt)s, envelope=$(envelope) m/s"
@assert envelope ≤ 1e-10  "T4 FAILED: rest atmosphere amplified at Δt=$(Δt)s. \
                            Indicates a Δt-dependent unstable mode in the substep \
                            loop (failure mode B). Check implicit-solve coefficients \
                            and predictor / post-solve weight matching."
```

# What this catches that the current suite doesn't

| Test  | Catches the bug we just found? |
|-------|-------------------------------|
| T1    | Yes — at construction time, before any time-stepping |
| T2    | Yes — flags the 3e-11 Pa pressure mismatch immediately |
| T3    | Yes — Gˢρw ≠ 0 at rest is the seed |
| T4    | Yes — Δt=20s case fails its drift bound by 6 orders of magnitude |
| T5    | Yes — same, on lat-lon |
| T6    | Probably no — the bug is not a Bounded/Periodic asymmetry per se, but T6 catches a *different* class of related bug |
| T7    | Yes — N-dependent amplification would be caught here |
| T8    | No — at Δt=0.01s the bug is dormant; this catches operator bugs not stability bugs |
| T9    | No — pulse arrival time is fine when the bug is at machine zero |
| T10   | Yes — the unstable mode is non-conservative |
| T11   | Probably no — WENO at rest has worked in our diagnostic |
| T12   | Yes — IGW wouldn't propagate cleanly at Δt=2s once the rest mode is excited |
| T13   | Yes — this is the production test, it would fire within 1 hour sim |

The current suite has only T14. With T1-T13 added (about 1 day of work to
implement them as `@test` blocks calling existing utility code), the bug we
spent multiple iterations finding would have surfaced on the *first* CI run
that touched the substepper.

---

# Tier 0 — initialization robustness

T1-T14 above test the *substepper*. But many user-visible failures come
from the **setup scripts** themselves — model-construction calls,
`set!`-ordering, reference-state choices, checkpoint I/O. These tests
guarantee that whatever a user types into a script produces a model state
that is internally consistent before the first time step is taken.

## I1: `set!` produces a model state that round-trips through the EoS

**Catches:** user passes `θ` and `ρ` that are individually right but
mutually inconsistent; or `set!`'s prioritization produces a state where
`p` from `update_state!` doesn't match what the user intended.

**Setup** — any AtmosphereModel with a CompressibleDynamics; pick the T1 grid.

**Procedure**
```julia
set!(model; θ = θᵇᵍ, ρ = ref.density)
Oceananigans.TimeSteppers.update_state!(model)
p_after_set = copy(interior(model.dynamics.pressure))

# Save state, re-create model, re-set with the EoS-derived pressure target,
# update_state! again — should give bit-identical pressure
ρ_saved = copy(interior(BreezeBC.dynamics_density(model.dynamics)))
ρθ_saved = copy(interior(model.formulation.potential_temperature_density))

model2 = build_t1_model()
set!(model2;
     ρ  = (x, y, z) -> ref.density(z),
     ρθ = (x, y, z) -> ref.density(z) * θᵇᵍ(z))
Oceananigans.TimeSteppers.update_state!(model2)
p_after_direct = copy(interior(model2.dynamics.pressure))
```

**Pass:** `maximum(abs, p_after_set - p_after_direct) ≤ 1e-9 * maximum(abs, p_after_set)`.

**Why:** the two `set!` paths (`θ + ρ` vs `ρθ + ρ`) must produce bit-identical
pressure once the EoS has run. If they disagree, `set!`'s priority ordering
is silently changing the state, and any downstream `make_pressure_correction!`
is fighting the IC.

---

## I2: Reference state is in discrete hydrostatic balance with the *substepper's* operators

**Catches:** the reference state is mathematically right but discretely
inconsistent with the operators the substepper uses for ∂z and z-face
averaging — exactly the seed in our failing case.

**Setup** — same as T1 but iterate over reference-state constructors:
isothermal-T₀ ∈ {220, 250, 280} K, plus an N²-stratified reference with
N² ∈ {1e-4, 4e-4}.

**Procedure** — for each reference, compute the discrete residual ε from
T1, additionally check that the *exact* operators used in the substepper's
slow-vertical-momentum-tendency assembly satisfy

```
maximum(abs, (-∂z(p_ref) - g·ρ_ref_face)) ≤ 100 * eps(Float64) * |g·ρ_max|
```

**Pass:** all reference-state constructors satisfy the bound.

**Why:** if isothermal-T₀ passes and N²-stratified fails (or vice-versa),
the reference-state construction is using a different discretization than
the substepper expects. This must be caught at construction time, not after
a 30-day run.

---

## I3: Checkpoint round-trip is bit-exact

**Catches:** save / load loses precision, reorders fields, drops halos, or
silently re-derives pressure from a different EoS path.

**Procedure**
```julia
model = build_full_setup()  # all physics turned on
set_full_ic!(model)
save_checkpoint(model, "/tmp/test_ckpt.jld2")
model2 = build_full_setup()
load_ic!(model2, "/tmp/test_ckpt.jld2")
```

**Pass:** for every prognostic field f, `parent(f_model) == parent(f_model2)`
exactly (no tolerance — equal bit pattern).

**Why:** users will checkpoint and resume. If the resumed state is even
1 ulp different, two runs that look "the same" diverge after a week of
sim time, undermining reproducibility claims.

---

## I4: Resolution-cascade interpolation preserves discrete balance

**Catches:** going from a 1° checkpoint to a ¼° checkpoint via
interpolation produces a state that's no longer balanced, so the ¼° run
blows up not because of substepper bugs but because the IC is broken.

**Procedure** — set up balanced-jet (DCMIP without perturbation) on a 1°
grid. Run for 1 sim hour. Save checkpoint. Build a ¼° model on the same
domain. Interpolate the 1° checkpoint onto the ¼° grid (existing BBI
`load_ic_interpolated!`).

```julia
res_1deg = compute_residuals(model_1deg)  # I2-style residual
res_qdeg = compute_residuals(model_qdeg)
```

**Pass:** `max(res_qdeg) ≤ 10 × max(res_1deg)` (the interpolation may amplify
imbalance by up to factor of 10, but not turn it from "negligible" to
"NaN-driving"). Also: `max|w|_qdeg ≤ 0.1 m/s` after the first 5 outer
steps on the ¼° grid (no immediate spurious convection).

**Why:** the cascade approach (1° spinup → ¼° → ⅛°) is a core BBI feature;
each handoff must pass an interpolation-quality bar. The bound `0.1 m/s`
is what you'd see for a smooth atmospheric flow; anything above 1 m/s
indicates the interpolated IC is shock-like.

---

## I5: First time step is NaN-free for every supported configuration

**Catches:** an IC builder accepts a kwarg combination it shouldn't, or a
boundary condition introduces a halo NaN.

**Procedure** — table-driven test over all supported `(grid type, surface
fluxes on/off, microphysics on/off, sponge on/off, viscous_sponge on/off,
Coriolis variant)` combinations. For each, build the model, set the IC,
take one outer step at the *smallest* recommended Δt for that
configuration, check no field has NaN.

**Pass:** all configurations produce finite state after one step.

**Why:** users will mix-and-match these flags. Every product-of-options
must work, and silent NaNs in moisture fields (e.g. ρqᶜⁱ when no ice
microphysics is wired) are common gotchas.

---

## I6: All prognostic fields have non-default BCs where physically required

**Catches:** the `ρe` BC bug we found — surface flux BCs attached to a
field that doesn't exist in the chosen formulation, silently dropped.

**Procedure** — after `set_analytic_ic!`, walk the model's
`prognostic_fields(model)` and check each field's `boundary_conditions`:

```julia
for (name, f) in pairs(prognostic_fields(model))
    @assert !is_default_noflux_bc(f.boundary_conditions, name) ||
            name in (:ρqᶜˡ, :ρqᶜⁱ, :ρqʳ, :ρqˢ)  # whitelist
end
```

i.e., every prognostic field that's *expected* to have a flux BC must have
a non-default one. The whitelist is the set of fields where no-flux is the
right answer.

**Pass:** assertion passes for every supported configuration.

**Why:** silent BC misnamings (`ρe` vs `ρθ`) lead to a model that runs but
produces wrong answers. Catching this at IC time means it can't ship.

---

# Tier 3 — full-physics validation cases

Each of these is a complete, runnable case that exercises the substepper
together with all the physics modules a real user would turn on. Failure
of any one of these means the *integrated* system is broken even when the
unit tests pass.

## F1: DCMIP-2016 Test 1-1 — moist baroclinic wave

This is the test we've been wrestling with. It is the standard moist-physics
benchmark for global atmospheric models.

**Setup** (verbatim from `Breeze/examples/moist_baroclinic_wave.jl`)
- Grid: LatitudeLongitudeGrid, 1° (Nλ=360, Nφ=160), Nz=64, halo (5,5,5),
  longitude (0, 360), latitude (-80, 80), z=(0, 30km), uniform.
- Coriolis: HydrostaticSphericalCoriolis with Ω = 7.29212e-5 s⁻¹.
- Reference state: isothermal at T₀ = 250 K.
- Microphysics: OneMomentCloudMicrophysics with NonEquilibriumCloudFormation,
  τ_relax = 200 s for both liquid and ice.
- Surface fluxes: BulkDrag (Cᴰ=1e-3, Uᵍ=1e-2 m/s) on ρu, ρv;
  BulkSensibleHeatFlux on ρθ; BulkVaporFlux on ρqᵛ; T_surface = analytic
  surface virtual temperature.
- IC: DCMIP-2016 analytic balanced jet + 1 m/s zonal-wind perturbation
  centered at (20°E, 40°N), tapered above z=15 km.
- Δt = 20 s (the published "rock-stable" value), stop_time = 15 days.

**Observables** (post-processed from JLD2 output):
1. `max|w|(t)` envelope over the run.
2. Surface pressure minimum vs time (the developing cyclone deepens p_surf).
3. Total atmospheric mass `M(t) = ∫ρ dV`.
4. Total ρθ `H(t) = ∫ρθ dV`.
5. Day-9 snapshot of θ' at z = 850 hPa (the BCI is fully developed).

**Pass criteria:**
- (a) Run completes 15 days without NaN.
- (b) `max|w|(t) ≤ 5 m/s` for all t — vertical motion in a healthy BCI is
  tens of cm/s; 5 m/s is the rough upper bound on the physical maximum
  (intense fronts) and would-be runaway acoustic modes.
- (c) Cyclone deepens to surface-pressure minimum < 970 hPa by day 9 (the
  textbook DCMIP result is 950-960 hPa at this resolution).
- (d) `|M(t) - M(0)| / M(0) ≤ 1e-8` over the full 15 days (no microphysics
  → mass exactly conserved).
- (e) Day-9 θ' snapshot has the canonical spiral structure: visual
  comparison against published DCMIP-2016 figures + a quantitative bound on
  zonal-wavenumber-7 amplitude (the dominant baroclinic mode).

**What it catches:** everything above plus end-to-end physics-substepper
coupling, surface-flux feedback, microphysics latent-heat release. If T1-T13
all pass and F1 fails, the bug is in a *physics* module, not the substepper
math.

---

## F2: Jablonowski-Williamson dry baroclinic wave (DCMIP 2008 Test 4-1)

The dry analog of F1. Standard intercomparison test with published
reference solutions from many models.

**Setup**
- Same grid as F1 except Lz=30 km is canonical here too.
- No microphysics, no surface fluxes (free-slip bottom).
- Same DCMIP IC formulas without the moisture profile.
- Δt = 225 s (the published advective-CFL step on this grid), stop_time = 14 days.

**Observables:** same as F1, plus:
- Surface pressure pattern at day 9 (compared against the JW06 reference).
- Zonal-mean U at 850 hPa at day 14.

**Pass criteria:**
- (a) Run completes 14 days.
- (b) Surface pressure minimum at day 9 within ±5 hPa of JW06 reference
  (≈940 hPa).
- (c) Zonal-mean U at 850 hPa at day 14 within ±2 m/s of JW06 in 30°-70°N
  band.
- (d) Mass conserved to 1e-10.

**What it catches:** dry-dynamics-only end-to-end correctness with no
moist confounders. If F1 fails but F2 passes, microphysics-substepper
coupling is the issue.

---

## F3: Held-Suarez climate

Long-time-statistical test. Tests stability and *reasonable physics* across
timescales no acoustic test can probe.

**Setup**
- Grid: same as F1 / F2.
- No moisture, no microphysics.
- Newtonian relaxation toward a prescribed equilibrium temperature
  T_eq(φ, p) (Held-Suarez 1994 specification).
- Rayleigh damping of u, v in the lower boundary layer (k_v(σ) profile from
  Held-Suarez).
- Δt = 225 s, stop_time = 1200 days. Discard first 200 days as spinup.

**Observables** (averaged over days 200-1200):
1. Zonal-mean U(φ, p).
2. Zonal-mean T(φ, p).
3. Eddy KE: ∫(u'² + v'²)/2 averaged in upper troposphere.
4. Mass and total energy.

**Pass criteria:**
- (a) Run completes 1200 days.
- (b) Zonal-mean U pattern reproduces the canonical Held-Suarez double-jet
  structure: jet maxima at ±35-45°N at 200-300 hPa with peak U = 28±3 m/s.
- (c) Eddy KE within ±20% of multi-model-mean in published intercomparisons.
- (d) Mass drift over 1200 days ≤ 1e-7 (allows for a few dozen ulp/step).

**What it catches:** robustness over O(10⁵) outer steps. Many bugs that
look "stable" over 14 days drift slowly and break by day 200.

---

## F4: Aquaplanet with simple moist physics (APE-style)

Adds moisture and surface evaporation to F3, but keeps physics minimal.

**Setup**
- Same grid.
- Surface temperature: zonal-mean prescribed function T_surf(φ) (APE
  "Qobs" profile).
- Microphysics: NonEquilibriumCloudFormation, τ_relax = 200 s, plus a
  simple Betts-Miller-style relaxation toward a moist adiabat (or whatever
  the simplest moist physics package Breeze ships).
- Surface fluxes: bulk drag, sensible heat, water vapor.
- Δt = 60 s, stop_time = 360 days.

**Observables:**
1. Zonal-mean precipitation P(φ).
2. Zonal-mean U, T, q.
3. ITCZ position (peak of P(φ)).
4. Mass, water-vapor mass conservation.

**Pass criteria:**
- (a) Run completes 360 days.
- (b) ITCZ within ±2° of equator.
- (c) Zonal-mean precipitation peak ≥ 5 mm/day at the ITCZ.
- (d) Water vapor mass conserved (within microphysics-driven sources/sinks
  budgeted as `∫(C-E)dV dt`) to within 0.1%.

**What it catches:** moist-physics stability over O(10⁶) outer steps, plus
mass-conservation correctness in the presence of moisture sources and sinks.

---

## F5: 2-D moist supercell (mid-latitude convection)

Curvilinear-grid stress test: dense moist convection on a Cartesian grid
embedded inside the lat-lon code path.

**Setup**
- Grid: RectilinearGrid 100×100×40, Δx=Δy=2 km, Δz=500 m, periodic xy,
  Bounded z, halo (5, 5, 5).
- IC: Weisman-Klemp idealized supercell environment (CAPE ≈ 2200 J/kg,
  shear vector turning with height) plus a warm bubble at the center.
- Coriolis: f₀ = 1e-4 s⁻¹.
- Microphysics: OneMomentCloudMicrophysics with rain + snow.
- No surface fluxes.
- Δt = 4 s, stop_time = 2 hours.

**Observables:**
- Updraft helicity, peak max|w|(t), peak max|q^r|(t).
- Storm propagation speed and direction.
- Reflectivity-style Z field at 1 km height.

**Pass criteria:**
- (a) Run completes 2 hours.
- (b) Peak max|w| reaches ≥ 30 m/s (a real supercell).
- (c) Storm propagates rightward of the mean wind by the predicted angle
  (Bunkers et al. 2000 right-mover diagnostic).
- (d) `max|w|` has stayed bounded — no acoustic-mode runaway behind the
  real updraft.

**What it catches:** moist physics in a convective, non-hydrostatic regime.
Bugs that pass F1 by virtue of small w can fail F5. Also tests reasonable
behavior under intense vertical motion.

---

# Recommended implementation order

Implementing this whole suite is ~3-4 weeks. Recommended ordering:

1. **Week 1 — Tier 0 + Tier 1 (T1-T7).** Initialization-robustness and
   rest-atmosphere drift tests. These are cheap, run on every PR, and
   would have caught the bug we found.
2. **Week 2 — Tier 1 (T8-T11), Tier 2 (T12, T13).** Acoustic-pulse, IGW,
   conservation, balanced-jet drift. These catch correctness bugs in the
   acoustic core and the BCI-specific failure mode without needing full
   physics.
3. **Week 3 — F2 (dry BW), F1 (moist BW).** Once Tier 0/1/2 pass, the
   full-physics BW cases should pass too. If they don't, the bug is in
   physics-substepper coupling, and the failing observable points at it.
4. **Week 4 — F3, F4, F5.** Long-time and convective tests for the
   production-readiness bar.

A passing run of all 14 unit tests + 5 full-physics tests is what I'd want
to see before claiming the substepper is "stable enough to ship to users".
