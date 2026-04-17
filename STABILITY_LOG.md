# Stability Log

Chronological record of the 1° moist-BCI Phase-1 stability sweep. Goal: get the
14-day spin-up through to completion and figure out the maximum stable CFL.

All runs use `experiments/single_gpu_cascade.jl`, single H200, Float32 unless
noted, `SplitExplicitTimeDiscretization` (acoustic substepping with adaptive
substep count, minimum 6 per outer step), `conjure_time_step_wizard!` for Δt.

`cfl_target` is the wizard's `cfl` keyword. `cfl_achieved` is the largest CFL
actually reached during the run, derived from the logged Δt and either the
jet speed (U≈60 m/s at BCI peak) or the characteristic initial U≈15 m/s.
Blowup is whenever `NaN found in field ρ` fires.

## 2026-04-17

### Runs

| # | lat      | H (km) | z-grid    | sponge (rate, width) | polar filt | cfl_target | max_Δt (s)        | cloud_τ (s) | Δt (s) | cfl_achieved | blowup time | notes                                   |
|---|----------|--------|-----------|----------------------|------------|-----------|--------------------|-------------|--------|---------------|-------------|-----------------------------------------|
| 4 | -80,80   | 30     | uniform   | -                    | -          | 0.7       | 150                | 120         | 150    | 0.93          | 16.0 h      | original baseline                       |
| 5 | -80,80   | 30     | uniform   | -                    | -          | 0.7       | 100                | 120         | 100    | 0.62          | 17.6 h      |                                         |
| 6 | -80,80   | 30     | uniform   | -                    | -          | 0.7       | 60                 | 120         | 60     | 0.37          | 18.3 h      |                                         |
| 7 | -80,80   | 30     | uniform   | -                    | -          | 0.7       | 60                 | 600         | 60     | 0.37          | 26.4 h      | τ raised; moisture feedback slower      |
| 8 | -80,80   | 45     | stretched | 1/600, 7 km          | -          | 0.5       | 120                | 600         | 120    | 0.74          | 38.9 h      | +domain, +stretch, +sponge              |
| 9 | -75,75   | 45     | stretched | 1/600, 7 km          | -          | 0.7       | spinup=120, Inf    | 600         | 120    | 0.25          | 39.0 h      | Δx_min 3× larger → cfl lower per Δt     |
| 10| -75,75   | 45     | stretched | 1/180, 12 km         | -          | 0.7       | spinup=120, Inf    | 600         | 120    | 0.25          | 26.5 h      | sponge→0 creates jet/top gradient; worse|
| 11| -75,75   | 45     | stretched | 1/600, 7 km          | -          | 0.7       | spinup=30, Inf     | 600         | 30     | 0.06          | 46.0 h      | smaller Δt ≠ stable; same blowup mode   |
| 12| -75,75   | 45     | stretched | 1/600, 7 km          | 60° thres  | 0.7       | spinup=30, Inf     | 600         | 30     | 0.06          | 46.0 h      | polar filter had **no** effect          |

### Takeaways

- **Maximum stable CFL reached: 0.74 (run 8)**, but only until t=39 h. That CFL
  value came from running at Δt=120 s on a grid with Δx_min≈9.7 km and U≈60 m/s.
  Higher max_Δt (run 4 at CFL 0.93) died much sooner.
- Lowering CFL (and Δt) to 0.06 only bought ~7 h of extra sim time before the
  same ρw blow-up reappeared in the top cells. The blow-up is **not CFL-driven.**
- Every mechanical fix (larger domain, stretched z, top sponge) pushed the
  blow-up later. Every fix targeting *dynamics* (polar filter, stronger sponge,
  lower CFL alone) either had zero effect or made things worse.
- Growth signature: ρw in the top cells doubles every ~4 h starting around day
  1. At the moment of NaN, ρ in a handful of top cells has been evacuated to
  half its normal value. In physical units (top ρ ≈ 5×10⁻⁴ kg/m³), ρw ≈ 0.05
  already corresponds to w ≈ 100 m/s — purely spurious.

### Working hypothesis

Float32 precision in the top cells. After Δt-independent growth signatures
and the lack of response to polar filter / CFL lowering, the most likely
driver is round-off in `w = ρw / ρ` (and the thermodynamic inversions) in
cells where ρ is 10⁻⁴ kg/m³. Under Float32 that's ~10⁻¹¹ relative precision
for w when ρu, ρw are order 10⁻⁴. Any slow pressure imbalance at the top gets
amplified exponentially by the acoustic substepper because the substepper
implicitly solves for (ρw)″, (ρθ)″ in each substep and the off-centering has
a small positive growth rate at high Courant numbers.

### Next experiment

Float64. If this stabilises, the answer is "Float32 is fine below ~30 km but
the 45 km / 5×10⁻⁴ kg·m⁻³ top cell needs 64-bit." If not, the sponge target
(currently zero for ρu, ρv, ρw) needs to be the balanced zonal wind instead.
