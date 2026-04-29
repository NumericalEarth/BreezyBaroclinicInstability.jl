#####
##### Acoustic substepper column-matrix eigenvalue scan
#####

This note preserves the rationale for a Breeze test that was removed from CI
because it was too specialized and research-diagnostic for the main test suite.
The removed file was `test/substepper_eigenvalue_scan.jl` on the
`glw/hevi-imex-docs` Breeze branch.

The diagnostic reconstructed the one-column tridiagonal operator used by the
acoustic substepper's implicit vertical solve. The substepper solves a matrix of
the form

```text
A = I + δτ_new^2 M
```

for the vertical momentum-like variable `μw`. The diagnostic then computed the
eigenvalues of `M`. The intended stability argument was:

- if every eigenvalue of `M` has non-negative real part, then the implicit
  off-centered Crank-Nicolson acoustic solve should not become singular for a
  positive acoustic substep `δτ_new`;
- if an eigenvalue has negative real part, then some positive `δτ_new` can make
  `I + δτ_new^2 M` singular, which is a possible mechanism for acoustic
  instability.

This was meant to test hypotheses from the BBI substepper investigation:

- predictor / matrix-weight mismatch in the split-explicit acoustic solve;
- sign asymmetry in the buoyancy off-diagonal terms on stretched grids;
- incorrect inclusion of boundary rows in the tridiagonal solve.

Important caveat: this diagnostic is easy to get wrong. One earlier negative
eigenvalue result was a bug in the test construction, not in the model: it
included the top boundary face `k = Nz + 1`, which is not a solver row. That
spurious row used a zero face potential temperature and produced a fake
near-zero/negative row in the matrix. The relevant interior block is solver rows
`2:Nz`, corresponding to vertical faces `2:Nz`.

Why it was removed from Breeze CI:

- it depends on `LinearAlgebra.eigvals`, which requires declaring the stdlib in
  the test environment on modern Julia;
- it is not a minimal behavioral regression test;
- the physical meaning is indirect compared with rest-atmosphere drift,
  hydrostatic balance, and short integration tests;
- it is more appropriate as an investigation script than as a required CI test.

If reinstated, keep it out of the default test suite unless it is rewritten as a
small, clearly documented structural test. A better CI-level version would avoid
full spectral analysis and directly check the coefficient signs, boundary-row
placement, finite matrix entries, and positive diagonal dominance for a tiny
reference column.
