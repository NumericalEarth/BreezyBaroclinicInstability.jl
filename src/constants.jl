#####
##### DCMIP-2016 Moist Baroclinic Wave — Physical Constants
#####
# Reference: Ullrich, Jablonowski, Reed, Zarzycki, Lauritzen, Nair, Kent, Verlet-Banide (2016)
#   "Dynamical Core Model Intercomparison Project (DCMIP) Test Case Document"

# ═══════════════════════════════════════════════════════════════════════════
# Physical constants
# ═══════════════════════════════════════════════════════════════════════════

const earth_radius   = 6371220.0      # [m]
const earth_rotation = 7.29212e-5     # [s⁻¹]
const gravity        = 9.80616        # [m s⁻²]
const Rd_dry         = 287.0          # [J kg⁻¹ K⁻¹]  dry-air gas constant
const cp_dry         = 1004.5         # [J kg⁻¹ K⁻¹]  specific heat at constant pressure
const κ_exponent     = 2.0 / 7.0     # Rd / cp
const Rv_vapor       = 461.5          # [J kg⁻¹ K⁻¹]  water-vapor gas constant
const ε_virtual      = 0.608          # Rv/Rd − 1 ≈ 0.608  (virtual-temperature factor)
const p_ref          = 1e5            # [Pa]  reference surface pressure

# ═══════════════════════════════════════════════════════════════════════════
# Balanced-state parameters  (Table VI, Ullrich et al. 2014)
# ═══════════════════════════════════════════════════════════════════════════

const T_equator    = 310.0            # [K]   equatorial surface temperature
const T_polar      = 240.0            # [K]   polar surface temperature
const T_mean       = 0.5 * (T_equator + T_polar)   # 275 K
const lapse_rate   = 0.005            # [K m⁻¹]  lapse-rate parameter Γ
const jet_width    = 3.0              # K parameter  (jet width)
const vert_width   = 2.0              # b parameter  (vertical half-width)

# Derived coefficients for the τ-integrals  (Eqs. 5–8)
const coeff_A      = 1.0 / lapse_rate
const coeff_B      = (T_mean - T_polar) / (T_mean * T_polar)
const coeff_C      = 0.5 * (jet_width + 2) * (T_equator - T_polar) / (T_equator * T_polar)
const scale_height = Rd_dry * T_mean / gravity     # ≈ 8 km

# ═══════════════════════════════════════════════════════════════════════════
# Perturbation parameters  (exponential type, Eq. 14)
# ═══════════════════════════════════════════════════════════════════════════

const u_perturb  = 1.0               # [m s⁻¹]  amplitude
const r_perturb  = 0.1               # perturbation radius  [Earth radii]
const λ_center   = π / 9             # 20°E   [rad]
const φ_center   = 2π / 9            # 40°N   [rad]
const z_perturb  = 15000.0           # [m]    height cap

# ═══════════════════════════════════════════════════════════════════════════
# Moisture parameters  (DCMIP-2016, Eq. 18)
# ═══════════════════════════════════════════════════════════════════════════
#   q(λ,φ,η) = q₀ exp[−(φ/φ_w)⁴] exp[−((η−1)p₀/p_w)²]   for η > η_t
#   q         = q_t                                          above tropopause

const q0_surface   = 0.018            # [kg kg⁻¹]  peak specific humidity
const φ_width      = 2π / 9           # [rad]  40° — latitudinal e-folding width
const p_width      = 34000.0          # [Pa]   340 hPa — vertical pressure width
const η_tropopause = 0.1              # p/p_ref cutoff
const q_tropopause = 1e-12            # [kg kg⁻¹]  humidity above the tropopause
