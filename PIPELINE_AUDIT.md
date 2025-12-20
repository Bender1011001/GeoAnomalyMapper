# GeoAnomalyMapper Pipeline Audit Report
# Generated: 2024-12-18

## ISSUES IDENTIFIED

### 1. CRITICAL: No Regional Trend Removal (train_usa_pinn.py)
**Location**: Lines 99-106
**Problem**: Raw Bouguer gravity is used without removing regional trends.
- Nevada (-168 to -270 mGal) vs Coastal regions (~0 mGal)
- Model learns crustal thickness, not ore deposits
- Basin and Range shows as uniform negative, hiding Carlin deposits

**Fix Required**: Add upward continuation or polynomial trend removal before training.

---

### 2. CRITICAL: Edge Masking Breaks Data (predict_usa.py)
**Location**: Lines 90-98
**Problem**: Edge masking with NaN propagates to overlapping tiles, creating data gaps.
- Nevada shows 59% NODATA in V2 model vs 0% in V1
- The fix for edge artifacts created worse data holes

**Fix Required**: Use weighted averaging instead of NaN masking, or reduce overlap.

---

### 3. MODERATE: Fixed Depth Assumption (train_usa_pinn.py)
**Location**: Line 145
**Problem**: `mean_depth=200.0` is hardcoded.
- Carlin deposits are often 300-500m deep
- VMS can be near-surface
- Single depth assumption loses accuracy

**Recommendation**: Consider multi-scale or depth-varying approach.

---

### 4. MODERATE: Physics Layer Approximations (pinn_gravity_inversion.py)
**Location**: GravityPhysicsLayer.forward
**Problem**: Uses flat-earth approximation with single slab thickness.
- Works for local anomalies
- May not capture complex 3D geology
- Assumes uniform thickness (1000m)

**Recommendation**: Acceptable for this scale, but noted.

---

### 5. MINOR: Global Normalization Constants (train_usa_pinn.py)
**Location**: Lines 101-102
**Problem**: 
- GLOBAL_MAX_GRAVITY = 100.0 mGal
- Nevada values reach -270 mGal (clipped/distorted)

**Fix Required**: Use data-driven normalization or expand range.

---

### 6. MINOR: No Data Augmentation (train_usa_pinn.py)
**Problem**: No rotation, flip, or noise augmentation during training.
- Model may overfit to specific orientations
- Limited generalization

**Recommendation**: Add standard augmentations.

---

## RECOMMENDED FIX ORDER

1. **Compute Residual Gravity** (removes regional trends) - CRITICAL
2. **Fix edge masking logic** (weighted blend instead of NaN) - CRITICAL  
3. **Expand normalization range** to handle -300 to +300 mGal - MODERATE
4. **Add data augmentation** - MINOR
5. **Retrain model** with fixes applied

## ESTIMATED EFFORT

| Fix | Time | Impact |
|-----|------|--------|
| Residual gravity preprocessing | 15 min code + 30 min compute | HIGH |
| Edge masking fix | 10 min | HIGH |
| Normalization fix | 5 min | MEDIUM |
| Data augmentation | 10 min | LOW |
| Full retraining | 30-60 min | - |

**Total estimated time: ~1-2 hours**
