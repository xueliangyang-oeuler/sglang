## PR Motivation

When using FP8 KV cache with FlashAttention backend, the code was attempting to use FP8 attention even when `layer.k_scale` was `None`. This caused:

1. **Incorrect output** - Missing descaling factors (`k_descale`, `v_descale`) led to garbage output
2. **Potential crashes** - FP8 operations without proper scaling factors can cause numerical instability

The root cause was that the FP8 path check did not verify whether `k_scale` and `v_scale` were actually available (not None) before proceeding with FP8 attention computation.

This PR adds a check for `layer.k_scale is not None` to the FP8 path condition, ensuring that when scaling factors are not available, the code falls back to the non-FP8 path (using FP16/BF16), which produces correct results.

Fixes #20820

## PR Modifications

### 1. Enhanced FP8 Path Condition (`python/sglang/srt/layers/attention/flashattention_backend.py`)

Added check for `layer.k_scale is not None` to the FP8 attention path condition:

```python
# Before (could use FP8 path even when k_scale is None):
if (
    self.kv_cache_dtype_str != "auto"
    and layer.head_dim <= 256
    and self.fa_impl_ver != 4
):
    # FP8 path - problematic when k_scale is None

# After (falls back to non-FP8 when k_scale is None):
if (
    self.kv_cache_dtype_str != "auto"
    and layer.head_dim <= 256
    and self.fa_impl_ver != 4
    and layer.k_scale is not None  # ✅ Added check
):
    # FP8 path - only when scaling factors are available
```

**Key Changes:**
- Added `layer.k_scale is not None` check to the FP8 path condition
- When `k_scale` is `None`, the code falls back to non-FP8 path (FP16/BF16)
- Added detailed comment explaining why this check is necessary
- Ensures correct output when quantization scaling factors are not available

### 2. Code Comments

Added explanatory comment:
```python
# 5) k_scale and v_scale are actually available (not None).
# If k_scale is None, converting to fp8 would produce garbage output due to
# missing descaling factors, so we fall back to non-fp8 path.
```

## Behavior Changes

| Scenario | Before | After |
|----------|--------|-------|
| FP8 KV enabled, k_scale available | FP8 attention | FP8 attention (unchanged) |
| FP8 KV enabled, k_scale is None | FP8 attention (garbage output) | Non-FP8 attention (correct output) ✅ |
| FP8 KV disabled | Non-FP8 attention | Non-FP8 attention (unchanged) |

## Backward Compatibility

- **No breaking changes**: This is a bug fix that ensures correct behavior
- **Improved correctness**: Models that previously produced incorrect output due to missing scaling factors now work correctly
- **No API changes**: The change is internal to the attention backend

## Testing

- [x] Verified FP8 attention works correctly when k_scale is available
- [x] Verified fallback to non-FP8 when k_scale is None
- [x] Verified correct output quality after the fix
- [x] Verified no regression for non-FP8 paths
- [x] Verified fix resolves issue #20820
