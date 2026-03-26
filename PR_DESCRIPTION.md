## PR Motivation

On Blackwell GPUs (SM120), when using `--fp8-gemm-backend auto`, DeepGEMM produces incorrect results if the checkpoint's scale format is not `ue8m0`. This causes:
- **Accuracy degradation** in model outputs
- **CUDA crashes** at high concurrency

The root cause is that DeepGEMM on Blackwell requires the `ue8m0` scale format to function correctly. When loading checkpoints with other scale formats (e.g., standard FP8 scales), DeepGEMM cannot handle them properly.

This PR fixes this by automatically falling back to the Triton backend when the checkpoint's scale format is not `ue8m0` on Blackwell GPUs. Triton is compatible with all scale formats and provides a safe fallback.

Fixes #20776

## PR Modifications

### 1. Enhanced `initialize_fp8_gemm_config()` Function (`python/sglang/srt/layers/quantization/fp8_utils.py`)

Added a new parameter `use_scale_ue8m0` to the function signature:

```python
def initialize_fp8_gemm_config(
    server_args: ServerArgs, use_scale_ue8m0: Optional[bool] = None
) -> None:
```

**New Logic for Blackwell (SM120):**
- When `backend == "auto"` and `is_sm120_supported()` is True:
  - If `use_scale_ue8m0 is False` (checkpoint uses non-ue8m0 format):
    - Fall back to `"triton"` backend
    - Log an info message about the fallback
  - Otherwise: Also fall back to `"triton"` (conservative approach for safety)

**Rationale:**
- DeepGEMM on Blackwell requires `ue8m0` scale format
- Non-ue8m0 formats cause incorrect results or crashes
- Triton backend is compatible with all scale formats
- Automatic fallback ensures stability without user intervention

### 2. Updated Scheduler Initialization (`python/sglang/srt/managers/scheduler.py`)

Modified the call to `initialize_fp8_gemm_config()` to pass the checkpoint's scale format information:

```python
initialize_fp8_gemm_config(
    self.server_args,
    getattr(self.model_config, "use_scale_ue8m0", None),
)
```

This ensures the FP8 GEMM configuration is aware of the checkpoint's scale format and can make appropriate backend decisions.

## Behavior Changes

| Scenario | Before | After |
|----------|--------|-------|
| Blackwell + auto backend + ue8m0 checkpoint | DeepGEMM enabled | Triton fallback (conservative) |
| Blackwell + auto backend + non-ue8m0 checkpoint | DeepGEMM enabled (incorrect results/crashes) | Triton fallback (safe) |
| Blackwell + explicit backend | As specified | As specified (unchanged) |
| Non-Blackwell | Original behavior | Original behavior (unchanged) |

## Backward Compatibility

- **Explicit backend selection**: When users explicitly set `--fp8-gemm-backend` to a specific value (e.g., `deep_gemm`, `triton`), the backend is used as specified without automatic fallback
- **Environment variables**: `SGLANG_ENABLE_FLASHINFER_FP8_GEMM` and `SGLANG_SUPPORT_CUTLASS_BLOCK_FP8` continue to work as before
- **Logging**: Clear info message is logged when fallback occurs, informing users of the backend selection

## Testing

- [x] Verified automatic fallback to Triton when `use_scale_ue8m0=False` on Blackwell
- [x] Verified Triton fallback on Blackwell when scale format is unknown
- [x] Verified explicit backend selection bypasses automatic fallback
- [x] Verified non-Blackwell GPUs are not affected by this change
- [x] Verified fix resolves accuracy issues reported in #20776
- [x] Verified fix resolves CUDA crashes at high concurrency
