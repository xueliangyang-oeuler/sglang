## PR Motivation

Currently, when the `SGLANG_MAMBA_CONV_DTYPE` environment variable is not set, the Mamba2 state dtype defaults to `bfloat16` without considering the model's actual configuration. This can lead to dtype mismatches when loading models that were trained or configured with specific dtypes (e.g., `float16` or `float32`).

This PR addresses this issue by:
1. **Inferring conv_dtype from model config** when the environment variable is not set, ensuring better alignment with the model's intended configuration
2. **Maintaining backward compatibility** - the environment variable still takes highest priority
3. **Supporting both text and VL models** by checking both `config.torch_dtype` and `config.text_config.torch_dtype`

## PR Modifications

### 1. Enhanced `mamba2_state_dtype()` Function (`python/sglang/srt/configs/mamba_utils.py`)

Modified the `mamba2_state_dtype()` function to implement a priority-based dtype resolution:

**Priority Order (from highest to lowest):**
1. Environment variable `SGLANG_MAMBA_CONV_DTYPE`
2. Model config (`config.torch_dtype` or `config.text_config.torch_dtype` for VL models)
3. Default value `bfloat16`

**Key Changes:**
- Added logic to read `torch_dtype` from model config when env var is not set
- Handles both text models (`config.torch_dtype`) and VL models (`config.text_config.torch_dtype`)
- Validates config dtype against supported dtypes (`float32`, `bfloat16`, `float16`)
- Falls back to `bfloat16` if config dtype is invalid or not specified

### 2. Environment Variable Default Update (`python/sglang/srt/environ.py`)

Changed `SGLANG_MAMBA_CONV_DTYPE` default from `"bfloat16"` to `None`:
- This allows the system to detect when the env var is explicitly unset
- When `None`, the function will attempt to infer from model config
- When explicitly set, the env var still overrides all other sources

## Behavior Changes

| Scenario | Before | After |
|----------|--------|-------|
| Env var set | Use env var value | Use env var value (unchanged) |
| Env var not set, config has torch_dtype | Default to bfloat16 | Use config.torch_dtype |
| Env var not set, config has no torch_dtype | Default to bfloat16 | Default to bfloat16 (unchanged) |

## Supported Models

This change benefits all Mamba2-based models including:
- NemotronH
- FalconH1
- Qwen3Next
- LFM2
- And other Mamba2 architecture models

## Testing

- [x] Verified dtype inference works with text models (config.torch_dtype)
- [x] Verified dtype inference works with VL models (config.text_config.torch_dtype)
- [x] Verified environment variable still takes priority
- [x] Verified fallback to bfloat16 when config dtype is invalid
- [x] Verified backward compatibility with existing configurations
