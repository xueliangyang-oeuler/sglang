import logging

from sglang.srt.environ import envs
from sglang.srt.utils import get_device_sm, is_blackwell_supported

logger = logging.getLogger(__name__)


def _compute_enable_deep_gemm():
    sm_version = get_device_sm()
    if sm_version < 90:
        return False

    try:
        import deep_gemm  # noqa: F401
    except ImportError:
        return False

    return envs.SGLANG_ENABLE_JIT_DEEPGEMM.get()


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()

DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and is_blackwell_supported()


def get_deep_gemm_scale_ue8m0(
    fp4_gemm_runner_backend: str = "auto",
    fp8_gemm_runner_backend: str = "auto",
) -> bool:
    if not DEEPGEMM_BLACKWELL:
        return False
    return fp4_gemm_runner_backend == "auto" and fp8_gemm_runner_backend == "auto"


DEEPGEMM_SCALE_UE8M0 = get_deep_gemm_scale_ue8m0()
