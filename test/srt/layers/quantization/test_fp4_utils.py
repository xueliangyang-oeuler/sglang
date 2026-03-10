import os
import sys
import unittest

sys.path.append(
    os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
            )
        ),
        "python",
    )
)

from unittest.mock import MagicMock, patch

import sglang.srt.layers.quantization.fp4_utils as fp4_utils
from sglang.srt.layers.quantization.fp4_utils import (
    get_fp4_gemm_runner_backend,
    initialize_fp4_gemm_config,
)


class TestFp4Utils(unittest.TestCase):
    @patch("sglang.srt.layers.quantization.fp4_utils.is_sm120_supported")
    def test_auto_backend_selection_sm120(self, mock_is_sm120):
        # Case 1: SM120 detected, backend="auto" -> should use "flashinfer_cudnn"
        mock_is_sm120.return_value = True
        server_args = MagicMock()
        server_args.fp4_gemm_runner_backend = "auto"

        # Reset global
        fp4_utils.FP4_GEMM_RUNNER_BACKEND = None

        initialize_fp4_gemm_config(server_args)
        backend = get_fp4_gemm_runner_backend()
        self.assertTrue(backend.is_flashinfer_cudnn())
        self.assertEqual(backend.get_flashinfer_backend(), "cudnn")

    @patch("sglang.srt.layers.quantization.fp4_utils.is_sm120_supported")
    def test_auto_backend_selection_non_sm120(self, mock_is_sm120):
        # Case 2: SM120 NOT detected, backend="auto" -> should use "flashinfer_cutlass"
        mock_is_sm120.return_value = False
        server_args = MagicMock()
        server_args.fp4_gemm_runner_backend = "auto"

        # Reset global
        fp4_utils.FP4_GEMM_RUNNER_BACKEND = None

        initialize_fp4_gemm_config(server_args)
        backend = get_fp4_gemm_runner_backend()
        self.assertTrue(backend.is_flashinfer_cutlass())
        self.assertEqual(backend.get_flashinfer_backend(), "cutlass")


if __name__ == "__main__":
    unittest.main()
