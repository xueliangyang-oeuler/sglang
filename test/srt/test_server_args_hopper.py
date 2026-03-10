import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add python path
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "python",
    )
)

from sglang.srt.server_args import ServerArgs


class TestServerArgsHopper(unittest.TestCase):
    def test_hopper_default_backend_fa3(self):
        with patch(
            "sglang.srt.server_args.is_hopper_with_cuda_12_3", return_value=True
        ), patch(
            "sglang.srt.server_args.is_no_spec_infer_or_topk_one", return_value=True
        ), patch(
            "sglang.srt.server_args.is_sm100_supported", return_value=False
        ), patch(
            "sglang.srt.server_args.is_hip", return_value=False
        ), patch(
            "sglang.srt.server_args.is_flashinfer_available", return_value=True
        ):

            # Mock the instance
            mock_self = MagicMock()
            mock_self.speculative_algorithm = None
            mock_self.speculative_eagle_topk = None

            # Call the method directly from the class
            backend = ServerArgs._get_default_attn_backend(
                mock_self, use_mla_backend=False, model_config=MagicMock()
            )

            self.assertEqual(
                backend,
                "fa3",
                "Should default to fa3 on Hopper with CUDA 12.3 due to Issue #17411",
            )


if __name__ == "__main__":
    unittest.main()
