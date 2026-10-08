import struct
import unittest
import torch
from back_prop.common.checkpoint_compat import adapt_cuda_rng_state


class CudaRngCompatibilityTests(unittest.TestCase):
    def test_seed_and_nonzero_offset_roundtrip(self):
        payload = torch.tensor(list(struct.pack('<Qq', 43, 98765432)), dtype=torch.uint8)
        old = torch.cat((torch.full((800,), 255, dtype=torch.uint8), payload))
        new = adapt_cuda_rng_state(old, 16)
        self.assertEqual(struct.unpack('<Qq', bytes(new.tolist())), (43, 98765432))
        torch.testing.assert_close(adapt_cuda_rng_state(new, 816), old)
        self.assertIs(adapt_cuda_rng_state(old, 816), old)

    def test_unknown_formats_are_rejected(self):
        with self.assertRaises(ValueError):
            adapt_cuda_rng_state(torch.zeros(816, dtype=torch.uint8), 16)
        with self.assertRaises(ValueError):
            adapt_cuda_rng_state(torch.zeros(128, dtype=torch.uint8), 16)


if __name__ == '__main__':
    unittest.main(verbosity=2)
