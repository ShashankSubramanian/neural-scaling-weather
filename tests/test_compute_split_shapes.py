import unittest

from parameterized import parameterized
from distributed.helpers import compute_split_shapes


class TestComputeSplitShapes(unittest.TestCase):
    @parameterized.expand(
        [[10, 4, 1], [10, 4, 2], [11, 4, 2], [721, 2, 4], [721, 8, 4], [721, 32, 4]]
    )
    def test_compute_split_shapes(self, size, num_chunks, patch_size):
        shapes = compute_split_shapes(size, num_chunks, patch_size)
        print(f"shapes: {shapes}")
        self.assertTrue(
            len(shapes) == num_chunks,
            "compute_split_shapes failed to return correct number of shapes",
        )
        self.assertTrue(
            sum(shapes) == size,
            "compute_split_shapes failed to return correct sum of shapes",
        )
        self.assertTrue(
            all([s % patch_size == 0 for s in shapes[:-1]]),
            "Some shape in the middle is not divisible by patch_size",
        )
        self.assertTrue(
            all([s > 0 for s in shapes]),
            "Some shape in the middle is not divisible by patch_size",
        )


if __name__ == "__main__":
    unittest.main()
