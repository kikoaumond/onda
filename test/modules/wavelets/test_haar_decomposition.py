import unittest
import torch
from onda.modules.wavelets import HaarDecomposition2D
import math


class TestHaarDecomposition(unittest.TestCase):

    def test_base_case(self):
        """
        test decomposition with defaults

        """

        t = torch.FloatTensor([[2,  10,  5,  16],
                               [15,  1,  14,  6],
                               [11,  8, 12, 14],
                               [3, 9, 7, 13]]).unsqueeze_(0).unsqueeze_(0)

        hd = HaarDecomposition2D(input_shape=t.shape)
        decomposition = hd(t)
        max_levels = math.floor(math.log2(min(t.shape[2:])))
        # check that the deocomposition has the expected shape
        expected_shape = (t.shape[0], max_levels + 1, t.shape[2], t.shape[3])
        self.assertEqual(decomposition.shape, expected_shape)

        #expected_level_one_low_pass = torch.FloatTensor([[3.5,   3.5,  5.5,  5.5],
        #                                                 [3.5,   3.5,  5.5,  5.5],
        #                                                 [11.5, 11.5, 13.5, 13.5],
        #                                                 [11.5, 11.5, 13.5, 13.5]])
        #self.assertTrue(torch.all(decomposition[0, 0, :, :] == expected_level_one_low_pass))

        expected_level_one_detail = torch.FloatTensor([[2.5, 1.5, -2.5,  -1],
                                                       [1.5, 2.5,  -1, -2.5],
                                                       [-2.5, -2.5, 1.5,  1.5],
                                                       [-2.5, -2.5,  1.5, 1.5]])
        self.assertTrue(torch.all(decomposition[0, 0, :, :] == expected_level_one_detail))

        expected_level_two_detail = torch.FloatTensor([[-5, 0, 0, 3],
                                                       [0, -5, 3, 0],
                                                       [0, 3, -5, 0],
                                                       [3, 0, 0, -5]])
        self.assertTrue(torch.all(decomposition[0, 1, :, :] == expected_level_two_detail))

        expected_level_two_low_pass = torch.FloatTensor([[8.5,  0,   0, 8.5],
                                                         [0,  8.5, 8.5,   0],
                                                         [0,  8.5, 8.5,   0],
                                                         [8.5,  0,   0, 8.5]])
        self.assertTrue(torch.all(decomposition[0, 2, :, :] == expected_level_two_low_pass))

        # check that the reconstructed tensor matches the original tensor
        reconstructed = HaarDecomposition2D.reconstruct(decomposition)

        self.assertTrue(torch.all(reconstructed == t))

    def test_single_channel(self):
        """
        Test decompostion of a single-channel random batch
        """
        batch_size = 1
        n_channels = 1
        batch = torch.rand(batch_size, n_channels, 4, 4)
        hd = HaarDecomposition2D(input_shape=batch.shape)
        decomposition = hd(batch)
        max_levels = math.floor(math.log2(min(batch.shape[2:])))
        # check that the decomposition has the expected shape
        expected_shape = (batch.shape[0], max_levels + 1, batch.shape[2], batch.shape[3])
        self.assertEquals(decomposition.shape, expected_shape)

    def test_sparsity(self):
        """
        test with sparsification
        """
        batch_size = 8
        n_channels = 3
        sparsity = 0.95
        batch = torch.rand(batch_size, n_channels, 32, 32)
        hd = HaarDecomposition2D(input_shape=batch.shape, sparsity=sparsity, by_channel=True)
        decomposition = hd(batch)
        # reconstruct the tensor and chcek that all the non-zero elements match the original tensor
        reconstructed = HaarDecomposition2D.reconstruct(decomposition, in_chanels=n_channels)
        self.assertTrue(torch.all(reconstructed[reconstructed != 0] == batch[reconstructed != 0]))

        max_levels = math.floor(math.log2(min(batch.shape[2:])))
        expected_shape = (batch.shape[0], 2 * max_levels, batch.shape[2], batch.shape[3])
        self.assertEquals(decomposition.shape, expected_shape)

        # Check sparsity for each individual tensor in the batch
        for batch_index in range(decomposition.shape[0]):
            image = decomposition[batch_index, ...]
            image_size = torch.numel(image)
            n_non_zero_elements = torch.numel(image[image != 0])
            actual_sparsity = 1 - (n_non_zero_elements / image_size)
            self.assertGreaterEqual(actual_sparsity, sparsity)


if __name__ == '__main__':
    unittest.main()
