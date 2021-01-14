import unittest
import torch
from onda.modules.diagonal_upsample import DiagonalUpsample


class TestDiagonalUpsample(unittest.TestCase):

    def test_base_case(self):
        """
        test convolution and deconvolution with a simple tensor

        """
        up_diagonal = torch.FloatTensor([[1,  3],
                                         [5,  7]])
        up_diagonal.unsqueeze_(0).unsqueeze_(0)

        down_diagonal = torch.FloatTensor([[2,  4],
                                           [6,  8]])
        down_diagonal.unsqueeze_(0).unsqueeze_(0)

        diagonal_upsample = DiagonalUpsample(in_channels=1)
        upsampled = diagonal_upsample((up_diagonal, down_diagonal))

        expected_upsampled = torch.FloatTensor([[2, 1, 4, 3],
                                                [1, 2, 3, 4],
                                                [6, 5, 8, 7],
                                                [5, 6, 7, 8]]).unsqueeze_(dim=0).unsqueeze_(dim=0)

        self.assertTrue(torch.all(expected_upsampled == upsampled))

    def test_multichannel(self):
        """
        Test with tensors with multiple channels
        """
        batch_size = 8
        n_channels = 3
        up_diagonal = torch.rand(batch_size, n_channels, 16, 16)
        down_diagonal = torch.rand(batch_size, n_channels, 16, 16)
        diagonal_upsample = DiagonalUpsample(in_channels=n_channels)
        upsampled = diagonal_upsample((up_diagonal, down_diagonal))

        self.assertEquals(upsampled.shape, (batch_size, n_channels, 32, 32))

        # Cycle through all elements and confirm they are correctly upsampled in a diagonal way
        for row in range(0, up_diagonal.shape[2]):

            for column in range(0, up_diagonal.shape[3]):
                dd = down_diagonal[:, :, row, column]
                ud = up_diagonal[:, :, row, column]
                u = upsampled[:, :, 2 * row: (2 * row) + 2, 2 * column: (2 * column) + 2]

                self.assertTrue(torch.all(u[:, :, 0, 0] == dd))
                self.assertTrue(torch.all(u[:, :, 1, 1] == dd))
                self.assertTrue(torch.all(u[:, :, 0, 1] == ud))
                self.assertTrue(torch.all(u[:, :, 1, 0] == ud))


if __name__ == '__main__':
    unittest.main()
