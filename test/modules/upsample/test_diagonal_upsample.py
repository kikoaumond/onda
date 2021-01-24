import unittest
import torch
from upsample.diagonal_upsample import DiagonalUpsample
from onda.modules.wavelets.haar import UP_DIAGONAL, DOWN_DIAGONAL


class TestDiagonalUpsample(unittest.TestCase):

    def test_base_case(self):
        """
        test diagonal upsampling with a simple tensor

        """
        up_diagonal = torch.FloatTensor([[1,  3],
                                         [5,  7]])
        up_diagonal.unsqueeze_(0).unsqueeze_(0)

        down_diagonal = torch.FloatTensor([[2,  4],
                                           [6,  8]])
        down_diagonal.unsqueeze_(0).unsqueeze_(0)

        ud_upsample = DiagonalUpsample(orientation=UP_DIAGONAL, in_channels=1)
        up_diagonal_upsampled = ud_upsample(up_diagonal)

        dd_upsample = DiagonalUpsample(orientation=DOWN_DIAGONAL, in_channels=1)
        down_diagonal_upsampled = dd_upsample(down_diagonal)

        upsampled = up_diagonal_upsampled + down_diagonal_upsampled

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

        ud_upsample = DiagonalUpsample(orientation=UP_DIAGONAL, in_channels=n_channels)
        up_diagonal_upsampled = ud_upsample(up_diagonal)

        dd_upsample = DiagonalUpsample(orientation=DOWN_DIAGONAL, in_channels=n_channels)
        down_diagonal_upsampled = dd_upsample(down_diagonal)
        upsampled = up_diagonal_upsampled + down_diagonal_upsampled

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

    def test_single_tensor(self):
        """
        Test DiagonalUpsample with a single tensor instead of a tuple of two tensors
        Returns:

        """
        up_diagonal = torch.FloatTensor([[1, 3],
                                         [5, 7]])
        up_diagonal.unsqueeze_(0).unsqueeze_(0)
        # upsample in UP_DIAGONAL orientation
        diagonal_upsample = DiagonalUpsample(orientation=UP_DIAGONAL, in_channels=1)
        upsampled = diagonal_upsample(up_diagonal)

        expected_upsampled = torch.FloatTensor([[0, 1, 0, 3],
                                                [1, 0, 3, 0],
                                                [0, 5, 0, 7],
                                                [5, 0, 7, 0]]).unsqueeze_(dim=0).unsqueeze_(dim=0)

        self.assertTrue(torch.all(expected_upsampled == upsampled))

        down_diagonal = torch.FloatTensor([[1, 3],
                                           [5, 7]])
        down_diagonal.unsqueeze_(0).unsqueeze_(0)
        # Upsample in DOWN_DIAGONAL orientation
        diagonal_upsample = DiagonalUpsample(orientation=DOWN_DIAGONAL, in_channels=1)
        upsampled = diagonal_upsample(down_diagonal)

        expected_upsampled = torch.FloatTensor([[1, 0, 3, 0],
                                                [0, 1, 0, 3],
                                                [5, 0, 7, 0],
                                                [0, 5, 0, 7]]).unsqueeze_(dim=0).unsqueeze_(dim=0)
        self.assertTrue(torch.all(expected_upsampled == upsampled))

    def test_recursive(self):
        """
        Test DiagonalUpsample recursively with alternating orientations
        Returns:

        """
        up_diagonal = torch.FloatTensor([[1]])
        up_diagonal.unsqueeze_(0).unsqueeze_(0)
        # upsample in UP_DIAGONAL orientation
        ud_upsample = DiagonalUpsample(orientation=UP_DIAGONAL, in_channels=1)
        upsampled = ud_upsample(up_diagonal)

        expected_upsampled = torch.FloatTensor([[0, 1],
                                                [1, 0]]).unsqueeze_(dim=0).unsqueeze_(dim=0)

        self.assertTrue(torch.all(expected_upsampled == upsampled))

        # upsample the upsampled tensor once again, but this time in down diagonal orientation
        dd_upsample = DiagonalUpsample(orientation=DOWN_DIAGONAL, in_channels=1)
        upsampled = dd_upsample(upsampled)

        expected_upsampled = torch.FloatTensor([[0, 0, 1, 0],
                                                [0, 0, 0, 1],
                                                [1, 0, 0, 0],
                                                [0, 1, 0, 0]]).unsqueeze_(dim=0).unsqueeze_(dim=0)

        self.assertTrue(torch.all(expected_upsampled == upsampled))


if __name__ == '__main__':
    unittest.main()
