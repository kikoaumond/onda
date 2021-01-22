import torch
import torch.nn.functional as F
from onda.modules.wavelets.haar import UP_DIAGONAL, DOWN_DIAGONAL


class DiagonalUpsample(torch.nn.Module):
    """
    Upsamples a Haar Diagonal decomposition of tensor (either a low pass or a detail component)
     along a Diagonal direction resulting in an upsampled tensor with width and height equal to twice
    the width and height of the original decomposition tensors

    If the decomposition tensors is
    [[x00, x01],
     [x10, x11]]

     and the direction is UP_DIAGONAL

    DiagonalUpsample will upsample it to

    [[  0, x00,   0, x01],
     [x00,   0, x01,   0],
     [  0, x10,   0, x11],
     [x10,   0, x11,   0]]

     If the direction is DOWN_DIAGONAL, DiagonalUpSample will upsample it to

    [[x00,   0, x01,   0],
     [  0, x00,   0, x01],
     [x10,   0, x10,   0],
     [  0, x10,   0, x10]]

    """

    def __init__(self, orientation, in_channels=3):
        """

        Args:
            orientation (int) one of
                (onda.modules.wavelets.haar.UP_DIAGONAL, onda.modules.wavelets.haar.DOWN_DIAGONAL),
                denoting the direction along which to upsample the tensor

            in_channels (int):  the number of channels in the tensor.  in_channels must
            be equal to the size of dimension of index 1 in the tensors submitted to forward()
        """
        super(DiagonalUpsample, self).__init__()

        assert orientation in (UP_DIAGONAL, DOWN_DIAGONAL), \
            'Illegal direction value: {}.  direction must be one of ' \
            '(onda.modules.wavelets.haar.UP_DIAGONAL, onda.modules.wavelets.haar.DOWN_DIAGONAL)'\
            .format(orientation)

        self._orientation = orientation
        self._in_channels = in_channels

        if self._orientation == UP_DIAGONAL:
            self._kernels = [torch.Tensor([[0, 0],
                                           [1, 0]]).unsqueeze_(dim=0).unsqueeze_(dim=0),
                             torch.Tensor([[0, 1],
                                           [0, 0]]).unsqueeze_(dim=0).unsqueeze_(dim=0)]

        elif self._orientation == DOWN_DIAGONAL:
            self._kernels = [torch.Tensor([[1, 0],
                                           [0, 0]]).unsqueeze_(dim=0).unsqueeze_(dim=0),
                             torch.Tensor([[0, 0],
                                           [0, 1]]).unsqueeze_(dim=0).unsqueeze_(dim=0)]

        if self._in_channels > 1:
            self._kernels = [k.repeat(self._in_channels, 1, 1, 1)
                             for k in self._kernels]

    def forward(self, x):
        """
        Upsample a tensor corresponding in a Diagonal direction

        Args:
            x(torch.Tensor): a tensor with dimensions (batch size, channels, height, width)

        Returns:
            (torch.Tensor): the resulting upsampled tensor with dimensions
                (batch size, channels, 2 * height, 2 * width)

        """
        # check that the number of channels matches what was specified in the constructor
        channels = x.shape[1]

        assert channels == self._in_channels, \
            'DiagonalUpSample was instantiated with in_channels = {} but tensor has {} channels'\
            .format(self._in_channels, channels)

        diagonal_upsampled = None

        for kernel in self._kernels:

            if diagonal_upsampled is None:
                diagonal_upsampled = F.conv_transpose2d(x,
                                                        weight=kernel,
                                                        stride=(2, 2),
                                                        padding=0,
                                                        groups=channels)
            else:
                diagonal_upsampled += F.conv_transpose2d(x,
                                                         weight=kernel,
                                                         stride=(2, 2),
                                                         padding=0,
                                                         groups=channels)

        return diagonal_upsampled
