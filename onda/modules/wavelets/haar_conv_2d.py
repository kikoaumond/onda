import torch
import haar
import torch.nn.functional as F


class HaarConv2D(torch.nn.Module):
    """
    2D Convolution by Haar wavelet functions;
    Applies vertical, horizontal and diagonal Haar kernels to 2D tensor
    """

    def __init__(self, orientation, in_channels=3, by_channel=True):
        """
        Haar Conv2D constructor: Takes a tensor of dimension <Samples, Channels, Width, Height>
        width and height dimensions must be even

        Args:

            orientation (int): the direction of the Haar kernel: one of
                haar.HORIZONTAL, haar.VERTICAL, haar.UP_DIAGONAL or haar.DOWN_DIAGONAL

            by_channel (bool): if True, apply Haar wavelet individually to each channel, resulting
                in an output tensor with the same number of channels as the input tensor
        """
        super(HaarConv2D, self).__init__()

        kernels = haar.KERNELS[orientation]
        self._orientation = orientation
        self._in_channels = in_channels
        self._by_channel = by_channel

        if self._by_channel:
            self._groups = self._in_channels
            self._out_channels = self._in_channels

        else:
            self._groups = 1
            self._out_channels = 1

        self._kernel_dim = self._in_channels // self._groups
        self._low_pass_kernel = kernels[haar.LOW_PASS].unsqueeze(0).unsqueeze(0)\
            .repeat(self._out_channels, self._kernel_dim, 1, 1)
        self._detail_kernel = kernels[haar.DETAIL].unsqueeze(0).unsqueeze(0)\
            .repeat(self._out_channels, self._kernel_dim, 1, 1)

    def forward(self, x):
        """
            Convolve tensor x with the appropriate low pass and detail Haar kernels and
            returns the two tensors resulting from the convolution

            Args:
                x (torch.tensor): a tensor with dimensions [batch_size, channels, width, height]
                    tensor must have an even dimension size for the relevant dimension
                    (height or width, depending on the orientation of the convolution)
        """
        if self._orientation == haar.HORIZONTAL:
            dim_idx = (3,)
            stride = (1, 2)
        elif self._orientation == haar.VERTICAL:
            dim_idx = (2,)
            stride = (2, 1)
        elif self._orientation == haar.UP_DIAGONAL or self._orientation == haar.DOWN_DIAGONAL:
            dim_idx = (2, 3)
            stride = (2, 2)

        for di in dim_idx:

            assert x.shape[di] % 2 == 0, \
                'size of dimension {} of input tensor ({}) must be even'.format(di, x.shape[di])

        low_pass = F.conv2d(x,
                            weight=self._low_pass_kernel,
                            stride=stride,
                            padding=0,
                            groups=self._groups)
        detail = F.conv2d(x,
                          weight=self._detail_kernel,
                          stride=stride,
                          padding=0,
                          groups=self._groups)

        return (low_pass, detail)

