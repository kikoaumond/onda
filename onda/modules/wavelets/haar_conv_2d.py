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
        assert orientation in (haar.HORIZONTAL,
                               haar.VERTICAL,
                               haar.UP_DIAGONAL,
                               haar.DOWN_DIAGONAL), \
            'orientation must be one of \n ' \
            '(onda.modules.wavelets.haar.HORIZONTAL,\n' \
            ' onda.modules.wavelets.haar.VERTICAL,\n' \
            ' onda.modules.wavelets.haar.UP_DIAGONAL,\n' \
            ' onda.modules.wavelets.haar.DOWN_DIAGONAL)'

        super(HaarConv2D, self).__init__()

        self._orientation = orientation
        self._in_channels = in_channels
        self._by_channel = by_channel

        if self._by_channel:
            self._groups = self._in_channels
            self._out_channels = self._in_channels

        else:
            self._groups = 1
            self._out_channels = 1

        # These fields are populated by _get_kernels
        self._kernel_dim = None
        self._low_pass_kernel = None
        self._detail_kernel = None

        self._get_kernels(self._in_channels)

    def _get_kernels(self, in_channels):
        """
        Get the kernels with the correct shape given the number of input channels
        Args:
            in_channels (int): the number of channels in the input tensor, corresponding to the size
                of dimension of index 0 in the tensor

        Returns:
                Tuple[torch.Tensor] a tuple with the two tensors corresponding to the low pass
                and detail convolutions, in this order
        """
        # get the kernels given the number of input channels
        kernels = haar.KERNELS[self._orientation]

        self._kernel_dim = in_channels // self._groups
        self._low_pass_kernel = kernels[haar.LOW_PASS].unsqueeze(0).unsqueeze(0) \
            .repeat(self._out_channels, self._kernel_dim, 1, 1)
        self._detail_kernel = kernels[haar.DETAIL].unsqueeze(0).unsqueeze(0) \
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
        assert isinstance(x, torch.Tensor), 'HaarConv2D accepts a single Tensor.  A {} was passed'\
            .format(x.__class__.name)

        in_channels = x.shape[1]
        # if the number of channels does not match what was used in the constructor, recmpute
        # the kernels
        if in_channels != self._in_channels:
            self._get_kernels(in_channels)

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

        # average channel outputs if by_channel = False
        if not self._by_channel:
            low_pass /= self._in_channels
            detail /= self._in_channels

        return (low_pass, detail)

