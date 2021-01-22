import torch
import haar
import torch.nn.functional as F
from onda.modules.upsample import DiagonalUpsample


class HaarDeconv2D(torch.nn.Module):
    """
    2D Deconvolution using low pass and detail components as inputs
    Reconstructs a tensor of original dimensions prior to decomposition
    """

    def __init__(self, orientation, in_channels=3):
        """
        HaarDeconv2D constructor: Takes a tensor of dimension <Samples, Channels, Width, Height>
        width and height dimensions must be even

        Args:

            orientation (int): the direction of the Haar kernel: one of
                haar.HORIZONTAL, haar.VERTICAL, haar.UP_DIAGONAL or haar.DOWN_DIAGONAL

            in_channels (int): the number of channels in the tensor, i.e. the size of dimenson
                index 1
        """
        super(HaarDeconv2D, self).__init__()

        kernels = haar.DECONV_KERNELS[orientation]
        self._orientation = orientation

        if self._orientation == haar.VERTICAL:
            self._stride = (2, 1)

        elif self._orientation == haar.HORIZONTAL:
            self._stride = (1, 2)

        elif self._orientation in (haar.UP_DIAGONAL, haar.DOWN_DIAGONAL):
            self._stride = (2, 2)

        else:
            raise ValueError('Unknown orientation: {}'.format(self._orientation))

        self._in_channels = in_channels
        self._out_channels = self._in_channels
        self._groups = self._in_channels
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
                x tuple(torch.tensor): a tuple of tensors with dimensions
                    [batch_size, channels, width, height]
                    the tuple contains the low pass and detail tensors originating from a Haar
                    Wavelet decomposition, in that order
                    both tensors must have the same shape

            Returns:
                (torch.tensor): the reconstructed tensor from the low pass and detail decomposition
                    tensors.  The tensor will have dimension size twice the size of the low pass
                    and detail tensors in the direction of decomposition, e.g. height will be twice
                    the decomposed tensors if orientation == haar.VERTICAL
        """

        low_pass = x[haar.LOW_PASS]
        detail = x[haar.DETAIL]

        assert low_pass.shape == detail.shape, \
            'Low Pass tensor shape: {} != {} Detail tensor shape'\
            .format(low_pass.shape, detail.shape)

        if self._orientation == haar.VERTICAL:
            # check that vertical dimension is half the size of the horizontal dimension
            assert low_pass.shape[2] == low_pass.shape[3] // 2, \
                'Expected dimension index {} size: {} != {} actual size'\
                .format(2, low_pass.shape[3] // 2, low_pass.shape[2])

        elif self._orientation == haar.HORIZONTAL:
            # check that horizontal dimension is half the size of the vertical dimension
            assert low_pass.shape[3] == low_pass.shape[2] // 2, \
                'Expected dimension index {} size: {} != {} actual size'\
                .format(3, low_pass.shape[2] // 2, low_pass.shape[3])

        elif self._orientation == haar.UP_DIAGONAL:
            # check both dimensions are equal
            assert low_pass.shape[2] == low_pass.shape[3], \
                'Expected height and width should be equal but dimension {} has size {} ' \
                'while dimension {} has size {}'\
                .format(2, low_pass.shape[2], 3, low_pass.shape[3])

        elif self._orientation == haar.DOWN_DIAGONAL:
            # check both dimensions are equal
            assert low_pass.shape[2] == low_pass.shape[3], \
                'Expected height and width should be equal but dimension {} has size {} ' \
                'while dimension {} has size {}' \
                .format(2, low_pass.shape[2], 3, low_pass.shape[3])

        lp1 = F.conv_transpose2d(low_pass,
                                 weight=self._low_pass_kernel,
                                 stride=self._stride,
                                 padding=0,
                                 groups=self._groups)

        d1 = F.conv_transpose2d(detail,
                                weight=self._low_pass_kernel,
                                stride=self._stride,
                                padding=0,
                                groups=self._groups)

        lp2 = F.conv_transpose2d(low_pass,
                                 weight=self._detail_kernel,
                                 stride=self._stride,
                                 padding=0,
                                 groups=self._groups)

        d2 = F.conv_transpose2d(detail,
                                weight=self._detail_kernel,
                                stride=self._stride,
                                padding=0,
                                groups=self._groups)
        deconv = lp1 + d1
        deconv += lp2 - d2

        return deconv
