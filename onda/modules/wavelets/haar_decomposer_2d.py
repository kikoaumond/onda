import torch
import math
from .haar import VERTICAL_HORIZONTAL, DIAGONAL, UP_DIAGONAL, DOWN_DIAGONAL, HORIZONTAL, VERTICAL
from .haar_conv_2d import HaarConv2D
from .haar_deconv_2d import HaarDeconv2D
from onda.modules.feature import Sparsifier


class HaarDecomposer2D(torch.nn.Module):
    """
    Decomposes a batch of tensor into its Haar components, performing sparsification if necessary
    No padding or trimming of tensors is conducted
    """

    def __init__(self,
                 input_shape,
                 dim=(2, 3),
                 orientation=DIAGONAL,
                 by_channel=False,
                 levels=None,
                 sparsity=None):
        """

        Args:
            input_shape (tuple[int]): the shape of the input tensor.  Shape is of the form
            (batch_size, number of channels, width, height)

            dim (tuple[int]): a tuple with the indices of the two dimensions to which the Haar 2D
                transform will be applied
                Default: (2, 3), which would be the height and width in a tensor with dimensions
                (batch size, channels, height, width)

            orientation (int): one of haar.VERTICAL_HORIZONTAL or haar.DIAGONAL, denoting
                the orientation of the Haar transform

            by_channel (bool):  Whether the decomposition will be performed on individual channels
                (by_channel = True) in which case the decompositions will be returned for each
                invidual channel at each level.  If False, a single decomposition for all channels
                will be returned at each level.
                Default: False

            levels (int): the number of levels for decomposition.  If  not supplied, will attempt
             decomposition of the tensor until either the width or height are not even.

            sparsity(float): If specified, will sparsify the Haar Low Pass and Detail coefficient
                tensors by magnitude; if sparsity = 0.95, only the top 5% coefficients in absolute
                value will be maintained, all others will be set to 0
        """
        super(HaarDecomposer2D, self).__init__()

        self._input_shape = input_shape

        assert len(dim) == 2, 'Two dimensions must be supplied to HaarDecomposer'

        assert self._input_shape[dim[0]] % 2 == 0, \
            'Dimensions to which HaarDecomposer will be applied must have even size. ' \
            'Dimension {} has size {}'.format(dim[0], self._input_shape[dim[0]])

        assert self._input_shape[dim[1]] % 2 == 0, \
            'Dimensions to which HaarDecomposer will be applied must have even size. ' \
            'Dimension {} has size {}'.format(dim[1], self._input_shape[dim[1]])

        assert max(dim) < len(self._input_shape), \
            'Dimension {} was specified for HaarDecomposer but input shape has only {} dimensions'\
            .format(max(dim), len(self._input_shape))

        assert sparsity is None or 0 <= sparsity < 1, 'sparsity must be a float between 0 and 1'

        assert orientation in (VERTICAL_HORIZONTAL, DIAGONAL), 'orientation must be one of '

        assert orientation in (VERTICAL_HORIZONTAL, DIAGONAL), \
            'unknown orientation: {}'.format(orientation)

        self._dim = dim
        # the transpositions required to move the two dimensions where HaarDecomposer will operate
        # so they are the last two dimensions
        self._transpositions = []

        if self._dim[0] != len(self._input_shape) - 2:
            self._transpositions.append((self._dim[0], len(self._input_shape) - 2))

        if self._dim[1] != len(self._input_shape) - 1:
            self._transpositions.append((self._dim[1], len(self._input_shape) - 1))

        self._orientation = orientation

        self._by_channel = by_channel

        self._sparsity = sparsity
        self._sparsifier = None

        if self._sparsity is not None:
            self._sparsifier = Sparsifier(self._sparsity, self._dim)

        self._channels = self._input_shape[1]

        if self._orientation == DIAGONAL:
            self._haar_up_diagonal = HaarConv2D(orientation=UP_DIAGONAL,
                                                in_channels=self._channels,
                                                by_channel=self._by_channels)
            self._haar_down_diagonal = HaarConv2D(orientation=DOWN_DIAGONAL,
                                                  in_channels=self._channels,
                                                  by_channel=self._by_channel)

            deconv_in_channels = self._channels if self._by_channel else 1

            self._deconv_up_diagonal = HaarDeconv2D(orientation=UP_DIAGONAL,
                                                    in_channels=deconv_in_channels)

            self._deconv_down_diagonal = HaarDeconv2D(orientation=DOWN_DIAGONAL,
                                                      in_channels=deconv_in_channels)

        # compute the maximum number of levels
        max_levels = min(math.floor(math.log2(self._input_shape[self._dim[0]])),
                         math.floor(math.log2(self._input_shape[self._dim[1]])))

        assert levels is None or max_levels <= levels, \
            'levels was specified as {} but maximum possible number of levels is {}'\
            .format(levels, max_levels)

        self._levels = max_levels if levels is None else levels

    def forward(self, x):
        """
        Decompose a Tensor x into its Haar components

        Args:
            x (torch.tensor):

        Returns:
            (tensor): a tensor containing the decomposed tensor.  The tensor will have dimensions
            (batch size, channels x levels, height//2, width //2) if by_channel is True or
            (batch size, levels, height // 2, width // 2) if by_channel is False
        """

        assert x.shape == self._input_shape, \
            'input shape was specified as {} but shape of tensor is {}'\
            .format(self._input_shape, x.shape)

        for transposition in self._transpositions:
            x = torch.transpose(x, **transposition)

        height_levels = math.floor(math.log2(x.shape[-2]))
        assert height_levels >= self._levels, \
            'height of {} does not support {} x levels.  ' \
            'Increase image height or decrease number of levels'\
            .format(height_levels, self._levels)

        width_levels = math.floor(math.log2(x.shape[-1]))
        assert width_levels >= self._levels, \
            'width of {} does not support {} x levels.  ' \
            'Increase image width or decrease number of levels' \
            .format(width_levels, self._levels)

        assert x.shape[-2] == self._input_shape[self._dim[0]] \
            and x.shape[-1] == self._input_shape[self._dim[-1]], \
            'Transposed shape should be height, width = {} but it id height, width = {} instead'\
            .format(self._input_shape[self._dim], x.shape[-2:])

        if self._orientation == DIAGONAL:
            haar_x = self._decompose_diagonal(x)

        else:
            raise NotImplemented

    def _decompose_diagonal(self, x):
        """
        Decompose tensor x in diagonal directions

        Args:
            x (torch.Tensor): a tensor to be decomposed

        Returns:
            (torch.Tensor): the decomposed tensor
        """


        # First apply the diagonal Haar Wavelet transformations
        x_up_diagonal_low_pass, x_up_diagonal_detail = self._haar_up_diagonal(x)
        x_down_diagonal_low_pass, x_down_diagonal_detail = self._haar_down_diagonal(x)

        # reconstruct original tensor, which may have been sparsified above
        x_up_diagonal = self._deconv_up_diagonal(x_up_diagonal_low_pass,
                                                 x_up_diagonal_detail)
        x_down_diagonal = self._deconv_down_diagonal(x_down_diagonal_low_pass,
                                                    x_down_diagonal_detail)

        # reconstruct tensor at original scale
        x_reconstructed = x_up_diagonal + x_down_diagonal






