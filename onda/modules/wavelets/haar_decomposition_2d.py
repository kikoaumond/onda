import torch
import math
from .haar import VERTICAL_HORIZONTAL, DIAGONAL, UP_DIAGONAL, DOWN_DIAGONAL, \
    UP_DIAGONAL_LOW_PASS_DECONV_KERNEL, DOWN_DIAGONAL_LOW_PASS_DECONV_KERNEL
from .haar_conv_2d import HaarConv2D
from .haar_deconv_2d import HaarDeconv2D
from onda.modules.upsample import DiagonalUpsample
from onda.modules.feature import Sparsifier
import torch.nn.functional as F


class HaarDecomposition2D(torch.nn.Module):
    """
    Decomposes a batch of tensor into its Haar components, performing sparsification if necessary
    No padding or trimming of tensors is conducted
    The low pass and detail components are upsampled to the original tensor's original width and
    height and the decomposition is returned as single tensor with the second dimension containing
    the components in the order specified below.  The detail components at all levels are stored
    in the decomposition tensor and the last element is the low pass tensor at level N.  Low pass
    components at each level can be reconstructed from the low pass and detail components in the
    level below, all the way up to the original tensor.

    [Detail Level 1,
     Detail Level 2,
     ...
     Detail Level N,
     Low Pass Level N]

    where increasing levels mean coarser levels of resolution
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
                individual channel at each level.  If False, a single decomposition for all channels
                will be returned at each level.

                If by_channel is True, the decomposition will have a number of channels equal to
                 <levels> * <number of original tensor channels>

                If by_channel is False, the decomposition will have a number of channels equal to
                <levels>

                Default: False

            levels (int): the number of levels for decomposition.  If  not supplied, will attempt
             decomposition of the tensor until either the width or height are not even.

            sparsity(float): If specified, will sparsify the Haar Low Pass and Detail coefficient
                tensors by magnitude; if sparsity = 0.95, only the top 5% coefficients in absolute
                value will be maintained, all others will be set to 0
        """
        super(HaarDecomposition2D, self).__init__()

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
            # sparsification is conducted across all channels of each tensor in the batch
            # hence we specify the channel, height and width dimensions to the sparsifier
            # we keep the batch dimension out of the sparsification as we want to sparsify
            # individual tensors (image) in the batch separately
            dims_to_sparsify = [d for d in range(1, len(self._input_shape))]
            self._sparsifier = Sparsifier(self._sparsity, dim=dims_to_sparsify)

        self._in_channels = self._input_shape[1]

        if self._orientation == DIAGONAL:

            self._haar_up_diagonal = HaarConv2D(orientation=UP_DIAGONAL,
                                                in_channels=self._in_channels,
                                                by_channel=self._by_channel)
            self._haar_down_diagonal = HaarConv2D(orientation=DOWN_DIAGONAL,
                                                  in_channels=self._in_channels,
                                                  by_channel=self._by_channel)

            upsample_in_channels = self._in_channels if self._by_channel else 1

            self._up_diagonal_upsample = DiagonalUpsample(orientation=UP_DIAGONAL,
                                                          in_channels=upsample_in_channels)

            self._down_diagonal_upsample = DiagonalUpsample(orientation=DOWN_DIAGONAL,
                                                            in_channels=upsample_in_channels)

        # compute the maximum number of levels
        max_levels = math.floor(math.log2(min(self._input_shape[self._dim[0]],
                                              self._input_shape[self._dim[1]])))

        assert levels is None or max_levels <= levels, \
            'levels was specified as {} but maximum possible number of levels for shape {} is {}. '\
            'Increase input shape width/height or decrease number of levels'\
            .format(levels, self._input_shape[self._dim], max_levels)

        self._levels = max_levels if levels is None else levels

    def forward(self, x):
        """
        Decompose a Tensor x into its Haar components

        The low pass and detail components are upsampled to the original tensor's original width and
        height and the decomposition is returned as single tensor with the second dimension
        containing the componenets in the order specified below

        [Low Pass Level 1,
         Detail Level 1,
         Low Pass Level 2,
         Detail Level 2,
         ...
         Low Pass Level N,
         Detail Level N]

        where increasing levels mean coarser levels of resolution

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

        assert x.shape[-2] == self._input_shape[self._dim[0]] \
            and x.shape[-1] == self._input_shape[self._dim[-1]], \
            'Transposed shape should be height, width = {} but it is height, width = {} instead'\
            .format(self._input_shape[self._dim], x.shape[-2:])

        if self._orientation == DIAGONAL:
            decomposition = self._decompose_diagonal(x)

        else:
            raise NotImplemented

        return decomposition

    def decompose_diagonal_one_level(self, x):
        ud, _ = self._haar_up_diagonal(x)

        ud_ud_low_pass, ud_ud_detail = self._haar_up_diagonal(ud)

        ud_ud_low_pass = self._up_diagonal_upsample(ud_ud_low_pass)
        ud_ud_low_pass = self._up_diagonal_upsample(ud_ud_low_pass)

        ud_ud_detail = self._up_diagonal_upsample(ud_ud_detail)
        ud_ud_detail = self._up_diagonal_upsample(ud_ud_detail)

        dd_ud_low_pass, dd_ud_detail = self._haar_down_diagonal(ud)

        dd_ud_low_pass = self._down_diagonal_upsample(dd_ud_low_pass)
        dd_ud_low_pass = self._up_diagonal_upsample(dd_ud_low_pass)

        dd_ud_detail = self._down_diagonal_upsample(dd_ud_detail)
        dd_ud_detail = self._up_diagonal_upsample(dd_ud_detail)

        dd, _ = self._haar_down_diagonal(x)

        dd_dd_low_pass, dd_dd_detail = self._haar_down_diagonal(dd)

        dd_dd_low_pass = self._down_diagonal_upsample(dd_dd_low_pass)
        dd_dd_low_pass = self._down_diagonal_upsample(dd_dd_low_pass)

        dd_dd_detail = self._down_diagonal_upsample(dd_dd_detail)
        dd_dd_detail = self._down_diagonal_upsample(dd_dd_detail)

        ud_dd_low_pass, ud_dd_detail = self._haar_up_diagonal(dd)

        ud_dd_low_pass = self._up_diagonal_upsample(ud_dd_low_pass)
        ud_dd_low_pass = self._down_diagonal_upsample(ud_dd_low_pass)

        ud_dd_detail = self._up_diagonal_upsample(ud_dd_detail)
        ud_dd_detail = self._down_diagonal_upsample(ud_dd_detail)

        ud_low_pass = ud_ud_low_pass + dd_ud_low_pass
        dd_low_pass = dd_dd_low_pass + ud_dd_low_pass

        low_pass = ud_low_pass + dd_low_pass

        ud_detail = ud_ud_detail + dd_ud_detail
        dd_detail = dd_dd_detail + ud_dd_detail

        detail = ud_detail + dd_detail

        return low_pass, detail

    def _decompose_diagonal(self, x):
        """
        Perform multiscale decomposition of tensor X using Haar wavelets generating
        a sequence of Low Pass and Detail tensors at multiple resolution levels.
        All Low Pass and Detail tensors are up-sampled so the final tensors have the same
        width and height dimensions as the original tensor

        Args:
            x (torch.Tensor): a tensor to be decomposed

        Returns:
            (torch.Tensor): the decomposed tensor
        """

        channels = x.shape[1]

        assert channels == self._in_channels, \
            'HaarDecomposition2D was instantiated to expect {} channels ' \
            'but input tensor has {} channels'.format(self._in_channels, channels)

        # Instantiate the tensor that will contain the low pass and detail tensors at each level
        # The tensors will have the same height and width as the original tensor
        # If by_channel = False,  there will be one detail component at each level and a low pass
        # component at the lowest level, in a total of <levels + 1> channels.  The resulting
        # decomposition tensor will have dimensions
        # (batch size, number of levels + 1, height, width)
        # If by_channel = True,  there will be one detail component per channel at each level
        # and one detail component per channel at the lowest level in a total of
        # <(levels + 1) * channels>
        # Therefore the resulting tensor will have dimensions
        # (batch size, (number of levels  + 1) * number of input channels, height, width)

        batch_size = x.shape[0]
        height, width = x.shape[2:]
        n_components = self._levels + 1

        if self._by_channel:
            n_components *= self._in_channels

        decomposition = torch.zeros(size=(batch_size,
                                          n_components,
                                          height,
                                          width))

        level = 1
        t = x

        while level <= self._levels:

            if level == 1:
                ud_low_pass, ud_detail = self._haar_up_diagonal(t)
                ud_low_pass = self._up_diagonal_upsample(ud_low_pass)
                ud_detail = self._up_diagonal_upsample(ud_detail)

                dd_low_pass, dd_detail = self._haar_down_diagonal(t)
                dd_low_pass = self._down_diagonal_upsample(dd_low_pass)
                dd_detail = self._down_diagonal_upsample(dd_detail)

                low_pass = ud_low_pass + dd_low_pass
                detail = ud_detail + dd_detail

            else:
                low_pass, detail = self.decompose_diagonal_one_level(t)

            t = low_pass  # for the next decomposition iteration

            n_out_channels = self._in_channels if self._by_channel else 1
            component_index = (level - 1) * n_out_channels

            if level == self._levels:
                decomposition[:,
                              component_index + n_out_channels: component_index + 2 * n_out_channels,
                              :,
                              :] = low_pass

            decomposition[:,
                          component_index: component_index + n_out_channels,
                          :,
                          :] = detail
            level += 1

        # sparsify tensor if specified
        if self._sparsifier is not None:

            # Sparsify each individual tensor in the batch separately
            for batch_index in range(decomposition.shape[0]):
                # sparsifier requires 4 dimensions so we add a singleton dimension to create
                #  a "batch" of a single image
                image = decomposition[batch_index, ...].unsqueeze_(dim=0)
                sparsified_image = self._sparsifier(image)
                decomposition[batch_index, ...] = sparsified_image

        return decomposition

    @classmethod
    def reconstruct(cls, decomposition, orientation=DIAGONAL, in_channels=1):
        """
        Given a Haar decomposition of a tensor as produced by HaarDecomposition2D (where the
        decomposition components have been upsampled up to the original tensor's width and height),
        reconstruct the original tensor.

        Note that if the original tensor had multiple channels and the decomposition was performed
        by aggregating channels, the reconstructed tensor will be equal to the original tensor
        averaged across channels.

        Also, if the decomposition has been sparsified, the reconstructed tensor will not match the
        original tensor

        Args:
            decomposition (torch.Tensor): the Haar decomposition of a tensor as produced by
                Haardecomposition2D

            orientation (int):  the orientation used to produce the decomposition.
                default: DIAGONAL

            in_channels (int): the number of channels at each level of the decomposition.  For
                instance, if the original tensor had 3 channels AND the decomposition was performed
                with by_channel = True, then in_channels = 3.  Conversely, if the original tensor
                had a single channel OR the decomposition was performed with by_channel = False then
                in_channels = 1

                Default: 1

        Returns:
            (torch.Tensor):  the reconstructed tensor
        """

        if orientation == DIAGONAL:
            reconstructed = cls.reconstruct_diagonal(decomposition, in_channels=in_channels)

            return reconstructed

        else:
            raise NotImplemented

    @classmethod
    def reconstruct_diagonal(cls, decomposition, in_channels=1):
        """
        Given a Haar Diagonal decomposition of a tensor as produced by HaarDecomposition2D
        (where the decomposition components have been upsampled up to the original tensor's width
        and height),reconstruct the original tensor.

        Note that if the original tensor had multiple channels and the decomposition was performed
        by aggregating channels, the reconstructed tensor will be equal to the original tensor
        averaged across channels.

        Also, if the decomposition has been sparsified, the reconstructed tensor will not match the
        original tensor.

        Args:
            decomposition (torch.Tensor): the Haar decomposition of a tensor as produced by
                Haardecomposition2D

            in_channels (int): the number of channels at each level of the decomposition.  For
                instance, if the original tensor had 3 channels AND the decomposition was performed
                with by_channel = True, then in_channels = 3.  Conversely, if the original tensor
                had a single channel OR the decomposition was performed with by_channel = False then
                in_channels = 1

                Default: 1

        Returns:
            (torch.Tensor):  the reconstructed tensor
        """
        assert len(decomposition.shape) == 4, \
            'decomposition must have 4 dimensions (batch, channel, height, width)'

        channels = decomposition.shape[1]

        assert channels % in_channels == 0, \
            'The number of channels in the decomposition tensor ({}) ' \
            'must be divisible by in_channels ({})'.format(channels, in_channels)

        n_levels = ((decomposition.shape[1]) // in_channels) - 1

        up_diagonal_conv = HaarConv2D(orientation=UP_DIAGONAL,
                                      in_channels=in_channels,
                                      by_channel=True)

        down_diagonal_conv = HaarConv2D(orientation=DOWN_DIAGONAL,
                                        in_channels=in_channels,
                                        by_channel=True)

        up_diagonal_deconv = HaarDeconv2D(orientation=UP_DIAGONAL, in_channels=in_channels)
        down_diagonal_deconv = HaarDeconv2D(orientation=DOWN_DIAGONAL, in_channels=in_channels)

        up_diagonal_upsample = DiagonalUpsample(orientation=UP_DIAGONAL, in_channels=in_channels)
        down_diagonal_upsample = DiagonalUpsample(orientation=DOWN_DIAGONAL, in_channels=in_channels)

        # start with the low pass tensor at the lowest granularity level
        low_pass = decomposition[:, -in_channels:, :, :]

        # starting from the lowest levels reconstruct the low pass tensor
        # at the immediately above level all the way up to the original tensor
        for level in reversed(range(1, n_levels + 1)):
            # get the detail tensor at this level
            detail_index = (level - 1) * in_channels
            detail = decomposition[:, detail_index: detail_index + in_channels, :, :]

            if level == 1:
                ud_d, _ = up_diagonal_conv(detail)
                dd_d, _ = down_diagonal_conv(detail)

                ud_reconstructed = up_diagonal_deconv((ud_lp, ud_d))
                dd_reconstructed = down_diagonal_deconv((dd_lp, dd_d))

                reconstructed = ud_reconstructed + dd_reconstructed

                return reconstructed

            # compute the detail tensors at this level
            ud_detail, _ = up_diagonal_conv(detail)
            ud_ud_detail, _ = up_diagonal_conv(ud_detail)
            dd_ud_detail, _ = down_diagonal_conv(ud_detail)

            dd_detail, _ = down_diagonal_conv(detail)
            dd_dd_detail, _ = down_diagonal_conv(dd_detail)
            ud_dd_detail, _ = up_diagonal_conv(dd_detail)

            # now reduce and reconstruct
            # First the Up Diagonal branch
            ud_low_pass, _ = up_diagonal_conv(low_pass)
            ud_ud_low_pass, _ = up_diagonal_conv(ud_low_pass)
            dd_ud_low_pass, _ = down_diagonal_conv(ud_low_pass)

            ud_ud_lp = up_diagonal_deconv((ud_ud_low_pass, ud_ud_detail))
            dd_ud_lp = down_diagonal_deconv((dd_ud_low_pass, dd_ud_detail))
            ud_lp = ud_ud_lp + dd_ud_lp

            # Then the Down Diagonal branch
            dd_low_pass, _ = down_diagonal_conv(low_pass)
            dd_dd_low_pass, _ = down_diagonal_conv(dd_low_pass)
            ud_dd_low_pass, _ = up_diagonal_conv(dd_low_pass)

            dd_dd_lp = down_diagonal_deconv((dd_dd_low_pass, dd_dd_detail))
            ud_dd_lp = up_diagonal_deconv((ud_dd_low_pass, ud_dd_detail))
            dd_lp = dd_dd_lp + ud_dd_lp

            if level > 2:
                ud_lp = up_diagonal_upsample(ud_lp)
                dd_lp = down_diagonal_upsample(dd_lp)
                low_pass = ud_lp + dd_lp

    @classmethod
    def reduce_diagonal_one_level(cls, x):
        """
        Reduce an upsampled Haar component tensor in a Haar Decomposition tensor which has the
        same width and height as the original tensor to down on level where the reduced tensor
        has half the width and height of the inpyt tensor

        If the upsampled tensor looks like

        [[a b c d]
         [b a d c]
         [e f g h]
         [f e h g]]

         The reduced tensors will be

         [[a c]   and  [[b d]
          [e g]]        [f h]]

        Args:
            x (torch.Tensor): a tensor corresponding to a Haar Decomposition component with
                shape (batch, channels, height, width)

        Returns:
            Tuple[torch.Tensor]: a tuple with the Up Diagonal and Low Diagonal reduced tensor
                 with the width and height they would have at the scale level
        """

        assert len(x.shape) == 4, 'input tensor must have 4 dimensions'

        in_channels = x.shape[1]

        up_diagonal_conv = HaarConv2D(orientation=UP_DIAGONAL,
                                      in_channels=in_channels,
                                      by_channel=True)

        down_diagonal_conv = HaarConv2D(orientation=DOWN_DIAGONAL,
                                        in_channels=in_channels,
                                        by_channel=True)

        up_diagonal_x,  _ = up_diagonal_conv(x)
        down_diagonal_x,  _ = down_diagonal_conv(x)

        return up_diagonal_x, down_diagonal_x
