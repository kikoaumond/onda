import torch
import torch.nn.functional as F
from onda.modules.wavelets.haar import UP_DIAGONAL, DOWN_DIAGONAL


class DiagonalUpsample(torch.nn.Module):
    """
    Upsamples the Up Diagonal and Down Diagonal components of a Haar Decomposition of a Tensor
    by combining them, resulting in an upsampled tensor with width and height equal to twice
    the width and height of the original decomposition tensors

    If the Up Diagonal and Down Diagonal decomposition tensors are, respectively
    [[u00, u01],
     [u10, u11]]

     and

     [[d00, d01],
      [d10, d11]]

    respectively, DiagonalUpsample will upsample each component to

    [[  0, u00,   0, u01],
     [u00,   0, u01, 0  ],
     [0,   u10,   0, u11],
     [u10, 0,   u11, 0  ]]

    and

    [[d00,   0,   d01, 0  ],
     [  0, d00,     0, d01],
     [d10,   0,   d11, 0  ],
     [  0, d10,     0, d11]]

     and then sum them, resulting in the upsampled tensor

    [[d00, u00, d01, u01],
     [u00, d00, u01, d01],
     [d10, u10, d11, u11],
     [u10, d10, u11, d11]]
    """

    def __init__(self, in_channels=3):
        """

        Args:
            in_channels (int):  the number of channels in the tensor.  in_channels must
            be equal to the size of dimension of index 1 in the tensors submitted to forward()
        """
        super(DiagonalUpsample, self).__init__()
        self._in_channels = in_channels

        udk1 = torch.Tensor([[0, 0],
                             [1, 0]]).unsqueeze_(dim=0).unsqueeze_(dim=0)

        udk2 = torch.Tensor([[0, 1],
                             [0, 0]]).unsqueeze_(dim=0).unsqueeze_(dim=0)

        ddk1 = torch.Tensor([[1, 0],
                             [0, 0]]).unsqueeze_(dim=0).unsqueeze_(dim=0)

        ddk2 = torch.Tensor([[0, 0],
                             [0, 1]]).unsqueeze_(dim=0).unsqueeze_(dim=0)

        if self._in_channels > 1:
            udk1 = udk1.repeat(self._in_channels, 1, 1, 1)
            udk2 = udk2.repeat(self._in_channels, 1, 1, 1)
            ddk1 = ddk1.repeat(self._in_channels, 1, 1, 1)
            ddk2 = ddk2.repeat(self._in_channels, 1, 1, 1)

        self._up_diagonal_kernels = [udk1, udk2]
        self._down_diagonal_kernels = [ddk1, ddk2]

    def forward(self, x, orientation=None):
        """
        Upsample 2 tensors corresponding to the Up Diagonal and Down Diagonal Haar Decomposition
        of a tensor resulting in an upsampled tensor with twice the width and height of the original
        tensors

        Args:
            x(Tuple[torch.Tensor]): a tuple containing one or two tensors,the Up Diagonal and the
                Down Diagonal tensor, in this order.  Both tensors must have the same dimensions
                If a single tensor is supplied, orientation must be specified

            orientation (int): one of onda.modules.wavelets.haar.UP_DIAGONAL
                or onda.modules.wavelets.haar.DOWN_DIAGONAL, denoting the direction along which
                to perform upsampling.  Must be supplied if a single tensor is passed

        Returns:
            (torch.Tensor): the resulting upsampled tensor

        """
        if len(x) == 2:
            up_diagonal, down_diagonal = x

        elif len(x) == 1:

            if orientation == UP_DIAGONAL:
                up_diagonal = x[0]
                down_diagonal = None

            elif orientation == DOWN_DIAGONAL:
                down_diagonal = x[0]
                up_diagonal = None

            else:
                raise ValueError('Illegal value for orientation: {}\n Orientation must be one of '
                                 '(onda.modules.wavelets.haar.UP_DIAGONAL, '
                                 'onda.modules.wavelets.haar.DOWN_DIAGONAL)'.format(orientation))

        else:
            raise ValueError('Only one or two tensors can be passed to DiagonalUpsample')

        if up_diagonal is not None and down_diagonal is not None:

            assert up_diagonal.shape == down_diagonal.shape, \
                'Tensors must habe the same shape but instead they have shapes {} and {}'\
                .format(up_diagonal.shape, down_diagonal.shape)

        # the number of convolution groups is equal to the number of channels
        if up_diagonal is not None:
            groups = up_diagonal.shape[1]
        else:
            groups = down_diagonal.shape[1]

        if up_diagonal is not None:
            up_diagonal_upsampled = None

            for up_diagonal_kernel in self._up_diagonal_kernels:

                if up_diagonal_upsampled is None:
                    up_diagonal_upsampled = F.conv_transpose2d(up_diagonal,
                                                               weight=up_diagonal_kernel,
                                                               stride=(2, 2),
                                                               padding=0,
                                                               groups=groups)
                else:
                    up_diagonal_upsampled += F.conv_transpose2d(up_diagonal,
                                                                weight=up_diagonal_kernel,
                                                                stride=(2, 2),
                                                                padding=0,
                                                                groups=groups)

        if down_diagonal is not None:
            down_diagonal_upsampled = None

            for down_diagonal_kernel in self._down_diagonal_kernels:

                if down_diagonal_upsampled is None:
                    down_diagonal_upsampled = F.conv_transpose2d(down_diagonal,
                                                                 weight=down_diagonal_kernel,
                                                                 stride=(2, 2),
                                                                 padding=0,
                                                                 groups=groups)
                else:
                    down_diagonal_upsampled += F.conv_transpose2d(down_diagonal,
                                                                  weight=down_diagonal_kernel,
                                                                  stride=(2, 2),
                                                                  padding=0,
                                                                  groups=groups)

        if up_diagonal is not None and down_diagonal is not None:
            upsampled = up_diagonal_upsampled + down_diagonal_upsampled

        elif up_diagonal is not None:
            upsampled = up_diagonal_upsampled

        elif down_diagonal is not None:
            upsampled = down_diagonal_upsampled

        return upsampled
