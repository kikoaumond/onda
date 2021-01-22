import torch

#  constants for Haar Wavelet decomposition

# Haar Convolution Kernels
HORIZONTAL_LOW_PASS_KERNEL = torch.FloatTensor([[0.5, 0.5]])
HORIZONTAL_DETAIL_KERNEL = torch.FloatTensor([[0.5, -0.5]])

HORIZONTAL_LOW_PASS_DECONV_KERNEL = torch.FloatTensor([[1, 0]])
HORIZONTAL_DETAIL_DECONV_KERNEL = torch.FloatTensor([[0, 1]])

VERTICAL_LOW_PASS_KERNEL = torch.FloatTensor([[0.5], [0.5]])
VERTICAL_DETAIL_KERNEL = torch.FloatTensor([[0.5], [-0.5]])

VERTICAL_LOW_PASS_DECONV_KERNEL = torch.FloatTensor([[1], [0]])
VERTICAL_DETAIL_DECONV_KERNEL = torch.FloatTensor([[0], [1]])

UP_DIAGONAL_LOW_PASS_KERNEL = torch.FloatTensor([[0, 0.5],
                                                 [0.5, 0]])
UP_DIAGONAL_DETAIL_KERNEL = torch.FloatTensor([[0, -0.5],
                                               [0.5,  0]])

UP_DIAGONAL_LOW_PASS_DECONV_KERNEL = torch.FloatTensor([[0, 0],
                                                        [1, 0]])
UP_DIAGONAL_DETAIL_DECONV_KERNEL = torch.FloatTensor([[0, 1],
                                                      [0, 0]])

DOWN_DIAGONAL_LOW_PASS_KERNEL = torch.FloatTensor([[0.5, 0],
                                                   [0, 0.5]])
DOWN_DIAGONAL_DETAIL_KERNEL = torch.FloatTensor([[0.5,  0],
                                                 [0, -0.5]])

DOWN_DIAGONAL_LOW_PASS_DECONV_KERNEL = torch.FloatTensor([[1, 0],
                                                          [0, 0]])
DOWN_DIAGONAL_DETAIL_DECONV_KERNEL = torch.FloatTensor([[0, 0],
                                                        [0, 1]])

# indices
LOW_PASS = 0
DETAIL = 1

# Kernel orientations
VERTICAL = 0
HORIZONTAL = 1
UP_DIAGONAL = 2
DOWN_DIAGONAL = 3

VERTICAL_HORIZONTAL = 10
DIAGONAL = 20

# mapping of orientation to Kernels
KERNELS = {VERTICAL: (VERTICAL_LOW_PASS_KERNEL,
                      VERTICAL_DETAIL_KERNEL),
           HORIZONTAL: (HORIZONTAL_LOW_PASS_KERNEL,
                        HORIZONTAL_DETAIL_KERNEL),
           UP_DIAGONAL: (UP_DIAGONAL_LOW_PASS_KERNEL,
                         UP_DIAGONAL_DETAIL_KERNEL),
           DOWN_DIAGONAL: (DOWN_DIAGONAL_LOW_PASS_KERNEL,
                           DOWN_DIAGONAL_DETAIL_KERNEL)}

DECONV_KERNELS = {VERTICAL: (VERTICAL_LOW_PASS_DECONV_KERNEL,
                             VERTICAL_DETAIL_DECONV_KERNEL),
                  HORIZONTAL: (HORIZONTAL_LOW_PASS_DECONV_KERNEL,
                               HORIZONTAL_DETAIL_DECONV_KERNEL),
                  UP_DIAGONAL: (UP_DIAGONAL_LOW_PASS_DECONV_KERNEL,
                                UP_DIAGONAL_DETAIL_DECONV_KERNEL),
                  DOWN_DIAGONAL: (DOWN_DIAGONAL_LOW_PASS_DECONV_KERNEL,
                                  DOWN_DIAGONAL_DETAIL_DECONV_KERNEL)}
