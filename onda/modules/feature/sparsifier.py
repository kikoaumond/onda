import math
import torch


class Sparsify(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, sparsity_mask):
        assert input.shape == sparsity_mask.shape, \
            'tensor and sparsity mask must have the same shape.\n' \
            'tensor shape: {} != {} sparsity mask shape'\
            .format(input.shape, sparsity_mask.shape)

        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.sparsity_mask = sparsity_mask
        ctx.save_for_backward(sparsity_mask)
        sparsified_tensor = input * sparsity_mask

        return sparsified_tensor

    @staticmethod
    def backward(ctx, grad_output):

        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        sparsity_mask, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input *= sparsity_mask

        return grad_input, None


class Sparsifier(torch.nn.Module):
    """
        Sparsifies tensors by setting its elements to 0 if their magnitude falls below a
        a specified quantile.  Sparsification can be performed for the whole tensor or individual
        channels.
    """

    def __init__(self, sparsity, dim, two_tailed=True):
        """
        Sparsifier constructor

        Args:
                input (torch.tensor): a tensor with dimensions (batch_size, channels, height, width)

                sparsity (float): the desired the sparsity; must be between 0 and 1, exclusive.
                    Higher sparsity means more elements will be set to 0

                dim (int / tuple(int)): the dimensions along which to sparsify; can be a single
                    dimension or a tuple of dimensions

                two_tailed (bool): if True, both the top and bottom elements will be retained.
                    for instance, if sparsity is 0.9, the bottom and top 5% elements will be
                    retained, all others will be set to 0
                    Default: False
        """
        super(Sparsifier, self).__init__()
        x = torch.nn.ReLU()

        if sparsity <= 0. or sparsity >= 1:
            raise ValueError('sparsity must be >= 0 and < 1.  Sparsity: {}'.format(sparsity))

        self._sparsity = sparsity

        assert isinstance(dim, int) \
            or (isinstance(dim, (tuple, list))
                and all([isinstance(d, int)
                         and d >= 0 for d in dim])), \
            'dim must be a single non-negative integer or a tuple/list of non-negative integers\n' \
            'dim: {}'.format(dim)

        self._dim = sorted(dim) if isinstance(dim, (tuple, list)) else (dim,)
        self._two_tailed = two_tailed

    def _compute_dims(self, shape):
        """
        compute the size of non-sparse elements as well as the number of dimensions given
        the shape of a tensor to be sparsified and the dimensions along which to sparsify

        Args:
            shape (tuple(int)): the shape of the tensor to be sparsified

        Returns:
            (int, tuple(int)): the length of the dimension(s) along which to sparsify and
                the shape of the tensor to be sorted

        """
        assert max(self._dim) < len(shape), \
            'Sparsification will happen at dimension(s) {} but tensor has only {} dimensions'\
            .format(self._dim, len(shape))

        # compute the length of the dimension(s) along which to sparsify
        length = 1

        for d in self._dim:
            length *= shape[d]

        # compute the shape to which the original tensor will be reshaped in order to be sorted
        sorted_shape = []

        for dim_idx, size in enumerate(shape):

            if dim_idx in self._dim:
                continue

            sorted_shape.append(size)

        # append the length of the sparsified dimension(s)
        sorted_shape.append(length)
        sorted_shape = tuple(sorted_shape)

        return length, sorted_shape

    def forward(self, x):
        """
            Sparsify a tensor by making its elements 0 where the element is lower than element
            in the quantile defined by sparsity.  For instance, if sparsity = 0.99 only
            the elements greater or equal than the element in the 99% quantile will be retained and
            all others will be set to 0

            Args:
                x (torch.tensor): a tensor with dimensions (batch_size, channels, height, width)

        """

        if self._sparsity == 0.:

            return x

        length, sorted_shape = self._compute_dims(x.shape)
        t = x

        if sorted_shape != x.shape:
            t = x.reshape(sorted_shape)

        dim_to_sort = len(t.shape) - 1

        if self._two_tailed:
            sorted_t, sorted_indices = torch.sort(t.abs(), dim=dim_to_sort)

        else:
            sorted_t, sorted_indices = torch.sort(t, dim=dim_to_sort)

        # the number of tensor elements that will not be zero after sparsification
        n_sparse = math.floor((1 - self._sparsity) * length)

        index = sorted_t.shape[-1] - n_sparse - 1
        values = sorted_t[..., index]

        while len(values.shape) < len(x.shape):
            dim_to_unsqueeze = len(values.shape)
            values.unsqueeze_(dim=dim_to_unsqueeze)

        if self._two_tailed:
            sparsity_mask = (x.abs() >= values).int()

        else:
            sparsity_mask = (x >= values).int()

        sparsified_tensor = Sparsify.apply(x, sparsity_mask)

        return sparsified_tensor
