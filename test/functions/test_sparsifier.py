import unittest
from sparsifier import Sparsifier
import torch


class TestSparsifier(unittest.TestCase):

    def test_two_tailed(self):
        """
        Test Sparsifier two-tailed
        """

        x = torch.randn(100, 1)
        # run x through an Identity so we can compute gradients
        identity = torch.nn.Identity()
        x_id = identity(x)
        # set require_drad to True so we can compute gradients for x_id
        x_id.requires_grad_(True)
        # set up sparsification
        sparsity = 0.95
        sparsify = Sparsifier(dim=(0, 1), sparsity=sparsity)
        # sparsify the tensor
        sparse_x = sparsify(x_id)
        # check that the tensor is indeed sparsified
        n_non_zero = torch.numel(sparse_x[sparse_x != 0])
        n = torch.numel(x_id)
        actual_sparsity = 1 - (n_non_zero / n)
        self.assertGreaterEqual(actual_sparsity, sparsity)
        # compute a sum of the sparsified tensor so we can compute gradients
        y = sparse_x.sum()
        # do backwards propagation so gradients are computed
        y.backward()
        # get the tensor gradient before sparsification
        x_grad = x_id.grad
        # check that the gradients are sparsified
        n_non_zero_grad = torch.numel(x_grad[x_grad != 0])
        n_grad = torch.numel(x_grad)
        grad_sparsity = 1 - (n_non_zero_grad / n_grad)
        self.assertEqual(grad_sparsity, actual_sparsity)
        # check that non-sparse gradients are 1, since we only applied an Identity to the input
        self.assertTrue(torch.all(x_grad[x_grad != 0] == 1))
        # check that the non-zero gradients correspond to the non-zero elements of the input array
        non_zero_grad_idxs = torch.where(x_grad != 0)
        non_zero_sparse_x = torch.where(sparse_x != 0)

        for d in range(len(non_zero_grad_idxs)):

            for c in range(len(non_zero_grad_idxs[d])):
                self.assertTrue(non_zero_grad_idxs[d][c] == non_zero_sparse_x[d][c])

    def test_single_tailed(self):
        """
        Test Sparsifier single-tailed
        """

        x = torch.randn(100, 1)
        # run x through an Identity so we can compute gradients
        identity = torch.nn.Identity()
        x_id = identity(x)
        # set require_drad to True so we can compute gradients for x_id
        x_id.requires_grad_(True)
        # set up sparsification
        sparsity = 0.95
        sparsify = Sparsifier(dim=(0, 1), sparsity=sparsity, two_tailed=False)
        # sparsify the tensor
        sparse_x = sparsify(x_id)
        # check that the tensor is indeed sparsified
        n_non_zero = torch.numel(sparse_x[sparse_x != 0])
        n = torch.numel(x_id)
        actual_sparsity = 1 - (n_non_zero / n)
        self.assertGreaterEqual(actual_sparsity, sparsity)
        # compute a sum of the sparsified tensor so we can compute gradients
        y = sparse_x.sum()
        # do backwards propagation so gradients are computed
        y.backward()
        # get the tensor gradient before sparsification
        x_grad = x_id.grad
        # check that the gradients are sparsified
        n_non_zero_grad = torch.numel(x_grad[x_grad != 0])
        n_grad = torch.numel(x_grad)
        grad_sparsity = 1 - (n_non_zero_grad / n_grad)
        self.assertEqual(grad_sparsity, actual_sparsity)
        # check that non-sparse gradients are 1, since we only applied an Identity to the input
        self.assertTrue(torch.all(x_grad[x_grad != 0] == 1))
        # check that the non-zero gradients correspond to the non-zero elements of the input array
        non_zero_grad_idxs = torch.where(x_grad != 0)
        non_zero_sparse_x = torch.where(sparse_x != 0)

        for d in range(len(non_zero_grad_idxs)):

            for c in range(len(non_zero_grad_idxs[d])):
                self.assertTrue(non_zero_grad_idxs[d][c] == non_zero_sparse_x[d][c])

    def test_multidimensional_two_tailed(self):
        """
        Test Sparsifier two-tailed with sparsification occurring at multiple dimensions
        """

        x = torch.randn(32, 3, 512, 512)
        # run x through an Identity so we can compute gradients
        identity = torch.nn.Identity()
        x_id = identity(x)
        # set require_drad to True so we can compute gradients for x_id
        x_id.requires_grad_(True)
        # set up sparsification
        sparsity = 0.95
        sparsify = Sparsifier(dim=(1, 2, 3), sparsity=sparsity)
        # sparsify the tensor
        sparse_x = sparsify(x_id)
        # check that the tensor is indeed sparsified
        n_non_zero = torch.numel(sparse_x[sparse_x != 0])
        n = torch.numel(x_id)
        actual_sparsity = 1 - (n_non_zero / n)
        self.assertGreaterEqual(actual_sparsity, sparsity)
        # compute a sum of the sparsified tensor so we can compute gradients
        y = sparse_x.sum()
        # do backwards propagation so gradients are computed
        y.backward()
        # get the tensor gradient before sparsification
        x_grad = x_id.grad
        # check that the gradients are sparsified
        n_non_zero_grad = torch.numel(x_grad[x_grad != 0])
        n_grad = torch.numel(x_grad)
        grad_sparsity = 1 - (n_non_zero_grad / n_grad)
        self.assertEqual(grad_sparsity, actual_sparsity)
        # check that non-sparse gradients are 1, since we only applied an Identity to the input
        self.assertTrue(torch.all(x_grad[x_grad != 0] == 1))
        # check that the non-zero gradients correspond to the non-zero elements of the input array
        non_zero_grad_idxs = torch.where(x_grad != 0)
        non_zero_sparse_x = torch.where(sparse_x != 0)

        for d in range(len(non_zero_grad_idxs)):
            self.assertTrue(torch.all(non_zero_grad_idxs[d] == non_zero_sparse_x[d]))

    def test_multidimensional_single_tailed(self):
        """
        Test Sparsifier single-tailed with sparsification occurring at multiple dimensions
        """

        x = torch.randn(32, 3, 512, 512)
        # run x through an Identity so we can compute gradients
        identity = torch.nn.Identity()
        x_id = identity(x)
        # set require_drad to True so we can compute gradients for x_id
        x_id.requires_grad_(True)
        # set up sparsification
        sparsity = 0.95
        sparsify = Sparsifier(dim=(1, 2, 3), sparsity=sparsity, two_tailed=False)
        # sparsify the tensor
        sparse_x = sparsify(x_id)
        # check that the tensor is indeed sparsified
        n_non_zero = torch.numel(sparse_x[sparse_x != 0])
        n = torch.numel(x_id)
        actual_sparsity = 1 - (n_non_zero / n)
        self.assertGreaterEqual(actual_sparsity, sparsity)
        # compute a sum of the sparsified tensor so we can compute gradients
        y = sparse_x.sum()
        # do backwards propagation so gradients are computed
        y.backward()
        # get the tensor gradient before sparsification
        x_grad = x_id.grad
        # check that the gradients are sparsified
        n_non_zero_grad = torch.numel((x_grad[x_grad != 0]))
        n_grad = torch.numel(x_grad)
        grad_sparsity = 1 - (n_non_zero_grad / n_grad)
        self.assertEqual(grad_sparsity, actual_sparsity)
        # check that non-sparse gradients are 1, since we only applied an Identity to the input
        self.assertTrue(torch.all(x_grad[x_grad != 0] == 1))
        # check that the non-zero gradients correspond to the non-zero elements of the input array
        non_zero_grad_idxs = torch.where(x_grad != 0)
        non_zero_sparse_x = torch.where(sparse_x != 0)

        for d in range(len(non_zero_grad_idxs)):
            self.assertTrue(torch.all(non_zero_grad_idxs[d] == non_zero_sparse_x[d]))


if __name__ == '__main__':
    unittest.main()
