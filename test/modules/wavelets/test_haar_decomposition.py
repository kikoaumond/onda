import unittest
import torch
from onda.modules.wavelets import HaarDecomposition2D
import math
import os
import pathlib
from PIL import Image
from bramble.utils import to_tensor, display


class TestHaarDecomposition(unittest.TestCase):

    def get_data_path(self):
        """
        Get the path for the image files
        Returns:
            (str): the path for the image files
        """

        dir_path = os.path.dirname(os.path.realpath(__file__))
        root_path = pathlib.Path(dir_path).parent.parent.parent
        data_path = os.path.join(root_path, 'data', 'images')

        return data_path

    def letterbox_image(self, image, size, canvas_color=(0, 0, 0)):
        """
        resize image  with unchanged aspect ratio using padding
        """

        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)

        if image.mode == 'L':
            canvas_color = canvas_color[0]

        new_image = Image.new(image.mode, size, canvas_color)
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

        return new_image

    def test_base_case(self):
        """
        test decomposition with defaults

        """

        t = torch.FloatTensor([[2,  10,  5,  16],
                               [15,  1,  14,  6],
                               [11,  8, 12, 14],
                               [3, 9, 7, 13]]).unsqueeze_(0).unsqueeze_(0)

        hd = HaarDecomposition2D(input_shape=t.shape)
        decomposition = hd(t)
        max_levels = math.floor(math.log2(min(t.shape[2:])))
        # check that the deocomposition has the expected shape
        expected_shape = (t.shape[0], max_levels + 1, t.shape[2], t.shape[3])
        self.assertEqual(decomposition.shape, expected_shape)
        '''
        expected_level_one_low_pass = torch.FloatTensor([[1.5,  12.5,  5.5,   15],
                                                         [12.5,  1.5,   15,  5.5],
                                                         [10,    5.5, 12.5, 10.5],
                                                         [5.5,    10, 10.5, 12.5]])
        
        self.assertTrue(torch.all(decomposition[0, 0, :, :] == expected_level_one_low_pass))
        '''

        expected_level_one_detail = torch.FloatTensor([[0.5,  2.5, -0.5,   -1],
                                                       [2.5,  0.5,  -1,  -0.5],
                                                       [1,   -2.5, -0.5, -3.5],
                                                       [-2.5,   1, -3.5, -0.5]])
        self.assertTrue(torch.all(decomposition[0, 0, :, :] == expected_level_one_detail))

        expected_level_two_detail = torch.FloatTensor([[-5.5,     1,  2.25, -4.75],
                                                       [1,     -5.5, -4.75,  2.25],
                                                       [2.25, -4.75,  -5.5,     1],
                                                       [-4.75, 2.25,     1,  -5.5]])
        self.assertTrue(torch.all(decomposition[0, 1, :, :] == expected_level_two_detail))

        expected_level_two_low_pass = torch.FloatTensor([[7,     11.5,  7.75, 10.25],
                                                         [11.5,     7, 10.25,  7.75],
                                                         [7.75, 10.25,     7,  11.5],
                                                         [10.25, 7.75,  11.5,     7]])
        self.assertTrue(torch.all(decomposition[0, 2, :, :] == expected_level_two_low_pass))

        # check that the reconstructed tensor matches the original tensor
        reconstructed = HaarDecomposition2D.reconstruct(decomposition)

        self.assertTrue(torch.all(reconstructed == t))

    def test_multi_channel(self):
        """
        Test decompostion of a multi-channel random batch
        """
        batch_size = 1
        n_channels = 1
        batch = torch.rand(batch_size, n_channels, 16, 16)
        hd = HaarDecomposition2D(input_shape=batch.shape)
        decomposition = hd(batch)
        max_levels = math.floor(math.log2(min(batch.shape[2:])))
        # check that the decomposition has the expected shape
        expected_shape = (batch.shape[0], max_levels + 1, batch.shape[2], batch.shape[3])
        self.assertEquals(decomposition.shape, expected_shape)
        reconstructed = HaarDecomposition2D.reconstruct(decomposition,
                                                        in_channels=n_channels)
        self.assertEquals(reconstructed.shape, batch.shape)
        reconstruction_error = torch.mean(torch.abs(batch - reconstructed))
        self.assertLessEqual(reconstruction_error, 1e-6)

    def test_sparsity(self):
        """
        test with sparsification
        """
        batch_size = 8
        n_channels = 3
        sparsity = 0.95
        height = 32
        width = 32
        batch = torch.rand(batch_size, n_channels, height, width)
        hd = HaarDecomposition2D(input_shape=batch.shape, sparsity=sparsity, by_channel=True)
        decomposition = hd(batch)
        reconstructed = HaarDecomposition2D.reconstruct(decomposition,
                                                        in_channels=n_channels)
        max_levels = math.floor(math.log2(min(batch.shape[2:])))
        expected_shape = (batch.shape[0], (max_levels + 1) * n_channels, batch.shape[2], batch.shape[3])
        self.assertEquals(decomposition.shape, expected_shape)

        # Check sparsity for each individual tensor in the batch
        for batch_index in range(decomposition.shape[0]):
            image = decomposition[batch_index, ...]
            image_size = torch.numel(image)
            n_non_zero_elements = torch.numel(image[image != 0])
            actual_sparsity = 1 - (n_non_zero_elements / image_size)
            self.assertGreaterEqual(actual_sparsity, sparsity - 0.01)

    def test_sparsity_with_image(self):
        """
        Apply sparsification to an image
        """
        path = self.get_data_path()
        img1_path = os.path.join(path, 'img1.jpg')
        img1_path = '/workspace/data/train/train_1.1.8_ml/images/COCO_train2014_000000000036.jpg'
        image = Image.open(img1_path).convert('L')
        image = self.letterbox_image(image, (512, 512))
        display(image, None)
        image = to_tensor(image).unsqueeze_(0)
        image.unsqueeze_(0)
        sparsity = 0.95
        hd = HaarDecomposition2D(input_shape=image.shape, sparsity=sparsity, by_channel=True)
        decomposition = hd(image)
        reconstructed_image = HaarDecomposition2D.reconstruct(decomposition, in_channels=1)
        reconstructed_image.squeeze_(0).transpose_(0, 2).transpose_(0, 1)
        display(reconstructed_image, None)




if __name__ == '__main__':
    unittest.main()
