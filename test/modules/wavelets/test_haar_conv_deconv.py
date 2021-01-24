import os
import pathlib
import unittest
from PIL import Image
import torch
from bramble.utils import to_tensor
from onda.modules.wavelets import HaarConv2D, HaarDeconv2D, \
    HORIZONTAL, VERTICAL, UP_DIAGONAL, DOWN_DIAGONAL


class TestHaar(unittest.TestCase):

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
        test convolution and deconvolution with a simple tensor

        """

        t = torch.FloatTensor([[1,  2,  3,  4],
                               [5,  6,  7,  8],
                               [9,  10, 11, 12],
                               [13, 14, 15, 16]])

        t.unsqueeze_(0).unsqueeze_(0)
        n_channels = t.shape[1]

        for orientation in [VERTICAL, HORIZONTAL, UP_DIAGONAL, DOWN_DIAGONAL]:
            haar = HaarConv2D(orientation=orientation, in_channels=n_channels)
            (low_pass, detail) = haar(t)

            if orientation == VERTICAL:
                expected_low_pass = torch.FloatTensor([[3,   4,  5,  6],
                                                       [11, 12, 13, 14]])
                self.assertTrue(torch.all(low_pass == expected_low_pass))

                expected_detail = torch.FloatTensor([[-2, -2, -2, -2],
                                                     [-2, -2, -2, -2]])
                self.assertTrue(torch.all(detail == expected_detail))

            elif orientation == HORIZONTAL:
                expected_low_pass = torch.FloatTensor([[1.5,  3.5],
                                                       [5.5,  7.5],
                                                       [9.5,  11.5],
                                                       [13.5, 15.5]])
                self.assertTrue(torch.all(low_pass == expected_low_pass))

                expected_detail = torch.FloatTensor([[-0.5, -0.5],
                                                     [-0.5, -0.5],
                                                     [-0.5, -0.5],
                                                     [-0.5, -0.5]])
                self.assertTrue(torch.all(detail == expected_detail))

            elif orientation == UP_DIAGONAL:
                expected_low_pass = torch.FloatTensor([[3.5,   5.5],
                                                       [11.5, 13.5]])
                self.assertTrue(torch.all(low_pass == expected_low_pass))

                expected_detail = torch.FloatTensor([[1.5, 1.5],
                                                     [1.5, 1.5]])
                self.assertTrue(torch.all(detail == expected_detail))

            elif orientation == DOWN_DIAGONAL:
                expected_low_pass = torch.FloatTensor([[3.5,  5.5],
                                                       [11.5, 13.5]])
                self.assertTrue(torch.all(low_pass == expected_low_pass))

                expected_detail = torch.FloatTensor([[-2.5, -2.5],
                                                     [-2.5, -2.5]])
                self.assertTrue(torch.all(detail == expected_detail))

            haar_deconv = HaarDeconv2D(orientation=orientation, in_channels=n_channels)
            deconv = haar_deconv((low_pass, detail))

            if orientation in [VERTICAL, HORIZONTAL]:
                self.assertTrue(torch.all(t == deconv))

            if orientation in [UP_DIAGONAL, DOWN_DIAGONAL]:
                self.assertTrue(torch.all(t[deconv != 0] == deconv[deconv != 0]))

    def test_base_case2(self):
        """
        test convolution and deconvolution with a simple tensor but with less symmetric tensor

        """

        t = torch.FloatTensor([[2, 10,  5, 16],
                               [15, 1, 14,  6],
                               [11, 8, 12, 14],
                               [3,  9,  7, 13]]).unsqueeze_(0).unsqueeze_(0)

        n_channels = t.shape[1]

        for orientation in [VERTICAL, HORIZONTAL, UP_DIAGONAL, DOWN_DIAGONAL]:
            haar = HaarConv2D(orientation=orientation, in_channels=n_channels)
            haar_deconv = HaarDeconv2D(orientation=orientation, in_channels=n_channels)

            (low_pass, detail) = haar(t)

            if orientation == VERTICAL:
                expected_low_pass = torch.FloatTensor([[8.5, 5.5, 9.5,   11],
                                                       [7,   8.5, 9.5, 13.5]])
                self.assertTrue(torch.all(low_pass == expected_low_pass))

                expected_detail = torch.FloatTensor([[-6.5, 4.5, -4.5,   5],
                                                     [4,   -0.5,  2.5, 0.5]])
                self.assertTrue(torch.all(detail == expected_detail))

            elif orientation == HORIZONTAL:
                expected_low_pass = torch.FloatTensor([[6,  10.5],
                                                       [8,    10],
                                                       [9.5,  13],
                                                       [6,   10]])
                self.assertTrue(torch.all(low_pass == expected_low_pass))

                expected_detail = torch.FloatTensor([[-4,  -5.5],
                                                     [7,      4],
                                                     [1.5,   -1],
                                                     [-3,    -3]])
                self.assertTrue(torch.all(detail == expected_detail))

            elif orientation == UP_DIAGONAL:
                expected_low_pass = torch.FloatTensor([[[[12.5,  15],
                                                         [5.5,  10.5]]]])
                self.assertTrue(torch.all(low_pass == expected_low_pass))

                expected_detail = torch.FloatTensor([[[[2.5, -1],
                                                     [-2.5, -3.5]]]])
                self.assertTrue(torch.all(detail == expected_detail))

                reconstructed = haar_deconv((low_pass, detail))
                self.assertTrue(torch.all(reconstructed[reconstructed != 0] == t[reconstructed != 0]))

                level2_low_pass, level2_detail = haar(low_pass)
                expected_level2_low_pass = torch.FloatTensor([[[[10.25]]]])
                self.assertTrue(torch.all(level2_low_pass == expected_level2_low_pass))

                expected_level2_detail = torch.FloatTensor([[[[-4.75]]]])
                self.assertTrue(torch.all(level2_detail == expected_level2_detail))

                low_pass_deconv = haar_deconv((level2_low_pass, level2_detail))
                expected_low_pass_deconv = torch.FloatTensor([[[[0,  15],
                                                                [5.5, 0]]]])
                self.assertTrue(torch.all(low_pass_deconv == expected_low_pass_deconv))



            elif orientation == DOWN_DIAGONAL:
                expected_low_pass = torch.FloatTensor([[1.5,  5.5],
                                                       [10,  12.5]])
                self.assertTrue(torch.all(low_pass == expected_low_pass))

                expected_detail = torch.FloatTensor([[0.5, -0.5],
                                                     [1,   -0.5]])
                self.assertTrue(torch.all(detail == expected_detail))

            deconv = haar_deconv((low_pass, detail))

            if orientation in [VERTICAL, HORIZONTAL]:
                self.assertTrue(torch.all(t == deconv))

            if orientation in [UP_DIAGONAL, DOWN_DIAGONAL]:
                self.assertTrue(torch.all(t[deconv != 0] == deconv[deconv != 0]))

    def test_haar_conv(self):
        """
        Base test.  Run image through convolution, check that the shapes are as expected,
        that the values are correct and that deconvolution yields the original image
        """

        path = self.get_data_path()
        img1_path = os.path.join(path, 'img1.jpg')
        image = Image.open(img1_path)
        image = self.letterbox_image(image, (512, 512))
        image = to_tensor(image)

        if len(image.shape) < 3:
            image.unsqueeze_(2)

        image.transpose_(0, 2)
        image.transpose_(2, 1)
        image.unsqueeze_(0)
        n_channels = image.shape[1]

        for orientation in [VERTICAL, HORIZONTAL, UP_DIAGONAL, DOWN_DIAGONAL]:
            haar = HaarConv2D(orientation=orientation, in_channels=n_channels)
            (image_low_pass, image_detail) = haar(image)

            self.assertIsNotNone(image_low_pass)
            self.assertIsNotNone(image_detail)

            assert image_low_pass.shape == image_detail.shape

            if orientation == VERTICAL:
                assert image_low_pass.shape[2] == image.shape[2] / 2
                assert image_low_pass.shape[3] == image.shape[3]

            elif orientation == HORIZONTAL:
                assert image_low_pass.shape[2] == image.shape[2]
                assert image_low_pass.shape[3] == image.shape[3] / 2

            elif orientation in [UP_DIAGONAL, DOWN_DIAGONAL]:
                assert image_low_pass.shape[2] == image.shape[2] / 2
                assert image_low_pass.shape[3] == image.shape[3] / 2

            for row in range(image_low_pass.shape[2]):

                for column in range(image_low_pass.shape[3]):

                    low_pass = image_low_pass[0, :, row, column]
                    detail = image_detail[0, :, row, column]

                    if orientation == VERTICAL:
                        original = image[0, :, 2 * row: (2 * row) + 2, column]
                        self.assertTrue(torch.all(low_pass == (original[:, 0] + original[:, 1]) / 2))
                        self.assertTrue(torch.all(detail == (original[:, 0] - original[:, 1]) / 2))

                    elif orientation == HORIZONTAL:
                        original = image[0, :, row, 2 * column: (2 * column) + 2]
                        self.assertTrue(torch.all(low_pass == (original[:, 0] + original[:, 1]) / 2))
                        self.assertTrue(torch.all(detail == (original[:, 0] - original[:, 1]) / 2))

                    elif orientation == UP_DIAGONAL:
                        original = image[0, :, 2 * row: (2 * row) + 2, 2 * column: (2 * column) + 2]
                        self.assertTrue(torch.all(low_pass == (original[:, 1, 0] + original[:, 0, 1]) / 2))
                        self.assertTrue(torch.all(detail == (original[:, 1, 0] - original[:, 0, 1]) / 2))

                    elif orientation == DOWN_DIAGONAL:
                        original = image[0, :, 2 * row: (2 * row) + 2, 2 * column: (2 * column) + 2]
                        self.assertTrue(torch.all(low_pass == (original[:, 0, 0] + original[:, 1, 1]) / 2))
                        self.assertTrue(torch.all(detail == (original[:, 0, 0] - original[:, 1, 1]) / 2))

    def test_batch(self):
        """
        test a batch. Run image through convolution, check that the shapes are as expected,
        that the values are correct and that deconvolution yields the original image
        """
        batch_size = 32
        n_channels = 3
        batch = torch.rand(batch_size, n_channels, 64, 64)

        for orientation in [VERTICAL, HORIZONTAL, UP_DIAGONAL, DOWN_DIAGONAL]:
            haar = HaarConv2D(orientation=orientation, in_channels=n_channels)
            (image_low_pass, image_detail) = haar(batch)

            self.assertIsNotNone(image_low_pass)
            self.assertIsNotNone(image_detail)

            assert image_low_pass.shape == image_detail.shape

            if orientation == VERTICAL:
                assert image_low_pass.shape[2] == batch.shape[2] // 2
                assert image_low_pass.shape[3] == batch.shape[3]

            elif orientation == HORIZONTAL:
                assert image_low_pass.shape[2] == batch.shape[2]
                assert image_low_pass.shape[3] == batch.shape[3] // 2

            elif orientation in [UP_DIAGONAL, DOWN_DIAGONAL]:
                assert image_low_pass.shape[2] == batch.shape[2] // 2
                assert image_low_pass.shape[3] == batch.shape[3] // 2

            for row in range(image_low_pass.shape[2]):

                for column in range(image_low_pass.shape[3]):

                    low_pass = image_low_pass[0, :, row, column]
                    detail = image_detail[0, :, row, column]

                    if orientation == VERTICAL:
                        original = batch[0, :, 2 * row: (2 * row) + 2, column]
                        self.assertTrue(torch.all(low_pass == (original[:, 0] + original[:, 1]) / 2))
                        self.assertTrue(torch.all(detail == (original[:, 0] - original[:, 1]) / 2))

                    elif orientation == HORIZONTAL:
                        original = batch[0, :, row, 2 * column: (2 * column) + 2]
                        self.assertTrue(torch.all(low_pass == (original[:, 0] + original[:, 1]) / 2))
                        self.assertTrue(torch.all(detail == (original[:, 0] - original[:, 1]) / 2))

                    elif orientation == UP_DIAGONAL:
                        original = batch[0, :, 2 * row: (2 * row) + 2, 2 * column: (2 * column) + 2]
                        self.assertTrue(torch.all(low_pass == (original[:, 1, 0] + original[:, 0, 1]) / 2))
                        self.assertTrue(torch.all(detail == (original[:, 1, 0] - original[:, 0, 1]) / 2))

                    elif orientation == DOWN_DIAGONAL:
                        original = batch[0, :, 2 * row: (2 * row) + 2, 2 * column: (2 * column) + 2]
                        self.assertTrue(torch.all(low_pass == (original[:, 0, 0] + original[:, 1, 1]) / 2))
                        self.assertTrue(torch.all(detail == (original[:, 0, 0] - original[:, 1, 1]) / 2))

    def test_batch_aggregate_channels(self):
        """
        test a batch with by_channel = False.  Run image through convolution, check that the shapes
        are as expected, that the values are correct and that deconvolution yields the original
        image
        """
        batch_size = 32
        n_channels = 3
        batch = torch.rand(batch_size, n_channels, 64, 64)

        for orientation in [VERTICAL, HORIZONTAL, UP_DIAGONAL, DOWN_DIAGONAL]:
            haar = HaarConv2D(orientation=orientation, in_channels=n_channels, by_channel=False)
            (image_low_pass, image_detail) = haar(batch)

            self.assertIsNotNone(image_low_pass)
            self.assertIsNotNone(image_detail)

            assert image_low_pass.shape == image_detail.shape

            if orientation == VERTICAL:
                assert image_low_pass.shape[2] == batch.shape[2] // 2
                assert image_low_pass.shape[3] == batch.shape[3]

            elif orientation == HORIZONTAL:
                assert image_low_pass.shape[2] == batch.shape[2]
                assert image_low_pass.shape[3] == batch.shape[3] // 2

            elif orientation in [UP_DIAGONAL, DOWN_DIAGONAL]:
                assert image_low_pass.shape[2] == batch.shape[2] // 2
                assert image_low_pass.shape[3] == batch.shape[3] // 2

            for row in range(image_low_pass.shape[2]):

                for column in range(image_low_pass.shape[3]):

                    low_pass = image_low_pass[:, :, row, column]
                    detail = image_detail[:, :, row, column]

                    if orientation in (HORIZONTAL, VERTICAL):

                        if orientation == VERTICAL:
                            original = batch[:, :, 2 * row: (2 * row) + 2, column]

                        elif orientation == HORIZONTAL:
                            original = batch[:, :, row, 2 * column: (2 * column) + 2]

                        original = torch.mean(original, dim=1)
                        original_low_pass = (original[:, 0] + original[:, 1]) / 2
                        original_detail = (original[:, 0] - original[:, 1]) / 2

                    elif orientation in (UP_DIAGONAL, DOWN_DIAGONAL):
                        original = batch[:, :, 2 * row: (2 * row) + 2, 2 * column: (2 * column) + 2]
                        original = torch.mean(original, dim=1)

                        if orientation == UP_DIAGONAL:
                            original_low_pass = (original[:, 1, 0] + original[:, 0, 1]) / 2
                            original_detail = (original[:, 1, 0] - original[:, 0, 1]) / 2

                        elif orientation == DOWN_DIAGONAL:
                            original_low_pass = (original[:, 0, 0] + original[:, 1, 1]) / 2
                            original_detail = (original[:, 0, 0] - original[:, 1, 1]) / 2

                    low_pass_diff = torch.abs(low_pass[:, 0] - original_low_pass)
                    detail_diff = torch.abs(detail[:, 0] - original_detail)

                    self.assertTrue(torch.all(low_pass_diff < 1e-6))
                    self.assertTrue(torch.all(detail_diff < 1e-6))


if __name__ == '__main__':
    unittest.main()
