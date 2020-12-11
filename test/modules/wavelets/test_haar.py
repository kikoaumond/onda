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
                expected_low_pass = torch.FloatTensor([[6, 8, 10, 12],
                                                       [22, 24, 26, 28]])
                self.assertTrue(torch.all(low_pass == expected_low_pass))

                expected_detail = torch.FloatTensor([[-4, -4, -4, -4],
                                                     [-4, -4, -4, -4]])
                self.assertTrue(torch.all(detail == expected_detail))

            elif orientation == HORIZONTAL:
                expected_low_pass = torch.FloatTensor([[3, 7],
                                                       [11, 15],
                                                       [19, 23],
                                                       [27, 31]])
                self.assertTrue(torch.all(low_pass == expected_low_pass))

                expected_detail = torch.FloatTensor([[-1, -1],
                                                     [-1, -1],
                                                     [-1, -1],
                                                     [-1, -1]])
                self.assertTrue(torch.all(detail == expected_detail))

            elif orientation == UP_DIAGONAL:
                expected_low_pass = torch.FloatTensor([[7,  11],
                                                       [23, 27]])
                self.assertTrue(torch.all(low_pass == expected_low_pass))

                expected_detail = torch.FloatTensor([[3, 3],
                                                     [3, 3]])
                self.assertTrue(torch.all(detail == expected_detail))

            elif orientation == DOWN_DIAGONAL:
                expected_low_pass = torch.FloatTensor([[7,  11],
                                                       [23, 27]])
                self.assertTrue(torch.all(low_pass == expected_low_pass))

                expected_detail = torch.FloatTensor([[-5, -5],
                                                     [-5, -5]])
                self.assertTrue(torch.all(detail == expected_detail))

            haar_deconv = HaarDeconv2D(orientation=orientation, in_channels=n_channels)
            deconv = haar_deconv((low_pass, detail))

            if orientation in [VERTICAL, HORIZONTAL]:
                self.assertTrue(torch.all(t == deconv))

            if orientation in [UP_DIAGONAL, DOWN_DIAGONAL]:
                self.assertTrue(torch.all(t[deconv > 0] == deconv[deconv > 0]))

    def notest_haar_conv(self):
        """
        Base test.  Run image through convolution, check that the shapes are as expected,
        that the values are correct and that deconvolution yields the original image
        """

        path = self.get_data_path()
        img1_path = os.path.join(path, 'img1.jpg')
        image = Image.open(img1_path)
        #image = ImageOps.grayscale(image)
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
                        self.assertTrue(torch.all(low_pass == original[:, 0] + original[:, 1]))
                        self.assertTrue(torch.all(detail == original[:, 0] - original[:, 1]))

                    elif orientation == HORIZONTAL:
                        original = image[0, :, row, 2 * column: (2 * column) + 2]
                        self.assertTrue(torch.all(low_pass == original[:, 0] + original[:, 1]))
                        self.assertTrue(torch.all(detail == original[:, 0] - original[:, 1]))

                    elif orientation == UP_DIAGONAL:
                        original = image[0, :, 2 * row: (2 * row) + 2, 2 * column: (2 * column) + 2]
                        self.assertTrue(torch.all(low_pass == original[:, 1, 0] + original[:, 0, 1]))
                        self.assertTrue(torch.all(detail == original[:, 1, 0] - original[:, 0, 1]))

                    elif orientation == DOWN_DIAGONAL:
                        original = image[0, :, 2 * row: (2 * row) + 2, 2 * column: (2 * column) + 2]
                        self.assertTrue(torch.all(low_pass == original[:, 0, 0] + original[:, 1, 1]))
                        self.assertTrue(torch.all(detail == original[:, 0, 0] - original[:, 1, 1]))


if __name__ == '__main__':
    unittest.main()
