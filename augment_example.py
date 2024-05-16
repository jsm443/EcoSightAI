import numpy as np
import random

import torch
import scipy


class Shift(object):
    """
    Shifts input image by random x amount between [-max_shift, max_shift]
      and separate random y amount between [-max_shift, max_shift]. A positive
      shift in the x- and y- direction corresponds to shifting the image right
      and downwards, respectively.

      Inputs:
          max_shift  float; maximum magnitude amount to shift image in x and y directions.
    """

    def __init__(self, max_shift=10):
        self.max_shift = max_shift

    def __call__(self, image):
        """
        Inputs:
            image         3 x H x W image as torch Tensor

        Returns:
            shift_image   3 x H x W image as torch Tensor, shifted by random x
                          and random y amount, each amount between [-max_shift, max_shift].
                          Pixels outside original image boundary set to 0 (black).
        """
        image = image.numpy()
        _, H, W = image.shape
        # TODO: Shift image
        # TODO-BLOCK-BEGIN
        shift_x = random.randint(-self.max_shift, self.max_shift)
        shift_y = random.randint(-self.max_shift, self.max_shift)

        # Create shifted image array
        shift_image = np.zeros_like(image)

        # Compute the shifted starting and ending indices
        start_x = max(0, shift_x)
        end_x = min(W, W + shift_x)
        start_y = max(0, shift_y)
        end_y = min(H, H + shift_y)

        # Compute the corresponding region of the original image
        orig_start_x = max(0, -shift_x)
        orig_end_x = min(W, W - shift_x)
        orig_start_y = max(0, -shift_y)
        orig_end_y = min(H, H - shift_y)

        # Assign the region of the original image to the shifted image
        shift_image[:, start_y:end_y, start_x:end_x] = image[
            :, orig_start_y:orig_end_y, orig_start_x:orig_end_x
        ]

        # TODO-BLOCK-END

        return torch.Tensor(shift_image)

    def __repr__(self):
        return self.__class__.__name__


class Contrast(object):
    """
    Randomly adjusts the contrast of an image. Uniformly select a contrast factor from
    [min_contrast, max_contrast]. Setting the contrast to 0 should set the intensity of all pixels to the
    mean intensity of the original image while a contrast of 1 returns the original image.

    Inputs:
        min_contrast    non-negative float; minimum magnitude to set contrast
        max_contrast    non-negative float; maximum magnitude to set contrast

    Returns:
        image        3 x H x W torch Tensor of image, with random contrast
                     adjustment
    """

    def __init__(self, min_contrast=0.3, max_contrast=1.0):
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast

    def __call__(self, image):
        """
        Inputs:
            image         3 x H x W image as torch Tensor

        Returns:
            shift_image   3 x H x W torch Tensor of image, with random contrast
                          adjustment
        """
        image = image.numpy()
        _, H, W = image.shape

        # TODO: Change image contrast
        # TODO-BLOCK-BEGIN
        # Compute mean intensity of the original image
        mean_intensity = np.mean(image)

        # Generate random contrast factor
        contrast_factor = random.uniform(self.min_contrast, self.max_contrast)

        # Adjust contrast of the image
        adjusted_image = (image - mean_intensity) * contrast_factor + mean_intensity

        # Clip the image intensities between 0 and 255
        adjusted_image = np.clip(adjusted_image, 0, 255)

        # TODO-BLOCK-END

        return torch.Tensor(adjusted_image)

    def __repr__(self):
        return self.__class__.__name__


class Rotate(object):
    """
    Rotates input image by random angle within [-max_angle, max_angle]. Positive angle corresponds to
    counter-clockwise rotation

    Inputs:
        max_angle  maximum magnitude of angle rotation, in degrees


    """

    def __init__(self, max_angle=10):
        self.max_angle = max_angle

    def __call__(self, image):
        """
        Inputs:
            image           image as torch Tensor

        Returns:
            rotated_image   image as torch Tensor; rotated by random angle
                            between [-max_angle, max_angle].
                            Pixels outside original image boundary set to 0 (black).
        """
        image = image.numpy()
        _, H, W = image.shape

        # TODO: Rotate image
        # TODO-BLOCK-BEGIN
        # Generate random rotation angle
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        rotated_image = scipy.ndimage.rotate(
            image, angle, axes=(1, 2), reshape=False, order=1, mode="constant", cval=0
        )
        # TODO-BLOCK-END

        return torch.Tensor(rotated_image)

    def __repr__(self):
        return self.__class__.__name__


class HorizontalFlip(object):
    """
    Randomly flips image horizontally.

    Inputs:
        p          float in range [0,1]; probability that image should
                   be randomly rotated
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        """
        Inputs:
            image           image as torch Tensor

        Returns:
            flipped_image   image as torch Tensor flipped horizontally with
                            probability p, original image otherwise.
        """
        image = image.numpy()
        _, H, W = image.shape

        # TODO: Flip image
        # TODO-BLOCK-BEGIN
        flip_prob = random.random()
        if flip_prob < self.p:
            flipped_image = np.flip(image, axis=2).copy()
        else:
            flipped_image = image.copy()
        # TODO-BLOCK-END

        return torch.Tensor(flipped_image)

    def __repr__(self):
        return self.__class__.__name__
