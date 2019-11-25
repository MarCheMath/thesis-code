"""Some utils for celebA dataset"""

import png
import dcgan_utils
import numpy as np
import utils
import matplotlib.pyplot as plt

def display_transform(image):
    image = dcgan_utils.inverse_transform(image)
    return image


def view_image(image, hparams, mask=None):
    """Process and show the image"""
    image = display_transform(image)
    if len(image) == hparams.n_input:
        image = image.reshape(hparams.image_shape)
        if mask is not None:
            mask = mask.reshape(hparams.image_shape)
            image = np.maximum(np.minimum(1.0, image - 1.0*image*(1-mask)), -1.0)
    utils.plot_image(image)


def save_image(image, path):
    """Save an image as a png file"""
#    plt.figure()
#    plt.imshow(255*image)
#    plt.savefig(path)
    image = dcgan_utils.inverse_transform(image)
#    image=(100*image+1)/2.
    png_writer = png.Writer(64, 64)
    with open(path, 'wb') as outfile:
        png_writer.write(outfile, 255*image.reshape([64,-1]))

