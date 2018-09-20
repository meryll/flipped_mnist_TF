import numpy as np
from scipy import ndimage
from sources.utils import utils


def shuffle(images, digits, axis, labels):
    permutation = np.random.permutation(labels.shape[0])

    labels = labels[permutation]
    images = images[permutation]
    axis = axis[permutation]
    digits = digits[permutation]

    return images, digits, axis, labels


def flip(images, digits):

    labels = np.zeros((images.shape[0]))
    plain_axis = np.full(shape=(images.shape[0]),fill_value=-1)

    augmented_images = []
    augmented_axis = []
    augmented_digits = []

    for i, image in enumerate(images):
        image = np.reshape(image, (-1, 28))
        axis = utils.get_random_axis()
        new_image = np.flip(image, axis=axis)

        augmented_images.append(np.reshape(new_image, 784))
        augmented_axis.append(axis)
        augmented_digits.append(digits[i])

    augmented_images = np.array(augmented_images)
    augmented_axis = np.array(augmented_axis)
    augmented_digits = np.array(augmented_digits)
    augmented_labels = np.ones((augmented_images.shape[0]))

    all_labels = np.hstack((labels, augmented_labels))
    all_digits = np.hstack((digits, augmented_digits))
    all_angles = np.hstack((plain_axis, augmented_axis))
    all_images = np.vstack((images, augmented_images))

    return all_images, all_digits, all_angles, all_labels
