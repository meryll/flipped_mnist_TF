from sources.utils import dataset_utils

class Dataset:

    def __init__(self, mnist_dataset, size=None, shuffle=False):
        images = mnist_dataset.images
        digits = mnist_dataset.labels

        if size is not None:
            images = images[:size]
            digits = digits[:size]

        all_images, all_digits, all_axis, all_labels = dataset_utils.flip(images=images, digits=digits)

        if shuffle:
            all_images, all_digits, all_axis, all_labels = dataset_utils.shuffle(images=all_images,
                                                                                 digits=all_digits,
                                                                                 axis=all_axis,
                                                                                  labels=all_labels)

        self.features = {'images': all_images, 'digits': all_digits, 'axis': all_axis}
        self.labels = all_labels

    def get_label(self, i):
        return self.labels[i]

    def get_digit(self, i):
        return self.features['digits'][i]

    def get_angle(self, i):
        return self.features['axis'][i]

    def get_image(self, i):
        return self.features['images'][i]
