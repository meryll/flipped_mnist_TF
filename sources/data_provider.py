from tensorflow.examples.tutorials.mnist import input_data
from sources.dataset import Dataset


class MnistDataProvider:

    def __init__(self):

        mnist = input_data.read_data_sets('../datasets/MNIST_data', one_hot=False)
        self.train = Dataset(mnist_dataset=mnist.train, size = 10000, shuffle=True)
        self.test = Dataset(mnist_dataset=mnist.test, shuffle=True)
        self.validation = Dataset(mnist_dataset=mnist.validation, shuffle=True)



