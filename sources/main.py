import numpy as np
import os
import tensorflow as tf
from sources.data_provider import MnistDataProvider
from sources import model
from sources import evaluator

np.set_printoptions(threshold=np.nan)


def evaluate(classificator, dataset):

    input_fn = tf.estimator.inputs.numpy_input_fn(x=dataset.features,
                                                  shuffle=False)
    predictions = list(classificator.predict(input_fn))

    evaluator.evaluate(dataset=dataset,
                       predictions=predictions)


def train(classificator, full_dataset, batch_size, num_epochs, num_steps):
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join('../logs', 'train'),
                                             sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join('../logs', 'test'),
                                            sess.graph)

        tf.global_variables_initializer().run()

        input_fn = tf.estimator.inputs.numpy_input_fn(
            x=full_dataset.train.features,
            y=full_dataset.train.labels,
            batch_size=batch_size,
            num_epochs=num_epochs,
            shuffle=True)

        classificator.train(input_fn, steps=num_steps)
        tf.contrib.distribute.OneDeviceStrategy("device:CPU:0")

        # ------
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x=full_dataset.test.features,
            y=full_dataset.test.labels,
            num_epochs=1,
            batch_size=batch_size,
            shuffle=True)

        evaluate = classificator.evaluate(input_fn)
        print(evaluate)


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    batch_size = 32
    num_steps = 1000
    num_epochs = 10000

    full_dataset = MnistDataProvider()
    classificator = tf.estimator.Estimator(model.model_fn)

    train(classificator=classificator,
          full_dataset=full_dataset,
          num_epochs=num_epochs,
          num_steps=num_steps,
          batch_size=batch_size)

    evaluate(classificator=classificator,
             dataset=full_dataset.validation)


if __name__ == '__main__':
    main()
