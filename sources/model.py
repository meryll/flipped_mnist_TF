import tensorflow as tf


def build(x, n_classes, dropout, reuse, is_training):
    with tf.variable_scope('ConvNet', reuse=reuse):
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        fc1 = tf.contrib.layers.flatten(conv2)
        fc1 = tf.layers.dense(fc1, 1024)

        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
        out = tf.layers.dense(fc1, n_classes)

    return out


def model_fn(features, labels, mode):
    x = features['images']
    number_of_classes = 2
    dropout = 0.7
    learning_rate = 0.001

    logits_train = build(x, number_of_classes, dropout, reuse=False, is_training=True)
    logits_test = build(x, number_of_classes, dropout, reuse=True, is_training=False)

    prediction_classes = tf.argmax(logits_test, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=prediction_classes)

    loss = _get_loss(logits=logits_train,
                     labels=labels)

    optimizer = _get_optimizer(learning_rate=learning_rate)
    train_optimizer = optimizer.minimize(loss=loss,
                                         global_step=tf.train.get_global_step())

    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=prediction_classes)

    # tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('loss', loss)

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=prediction_classes,
                                      loss=loss,
                                      train_op=train_optimizer,
                                      eval_metric_ops={'accuracy': accuracy})


def _get_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                         labels=tf.cast(labels, dtype=tf.int32)))


def _get_optimizer(learning_rate):
    return tf.train.AdamOptimizer(learning_rate=learning_rate)
