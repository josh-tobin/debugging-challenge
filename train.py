import tensorflow as tf
import click
from dataset import MNISTTrain, MNISTTest
import numpy as np

def lenet(input, scope='lenet', reuse=False, training=True):
    with tf.variable_scope(scope, reuse=reuse):
        conv1 = tf.layers.conv2d(inputs=input,
                                 filters=32,
                                 kernel_size=[5, 5],
                                 padding='same',
                                 activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5],
                                 padding='same', activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=512, activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, rate=0.25, training=training)
        logits = tf.layers.dense(inputs=dense, units=10)
    return logits

def train(n_epochs, learning_rate, batch_size):
    train_dataset = MNISTTrain(batch_size)
    test_dataset = MNISTTest(batch_size)
    
    x, y = train_dataset.next_batch
    y_pred = lenet(x)
    train_loss = tf.losses.softmax_cross_entropy(y, y_pred)
    train_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1),
                                                tf.argmax(y_pred, 1)),
                                       tf.float32))

    opt = tf.train.AdamOptimizer(learning_rate)
    train_op = opt.minimize(train_loss)

    x_test, y_test = test_dataset.next_batch
    y_pred_test = lenet(x_test, reuse=True, training=False)
    test_loss = tf.losses.softmax_cross_entropy(y_test, y_pred_test)
    test_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, 1),
                                               tf.argmax(y_pred_test, 1)),
                                       tf.float32))

    def eval_test(sess):
        test_dataset.init(sess)
        errs = []
        accs = []
        try:
            while True:
                err, acc = sess.run([test_loss, test_acc])
                errs.append(err)
                accs.append(acc)
        except tf.errors.OutOfRangeError:
            return np.mean(errs), np.mean(accs) 

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            train_dataset.init(sess)
            epoch_err = []
            epoch_acc = []
            try:
                while True:
                    _, err, acc = sess.run([train_op, train_loss, train_acc])
                    epoch_err.append(err)
                    epoch_acc.append(acc)
            except tf.errors.OutOfRangeError:
                print(f"Epoch {epoch}:")
                print(f"  - Train err: {np.mean(epoch_err)}")
                print(f"  - Train acc: {np.mean(epoch_acc)}")
                # BUG: Overwrite test_err and test_acc here
                epoch_test_err, epoch_test_acc = eval_test(sess)
                print(f"  - Test err: {epoch_test_err}")
                print(f"  - Test acc: {epoch_test_acc}")

@click.command()
@click.option('--n-epochs', type=int, default=5)
@click.option('--lr', type=float, default=3e-4)
@click.option('--batch-size', type=int, default=32)
def main(n_epochs, lr, batch_size):
    train(n_epochs, lr, batch_size)

if __name__ == '__main__':
    main()
