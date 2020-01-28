import collections
import time
from src.models import model
from src.data import dataset
from src import project_paths

import redis
import tensorflow as tf
import yaml


class MovingAverage:
    def __init__(self, window):
        self.window = window
        self.queue = collections.deque()
        self.simple_ma = 0

    def add(self, x):
        if len(self.queue) >= self.window:
            pop = self.queue.popleft()
            ma = self.simple_ma - (pop - x) / self.window
        else:
            ma = (self.simple_ma * len(self.queue) + x) / (len(self.queue) + 1)
        self.simple_ma = ma
        self.queue.append(x)


@tf.function
def test_step(model, images, labels, batch_loss):
    loss = forward_pass(model, images, labels)
    batch_loss.assign(loss)
    return batch_loss


@tf.function
def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        loss = forward_pass(model, images, labels)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def forward_pass(model, images, labels):
    predictions = model(images)

    loss_1 = loss_object(labels[:, :168], predictions[:, :168])
    loss_2 = loss_object(labels[:, 168:179], predictions[:, 168:179])
    loss_3 = loss_object(labels[:, 179:], predictions[:, 179:])

    l_cat = tf.stack([loss_1, loss_2, loss_3])
    l_mean = tf.math.reduce_mean(l_cat, 1)
    print(l_mean.shape)
    loss, _ = tf.nn.weighted_moments(l_mean, 0, [0.5, .25, .25])
    return loss


if __name__ == '__main__':

    # Load config file.
    paths = project_paths()
    root = paths['root']
    config_path = paths['config']

    with config_path.open(mode='r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Get train/test parameters.
    epochs = config['epochs']
    folds = config['folds']
    batch_size = config['batch_size']

    # Uncomment to verify what devices operations are occuring on:
    # tf.debugging.set_log_device_placement(True)

    print("Num GPUs Available: ",
          len(tf.config.experimental.list_physical_devices('GPU')))

    # Create training and testing folds for K-fold cross-validation.
    r = redis.Redis(
        host='localhost',
        port=6379
    )

    kfolds = dataset.KFoldDataset(r, folds)
    train_folds = kfolds.train_folds
    test_folds = kfolds.test_folds


    # Define loss function and optimizer.
    loss_object = tf.nn.softmax_cross_entropy_with_logits
    optimizer = tf.keras.optimizers.Adam()

    rate_ma = MovingAverage(50)

    # Begin K-fold cross-validation.
    for fold in range(folds):
        cnn = model.BengaliCNN()
        train_dataset = train_folds[fold]
        test_dataset = train_folds[fold]
        for epoch in range(epochs):

            processed = 0
            for idx, minibatch in enumerate(train_dataset.batch(batch_size)):
                t_start = time.perf_counter()
                ids, images, labels = minibatch
                train_step(cnn, images, labels)
                processed += len(ids)
                t_end= time.perf_counter()
                rate_ma.add(len(ids)/(t_end - t_start))

                print(rate_ma.simple_ma)
                print(f"Fold: {fold}, Epoch: {epoch}, Processed: {processed}")

        batch_loss = tf.Variable(0.0, dtype=tf.float32)
        processed_samples = 0
        error = 0
        for idx, minibatch in enumerate(test_dataset.batch(batch_size)):
            test_ids, test_images, test_labels = minibatch
            samples = len(test_ids)
            test_step(cnn, test_images, test_labels, batch_loss)
            error = (error * processed_samples + batch_loss.numpy()) / (
                    processed_samples + samples)
            processed_samples += samples
            print(error)

