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
            ma = self.simple_ma - (pop - x)/self.window
        else:
            ma = (self.simple_ma * len(self.queue) + x) / (len(self.queue) + 1)
        self.simple_ma = ma
        self.queue.append(x)


@tf.function
def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)

        loss_1 = loss_object(labels[:, :168], predictions[:, :168])
        loss_2 = loss_object(labels[:, 168:179], predictions[:, 168:179])
        loss_3 = loss_object(labels[:, 179:], predictions[:, 179:])

        l_cat = tf.stack([loss_1, loss_2, loss_3])
        l_mean = tf.math.reduce_mean(l_cat, 1)
        loss = tf.nn.weighted_moments(l_mean, 0, [0.5, .25, .25])

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


if __name__ == '__main__':

    paths = project_paths()
    root = paths['root']
    config_path = paths['config']

    print(config_path)
    with config_path.open(mode='r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    epochs = config['epochs']
    folds = config['folds']
    batch_size = config['batch_size']
    tf.debugging.set_log_device_placement(True)
    print("Num GPUs Available: ",
          len(tf.config.experimental.list_physical_devices('GPU')))

    r = redis.Redis(
        host='localhost',
        port=6379
    )

    kfolds = dataset.KFoldDataset(r, folds)
    train_folds = kfolds.train_folds
    test_folds = kfolds.test_folds

    cnn = model.BengaliCNN()

    loss_object = tf.nn.softmax_cross_entropy_with_logits
    optimizer = tf.keras.optimizers.Adam()
    rate_ma = MovingAverage(50)
    for fold in range(folds):
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
