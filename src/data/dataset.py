import csv
import json
import lzma
import lz4.frame
import pathlib
import random

import fastparquet
import numpy as np
import redis
import tensorflow as tf
import yaml

from src import project_paths
from sklearn.model_selection import KFold


def class_label_dicts(path_string):
    """Generate dictionaries to access the class labels corresponding
    to each part of a grapheme.
    """

    label_path = pathlib.Path(path_string)
    grapheme_roots = {}
    vowel_diacritics = {}
    consonant_diacritics = {}

    with label_path.open(mode='r') as label_csv:

        csv_reader = csv.reader(label_csv)
        next(csv_reader)

        for row in csv_reader:

            if row[0] == 'grapheme_root':
                grapheme_roots[row[2]] = row[1]

            if row[0] == 'vowel_diacritics':
                vowel_diacritics[row[2]] = row[1]

            if row[0] == 'consonant_diacritics':
                consonant_diacritics[row[2]] = row[1]

    return grapheme_roots, vowel_diacritics, consonant_diacritics


def row_group_count(directory, prefix):
    row_groups = 0
    dir_path = pathlib.Path(directory)
    print(dir_path)
    if dir_path.is_dir():
        for file_path in dir_path.iterdir():
            print(file_path)
            if file_path.stem.startswith(prefix):
                row_groups += len(fastparquet.ParquetFile(file_path.resolve(
                    0).as_posix()).row_groups)

    return row_groups


def data_generator(redis_db, ids):
    for image_id in ids:
        image_bytes = lz4.frame.decompress(redis_db.hget(image_id, 'grapheme'))
        label = json.loads(redis_db.hget(image_id, 'grapheme_label').decode())
        image = np.array(json.loads(image_bytes.decode()))
        one_hot = np.zeros(168 + 11 + 7, dtype=float)

        one_hot[label[0]] = 1.0
        one_hot[168 + label[1]] = 1.0
        one_hot[168 + 11 + label[2]] = 1.0

        image_id = tf.convert_to_tensor(image_id, dtype=tf.int32)
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = image[...,tf.newaxis]
        one_hot = tf.convert_to_tensor(one_hot, dtype=tf.float32)
        #print(tf.shape(one_hot))
        # print(tf.shape(image))
        yield image_id, image, one_hot


class KFoldDataset:

    def __init__(self, redis_db, folds):
        self.train_folds = []
        self.test_folds = []
        ids = json.loads(redis_db.get('grapheme_ids'))
        random.shuffle(ids)
        kf = KFold(n_splits=folds)
        output_type = (tf.int32, tf.float32, tf.float32)
        for train_idx, test_ids in kf.split(ids):
            ds1 = data_generator(redis_db, [ids[idx] for idx in train_idx])
            ds2 = data_generator(redis_db, [ids[idx] for idx in train_idx])
            dsg1 = tf.data.Dataset.from_generator(lambda: ds1, output_type)
            dsg2 = tf.data.Dataset.from_generator(lambda: ds2, output_type)
            self.train_folds.append(dsg1)
            self.test_folds.append(dsg2)


if __name__ == '__main__':
    paths = project_paths()
    root_path = paths['root']
    config_path = paths['config']

    with config_path.open(mode='r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    r = redis.Redis(
        host='localhost',
        port=6379
    )
    ids = json.loads(r.get('grapheme_ids'))
    dataset = data_generator(r, ids)

    # for x in range(4):
    #     print(next(dataset))
    out_type = (tf.int32, tf.float32, tf.float32)
    tfds = tf.data.Dataset.from_generator(lambda: dataset, out_type)

    kfold = KFoldDataset(r, 2)
    print(kfold.train_folds)
    for ds in kfold.train_folds:
        print(type(ds))
        for x in ds.batch(3):
            print(x)
