import csv
import json

import numpy as np
import redis
import yaml

import src.data.preprocess as preprocess
from src import project_paths


def create_pipeline(config_path):
    with config_path.open(mode='r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    pipeline_config = {}
    for parameter in config['pipe_parameters']:
        pipeline_config[parameter] = config['pipe_parameters'][parameter]
    methods = config['pipeline']
    pipeline = [getattr(preprocess, method) for method in methods]
    return pipeline, pipeline_config


def pipe(image, pipeline, pipeline_config):
    preprocessed_image = image
    for operation in pipeline:
        preprocessed_image = operation(pipeline_config, preprocessed_image)
    return preprocessed_image


def images_to_redis(redis_db, source_path, config_path):
    redis_db.set('grapheme_ids', json.dumps([]))
    pipeline, pipeline_config = create_pipeline(config_path)
    for np_file in source_path.iterdir():
        if np_file.stem.startswith('train_image_data'):

            print(np_file.stem)
            np_file.resolve().as_posix()
            data = np.load(np_file.resolve().as_posix())
            ids = data['ids']

            images = data['images']

            loaded_ids = json.loads(redis_db.get('grapheme_ids'))
            loaded_ids += ids.tolist()
            redis_db.set('grapheme_ids', json.dumps(loaded_ids))

            samples = images.shape[0]

            for sample in range(samples):
                image_id = int(ids[sample])
                image = pipe(images[sample], pipeline, pipeline_config)
                redis_db.hset(image_id, 'grapheme', image)


def labels_to_redis(redis_db, source_path):
    with source_path.open(mode='r') as label_csv:
        csv_reader = csv.reader(label_csv)

        next(csv_reader)

        for row in csv_reader:
            redis_db.hset(int(row[0].split('_')[-1]), "grapheme_label",
                          json.dumps(
                              [int(i) for i in row[1:4]] + [ord(i) for i in
                                                            row[4]]))

def push():
    paths = project_paths()
    root_path = paths['root']
    config_path = paths['config']

    with config_path.open(mode='r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    train_path = root_path / config['interim_data_dir']
    label_path = root_path / config['train_data']

    r = redis.Redis(
        host='localhost',
        port=6379
    )

    images_to_redis(r, train_path, config_path)
    labels_to_redis(r, label_path)


if __name__ == '__main__':
    push()
