import argparse
import json
import pathlib

import fastparquet as pq
import numpy as np
import pandas as pd
import redis
import yaml

from src import project_paths


def row_group_mod(row_group_size, source, destination, prefix):
    """A function to modify the row groups of the original parquet files. This
    allows for select parts of the parquet files to be loaded in memory as
    opposed to the entire file.
    """

    dir_path = pathlib.Path(source)

    if dir_path.is_dir():
        for file_path in dir_path.iterdir():
            if file_path.suffix == '.parquet':

                output_file = (f"{destination}/{file_path.stem}_rg"
                               f"{row_group_size}.parquet")

                if file_path.stem.startswith(prefix) and not pathlib.Path(
                        output_file).is_file():

                    print(f"Loading {file_path.resolve().as_posix()}.")

                    parquet = pq.ParquetFile(file_path.resolve().as_posix())
                    df = parquet.to_pandas()

                    print(f"Processing {file_path.resolve().as_posix()}.")

                    pq.write(output_file, df, row_group_size)

                    print("Complete.")

                elif pathlib.Path(output_file).is_file():

                    print(f"{file_path.resolve().as_posix()} has already"
                          "been processed.")

    return


def pq_to_np(rows, source, destination, prefix):
    """A function to save the parquet files as npy files instead. The data can
    be split into many smaller npy files by using the rows argument.
    """

    dir_path = pathlib.Path(source)
    id_prefix_len = len('Train_')

    if dir_path.is_dir():
        for file_path in dir_path.iterdir():
            if file_path.suffix == '.parquet':
                if file_path.stem.startswith(prefix):

                    print(f"Loading {file_path.resolve().as_posix()}.")

                    parquet = pq.ParquetFile(file_path.resolve().as_posix())
                    df = parquet.to_pandas()
                    df['image_id'] = df['image_id'].map(lambda x:
                                                        x[id_prefix_len:])
                    df['image_id'] = pd.to_numeric(df['image_id'],
                                                   downcast='unsigned')
                    print(df.dtypes)
                    print(f"Exporting data as .npz files.")

                    df_samples = df.shape[0]
                    processed_samples = 0
                    file_idx = 0

                    while processed_samples < df_samples:
                        file_size = min(rows, df_samples - processed_samples)
                        output_file = f"{destination}/{file_path.stem}_" \
                                      f"{file_idx}_{file_size}rows.npz"
                        np_samples = df.iloc[processed_samples:
                                             processed_samples + file_size,
                                     1:].to_numpy()
                        np_ids = df.iloc[processed_samples:
                                            processed_samples + file_size, 0
                                 ].to_numpy()
                        np.savez(output_file, ids=np_ids, images=
                        np_samples)
                        print(np_ids.shape, np_samples.shape)
                        processed_samples += rows
                        file_idx += 1

                    print("Complete.")

    return


if __name__ == '__main__':
    CONFIG_PATH = project_paths()["config"]

    with CONFIG_PATH.open(mode='r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', nargs='?', type=str, default='train')
    parser.add_argument('-p', '--parquet', action="store_true")
    parser.add_argument('-n', '--numpy', action="store_true")
    parser.add_argument('-r', '--rows', type=int, nargs='?', const=int(
        config['row_group_size']), default=int(config['row_group_size']))
    args = parser.parse_args()

    raw_dir = project_paths()["data"] / 'raw'
    interim_dir = project_paths()["data"] / 'interim'

    rows = args.rows
    prefix = args.prefix

    if args.parquet:
        row_group_mod(rows, raw_dir, interim_dir, prefix)
    if args.numpy or (not args.parquet and not args.numpy):
        pq_to_np(rows, raw_dir, interim_dir, prefix)
