import argparse
import numpy as np
import os
import pandas as pd


def generate_train_val_test(seq_length_xy = 24,traffic_df_filename='data/traffic/metr-la.h5'):
    df = pd.read_hdf(traffic_df_filename)
    xy_offsets = np.sort(np.concatenate((np.arange(-(seq_length_xy - 1), 1, 1),)))

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    feature_list = [data]

    data = np.concatenate(feature_list, axis=-1)
    xy = []
    min_t = abs(min(xy_offsets))
    max_t = abs(num_samples - abs(max(xy_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        xy.append(data[t + xy_offsets, ...])
    xy = np.stack(xy, axis=0)

    print("xy shape of all samples: ", xy.shape)
    # Write the data into npz file.
    num_samples = xy.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train
    xy_train = xy[:num_train]
    xy_val = xy[num_train: num_train + num_val]
    xy_test = xy[-num_test:]

    return xy_train,xy_val,xy_test


if __name__ == "__main__":

    xy_train,xy_val,xy_test = generate_train_val_test()
    print(xy_train.shape)
