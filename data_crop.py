import numpy as np
import tensorflow as tf
import librosa as lb
import os
import seaborn as sn
import matplotlib.pyplot as plt

def extract_ts(base_directory, start_time=30, duration=31):

    class_list = sorted(os.listdir(base_directory))
    n_classes = len(class_list)

    train_index = 0
    test_index = 0

    base, dir = os.path.split(base_directory)

    if not os.path.exists(os.path.join(base, dir + '_cropped')):
        os.mkdir(os.path.join(base, dir + '_cropped'))

    for (gi, genre) in enumerate(class_list):

        path = os.path.join(base_directory, genre)

        if not os.path.exists(os.path.join(base, dir + '_cropped', genre)):
            os.mkdir(os.path.join(base, dir + '_cropped', genre))

        for (si, song) in enumerate(os.listdir(path)):

            print(' * {} {} * '.format(genre, song))

            song_timeseries = lb.load(os.path.join(path, song), offset=start_time, duration=duration)

            sampling_rate = song_timeseries[1]

            song_timeseries = song_timeseries[0].astype(np.float32)

            print(sampling_rate)
            print(song_timeseries.shape)

            lb.output.write_wav(os.path.join(base, dir + '_cropped', genre, song),
                                song_timeseries, sampling_rate, norm=False)

if __name__ == '__main__':

    extract_ts('../../indian')

    print('oo')

