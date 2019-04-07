import numpy as np
import tensorflow as tf
import librosa as lb
import os
import seaborn as sn
import matplotlib.pyplot as plt

def extract_ts(base_directory, train_ratio = .7, time_limit=661794):

    class_list = sorted(os.listdir(base_directory))
    n_classes = len(class_list)

    n_songs_train = 0
    n_songs_test = 0

    if time_limit == None:
        temp_path = os.path.join(base_directory, class_list[0])
        random_file_name = os.listdir(temp_path)[0]
        random_file_path = lb.load(os.path.join(temp_path, random_file_name))[0]
        time_limit = random_file_path.shape[0]

        print(time_limit)

    n_songs_train_per_class = [0] * n_classes
    n_songs_test_per_class = [0] * n_classes

    for gi, genre in enumerate(class_list):
        path = os.path.join(base_directory, genre)
        count_class = len(os.listdir(path))
        count_train = int(train_ratio *  count_class)
        count_test = count_class - count_train

        n_songs_train_per_class[gi] = count_train
        n_songs_test_per_class[gi] = count_test

        n_songs_train += count_train
        n_songs_test += count_test

    X_train = np.zeros((n_songs_train, time_limit), np.float32)
    Y_train = np.zeros((n_songs_train), np.float32)

    X_test = np.zeros((n_songs_test, time_limit), np.float32)
    Y_test = np.zeros((n_songs_test), np.float32)

    train_index = 0
    test_index = 0

    for (gi, genre) in enumerate(class_list):

        path = os.path.join(base_directory, genre)

        for (si, song) in enumerate(os.listdir(path)):

            print(' * {} {} * '.format(genre, song))

            song_timeseries = lb.load(os.path.join(path, song))[0].astype(np.float32)

            # TODO changed some shit

            if song_timeseries.shape[0] > time_limit:
                song_timeseries = song_timeseries[: time_limit]

            elif song_timeseries.shape[0] < time_limit:

                while song_timeseries.shape[0] < time_limit:
                    song_timeseries = np.hstack((song_timeseries, song_timeseries))

                song_timeseries = song_timeseries[:time_limit]

            if si < n_songs_train_per_class[gi]:
                X_train[train_index] = song_timeseries
                Y_train[train_index] = gi

                train_index += 1

            else:
                X_test[test_index] = song_timeseries
                Y_test[test_index] = gi

                test_index += 1

    np.save('X_train.npy', X_train)
    np.save('Y_train.npy', Y_train)
    np.save('X_test.npy', X_test)
    np.save('Y_test.npy', Y_test)

    return X_train, X_test, Y_train, Y_test

if __name__ == '__main__':

    X_train, X_test, Y_train, Y_test = extract_ts('../more_data')

    print('oo')

