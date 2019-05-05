import numpy as np
import tensorflow as tf
import librosa as lb
import os
import seaborn as sn
import sklearn
import librosa.display
import matplotlib.pyplot as plt

def extract_ts_multiple_binned_test(base_directory, save_path, train_ratio = .7, time_limit=661794, train=True):
    ori_base = base_directory

    if train:
        base_directory = os.path.join(ori_base, 'train')

        class_list = sorted(os.listdir(base_directory))

        if time_limit == None:
            temp_path = os.path.join(base_directory, class_list[0])
            random_file_name = os.listdir(temp_path)[0]
            random_file_path = lb.load(os.path.join(temp_path, random_file_name))[0]
            time_limit = random_file_path.shape[0]

            print(time_limit)

        X = []
        Y = []
        segment_count = []

        for (gi, genre) in enumerate(class_list):

            path = os.path.join(base_directory, genre)

            for (si, song) in enumerate(os.listdir(path)):

                try:

                    song_timeseries = lb.load(os.path.join(path, song))[0].astype(np.float32)
                    length = song_timeseries.shape[0]

                    for sid in range(length // time_limit):
                        song_timeseries_cropped = song_timeseries[time_limit * sid: time_limit * sid + time_limit]

                        X.append(song_timeseries_cropped)
                        Y.append(gi)
                        print(sid)

                    segment_count.append(sid + 1)

                    print(' * {} {} * '.format(genre, song))
                except:
                    print('couldnt load song')

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        segment_count = np.array(segment_count, dtype=np.float32)

        print(X, X.shape)
        print(Y, Y.shape)
        print(segment_count, segment_count.shape)

        np.save(save_path + '_X_train.npy', X)
        np.save(save_path + '_Y_train.npy', Y)
        np.save(save_path + '_segment_count_train', segment_count)

    ############################################################################################

    base_directory = os.path.join(ori_base, 'test')

    class_list = sorted(os.listdir(base_directory))

    if time_limit == None:
        temp_path = os.path.join(base_directory, class_list[0])
        random_file_name = os.listdir(temp_path)[0]
        random_file_path = lb.load(os.path.join(temp_path, random_file_name))[0]
        time_limit = random_file_path.shape[0]

        print(time_limit)

    X = []
    Y = []
    segment_count = []

    for (gi, genre) in enumerate(class_list):

        path = os.path.join(base_directory, genre)

        for (si, song) in enumerate(os.listdir(path)):

            try:

                song_timeseries = lb.load(os.path.join(path, song))[0].astype(np.float32)
                length = song_timeseries.shape[0]

                for sid in range(length // time_limit):
                    song_timeseries_cropped = song_timeseries[time_limit * sid: time_limit * sid + time_limit]

                    X.append(song_timeseries_cropped)
                    Y.append(gi)
                    print(sid)

                segment_count.append(sid + 1)

                print(' * {} {} * '.format(genre, song))
            except:
                print('couldnt load song')

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    segment_count = np.array(segment_count, dtype=np.float32)

    print(X, X.shape)
    print(Y, Y.shape)
    print(segment_count, segment_count.shape)

    np.save(save_path + '_X_test.npy', X)
    np.save(save_path + '_Y_test.npy', Y)
    np.save(save_path + '_segment_count_test', segment_count)

def extract_ts_multiple(base_directory, train_ratio = .7, time_limit=661794):

    ori_base = base_directory
    base_directory = os.path.join(ori_base, 'train')

    class_list = sorted(os.listdir(base_directory))

    if time_limit == None:
        temp_path = os.path.join(base_directory, class_list[0])
        random_file_name = os.listdir(temp_path)[0]
        random_file_path = lb.load(os.path.join(temp_path, random_file_name))[0]
        time_limit = random_file_path.shape[0]

        print(time_limit)

    X = []
    Y = []

    for (gi, genre) in enumerate(class_list):

        path = os.path.join(base_directory, genre)

        for (si, song) in enumerate(os.listdir(path)):

            try:

                song_timeseries = lb.load(os.path.join(path, song))[0].astype(np.float32)
                length = song_timeseries.shape[0]

                for sid in range(length // time_limit):
                    print(sid)
                    song_timeseries_cropped = song_timeseries[time_limit*sid: time_limit*sid + time_limit]

                    X.append(song_timeseries_cropped)
                    Y.append(gi)

                print(' * {} {} * '.format(genre, song))
            except:
                print('couldnt load song')


    X = np.vstack(X).astype(np.float32)
    Y = np.array(Y, dtype=np.float32)

    print(X, X.shape)
    print(Y, Y.shape)

    np.save('data/indian_4_sana_segmented_X_train.npy', X)
    np.save('data/indian_4_sana_segmented_Y_train.npy', Y)

    ############################################################################################

    base_directory = os.path.join(ori_base, 'test')

    class_list = sorted(os.listdir(base_directory))

    if time_limit == None:
        temp_path = os.path.join(base_directory, class_list[0])
        random_file_name = os.listdir(temp_path)[0]
        random_file_path = lb.load(os.path.join(temp_path, random_file_name))[0]
        time_limit = random_file_path.shape[0]

        print(time_limit)

    X = []
    Y = []

    for (gi, genre) in enumerate(class_list):

        path = os.path.join(base_directory, genre)

        for (si, song) in enumerate(os.listdir(path)):

            try:

                song_timeseries = lb.load(os.path.join(path, song))[0].astype(np.float32)
                length = song_timeseries.shape[0]

                for sid in range(length // time_limit):
                    song_timeseries_cropped = song_timeseries[time_limit * sid: time_limit * sid + time_limit]

                    X.append(song_timeseries_cropped)
                    Y.append(gi)

                print(' * {} {} * '.format(genre, song))
            except:
                print('couldnt load song')

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    print(X, X.shape)
    print(Y, Y.shape)

    np.save('data/indian_4_sana_segmented_X_test.npy', X)
    np.save('data/indian_4_sana_segmented_Y_test.npy', Y)


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



            try:

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

                print(' * {} {} * '.format(genre, song))
            except:
                print('couldnt load song')


    np.save('data/esc_X_train.npy', X_train)
    np.save('data/esc_Y_train.npy', Y_train)
    np.save('data/esc_X_test.npy', X_test)
    np.save('data/esc_Y_test.npy', Y_test)

    return X_train, X_test, Y_train, Y_test



if __name__ == '__main__':

    # X_train, X_test, Y_train, Y_test = extract_ts('esc')

    # extract_ts_multiple('../indian_4_sana_segmented')

    # extract_ts_multiple_binned_test('../indian_4_sana_segmented', 'data/indian_4_sana_segmented', train=True)
    # extract_ts_multiple_binned_test('indian_fake', 'data/indian_4_sana_segmented', train=False)

    print('oo')


print('oo')
base_directory = '/home/sam/storage/fyp/indian_4_sana_segmented/test'

song1 = lb.load(os.path.join(base_directory, 'rahman', 'checkg24.wav'))[0]

print(song1.shape)


# S = lb.feature.melspectrogram(song1)
S = np.abs(lb.core.stft(song1))

plt.figure(figsize=(10, 4))
lb.display.specshow(librosa.amplitude_to_db(S,
                          ref=np.max),
                          y_axis='linear', fmax=8000,
                          x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear frequency spectrogram')
plt.tight_layout()

plt.savefig('rahman_stft.pdf')
plt.show()

