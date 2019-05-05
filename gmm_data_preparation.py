import os
import librosa as lb
import pickle
import numpy as np
import matplotlib.pyplot as pp


def extract_MFCC(base_directory, hop_length=512, n_freq=13, sampling_rate=22050):

    genres = []

    songs = {}

    for dir in os.listdir(base_directory):
        genres.append(dir)

        path = os.path.join(base_directory, dir)

        print(os.listdir(path))

        songs[dir] = []

        for song in os.listdir(path):
            song_timeseries = lb.load(os.path.join(path, song))[0]
            mfcc = lb.feature.mfcc(y=song_timeseries, sr=sampling_rate,  n_mfcc=n_freq, hop_length = 512)

            mfcc = mfcc.astype(np.int32)

            songs[dir].append(mfcc)

    with open('song_mfcc.pkl', 'wb') as file:
        pickle.dump(songs, file)


def extract_stft(base_directory, max_amplitude=300, hop_length=512, n_freq=13, sampling_rate=22050):

    genres = []

    songs = {}

    max = 0
    min = 1e5

    for dir in os.listdir(base_directory):
        genres.append(dir)

        path = os.path.join(base_directory, dir)

        print(os.listdir(path))

        songs[dir] = []

        for song in os.listdir(path):
            song_timeseries = lb.load(os.path.join(path, song))[0]
            stft = lb.core.stft(y=song_timeseries)

            stft = np.abs(stft).astype(np.float32)

            temp_max = np.max(stft)

            if temp_max > max:
                max = temp_max

            songs[dir].append(stft)

    real_songs = {}

    print(max)

    for dir in songs.keys():

        real_songs[dir] = []

        for song in songs[dir]:
            # song /= max
            # song *= max_amplitude
            #
            # song = song.astype(np.int32)
            real_songs[dir].append(song)


    with open('song_stft.pkl', 'wb') as file:
        pickle.dump(real_songs, file)


if __name__ == '__main__':
    extract_stft('esc')

    with open('song_stft.pkl', 'rb') as file:
        songs = pickle.load(file)

    for genre in songs.keys():

        # print(songs[genre][0].shape)

        print('max ', np.max(songs[genre][0]))
        print('min ', np.min(songs[genre][0]))

        print(songs[genre][0])

        print(songs[genre][0][:,100].shape)
        pp.bar(list(range(1,1026)), songs[genre][0][:,0])
        pp.xlabel('Frequency bins', fontsize=20)
        pp.ylabel('Amplitude', fontsize=20)
        pp.savefig('freq_hist_' + genre + '.png')
        pp.show()
    #
    #     print(type(songs[genre][0][0][0]))
    #
    #     print('====================================================')
    #
    # pp.show()
    # pp.imsave('plot_1')