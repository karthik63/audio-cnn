import numpy
from PIL import Image
import librosa as lb
import os
import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.mixture import *
from statistics import mean
from iteration_utilities import deepflatten

import pickle

# base_directory = '../mydata'
#
# song1 = lb.load(os.path.join(base_directory, 'metal', 'metal.00000.au'))[0]
# song2 = lb.load(os.path.join(base_directory, 'metal', 'metal.00001.au'))[0]
# song3 = lb.load(os.path.join(base_directory, 'hiphop', 'hiphop.00000.au'))[0]
# song4 = lb.load(os.path.join(base_directory, 'hiphop', 'hiphop.00001.au'))[0]
#
# print(song1.shape)
# print(song2.shape)
#
# spect1 = lb.feature.melspectrogram(song1)
# spect2 = lb.feature.melspectrogram(song2)
# spect3 = lb.feature.melspectrogram(song3)
# spect4 = lb.feature.melspectrogram(song4)
#
# img = sn.heatmap(spect1, robust=True)
# plt.show()
#
#
# img = sn.heatmap(spect2, robust=True)
# plt.show()
#
# img = sn.heatmap(spect3, robust=True)
# plt.show()
#
# img = sn.heatmap(spect4, robust=True)
# plt.show()

class Model():

    def __init__(self, prepare_data=False, hop_len=None, n_classes=4, n_bins=513, n_t=1293, n_freq=9):

        self.prepare_data = prepare_data
        self.n_classes = n_classes
        self.n_bins = n_bins
        self.n_t = n_t
        self.de = None
        self.genre_to_index = {}
        self.index_to_genre = None
        self.n_freq = n_freq

        name = 'data/esc'
        # name = 'data/esc'



        self.X_train = np.load(name + '_X_train.npy')
        self.X_test = np.load(name + '_X_test.npy')

        self.Y_train = np.load(name + '_Y_train.npy')
        self.Y_test = np.load(name + '_Y_test.npy')

        p = numpy.random.permutation(len(self.Y_test))

        self.X_test = self.X_test[p]
        self.Y_test = self.Y_test[p]

    def build_model(self):

        # print(self.dataset_train['jazz'][0].shape)

        frequency_tally = np.zeros([self.n_classes, self.n_bins], np.float32)

        to_fit_time = [[[[] for _ in range(self.n_bins)] for _ in range(self.n_freq)]for _ in range(self.n_classes)]
        to_fit_freq = [[[[] for _ in range(self.n_bins)] for _ in range(self.n_freq)]for _ in range(self.n_classes)]
        to_fit_amplitude = [[[[] for _ in range(self.n_bins)] for _ in range(self.n_freq)] for _ in range(self.n_classes)]

        for ci in range(self.n_classes):
            print(ci)
            for si, song in enumerate(self.X_train):

                print(ci,si)
                if self.Y_train[si] == ci:

                    stft = lb.core.stft(y=song, n_fft=1024)

                    song = np.abs(stft).astype(np.float32)

                    argsort = np.argsort(-song, 0)[:self.n_freq, :]

                    dominant_frequencies = np.transpose(argsort)
                    dominant_amplitudes = np.transpose(-np.sort(-song, 0)[:self.n_freq, :])

                    # print(dominant_amplitudes)

                    for ti in range(dominant_frequencies.shape[0] - 1):
                        frequency_tally[ci, dominant_frequencies[ti][0]] += 1

                        for fi in range(self.n_freq - 1):

                            # if dominant_frequency1[i] == 186:
                            #     print(dominant_frequency1[i+1])

                            # print(dominant_frequency1[i], dominant_frequency1[i+1])

                            # print(to_fit[k][dominant_frequency1[i]])

                            # print(to_fit[k][dominant_frequency1[i]])

                            to_fit_time[ci][fi][dominant_frequencies[ti][fi]].append(dominant_frequencies[ti+1][fi])
                            to_fit_freq[ci][fi][dominant_frequencies[ti][fi]].append(dominant_frequencies[ti][fi+1])
                            to_fit_amplitude[ci][fi][dominant_frequencies[ti][fi]].append(dominant_amplitudes[ti][fi])




            # frequency_tally[ci, dominant_frequencies[self.n_t - 1]] += 1

        self.to_fit_time = to_fit_time
        self.to_fit_freq = to_fit_freq
        self.to_fit_amplitude = to_fit_amplitude

        freq_sum = np.sum(frequency_tally, 1)[0]

        freq_prob = frequency_tally / freq_sum

        freq_prob += np.average(freq_prob) / 100

        self.freq_prob = freq_prob

        self.log_freq_prob = np.log(freq_prob)

        print(np.min(self.log_freq_prob))
        print(np.max(self.log_freq_prob))

    def fit(self):

        de_time = np.array([[[GaussianMixture() for _ in range(self.n_bins)] for _ in range(self.n_freq)]
                            for _ in range(self.n_classes)])
        de_freq = np.array([[[GaussianMixture() for _ in range(self.n_bins)]for _ in range(self.n_freq)]
                            for _ in range(self.n_classes)])
        de_amplitude = np.array([[[GaussianMixture() for _ in range(self.n_bins)]for _ in range(self.n_freq)]
                                 for _ in range(self.n_classes)])

        valid = np.ndarray(shape=[self.n_classes, self.n_bins], dtype=bool)
        valid.fill(True)

        avg_amplitude = mean(list(map(float, list(deepflatten(self.to_fit_amplitude)))))

        for class_i in range(self.n_classes):
            for domi_i in range(self.n_freq):
                for freq_i in range(self.n_bins):

                    print(class_i, domi_i, freq_i)

                    if len(self.to_fit_freq[class_i][domi_i][freq_i]) < 2:
                        self.to_fit_freq[class_i][domi_i][freq_i].append(freq_i)
                        self.to_fit_freq[class_i][domi_i][freq_i].append(freq_i)

                    if len(self.to_fit_time[class_i][domi_i][freq_i]) < 2:
                        self.to_fit_time[class_i][domi_i][freq_i].append(freq_i)
                        self.to_fit_time[class_i][domi_i][freq_i].append(freq_i)

                    if len(self.to_fit_amplitude[class_i][domi_i][freq_i]) < 2:
                        self.to_fit_amplitude[class_i][domi_i][freq_i].append(avg_amplitude)
                        self.to_fit_amplitude[class_i][domi_i][freq_i].append(avg_amplitude)

                    de_time[class_i][domi_i][freq_i].fit(np.reshape(np.array(self.to_fit_freq[class_i][domi_i][freq_i]),
                                                                    [-1, 1]))

                    de_freq[class_i][domi_i][freq_i].fit(np.reshape(np.array(self.to_fit_freq[class_i][domi_i][freq_i]),
                                                                    [-1, 1]))

                    de_amplitude[class_i][domi_i][freq_i].fit(np.reshape(np.array(self.to_fit_freq[class_i][domi_i][freq_i]),
                                                                         [-1, 1]))

        self.de_time = de_time
        self.de_freq = de_freq
        self.de_amplitude = de_amplitude

        with open('de.pkl', 'wb') as file:
            pickle.dump([self.de_time, self.de_freq, self.de_amplitude], file)

    def predict(self, song):

        stft = lb.core.stft(y=song,n_fft=1024)

        song = np.abs(stft).astype(np.float32)

        try:
            _ = self.de_time == 1

        except:
            with open('de.pkl', 'rb') as file:
                de = pickle.load(file)
                self.de_time = de[0]
                self.de_freq = de[1]
                self.de_amplitude = de[2]

        argsort = np.argsort(-song, 0)[:self.n_freq, :]
        dominant_frequencies = np.transpose(argsort)
        dominant_amplitudes = np.transpose(-np.sort(-song, 0)[:self.n_freq, :])

        probs = np.zeros([self.n_classes], np.float32)

        # TODO fix this

        try:
            probs += np.squeeze(self.log_freq_prob[:, dominant_frequencies[0][0]])

        except:
            pass

        print(probs)

        for class_i in range(self.n_classes):
            for time_i in range(dominant_frequencies.shape[0] - 1):
                for domi_i in range(self.n_freq):

                    try:
                        if time_i > 0:
                            probs[class_i] += self.de_time[class_i][domi_i][dominant_frequencies[time_i - 1][domi_i]]\
                                .score(dominant_frequencies[time_i][domi_i])

                        if domi_i > 0:
                            probs[class_i] += self.de_freq[class_i][domi_i-1][dominant_frequencies[time_i][domi_i-1]]\
                                    .score(dominant_frequencies[time_i][domi_i])


                        probs[class_i] += self.de_amplitude[class_i][domi_i][dominant_frequencies[time_i][domi_i]]\
                            .score(dominant_amplitudes[time_i][domi_i])
                    except:
                        pass

        print(probs)
        ans_index = np.argmax(probs)

        # TODO fix this

        try:
            ans_class = self.index_to_genre[ans_index]
        except:
            ans_class = ans_index

        return ans_class

    def test(self):

        predictions = 0
        correct_predictions = 0

        p_list  = []
        l_list = []

        for i, song in enumerate(self.X_test[::-1]):

            p = self.predict(song)

            p_list.append(p)
            l_list.append(self.Y_test[i])

            print('predict ', p)
            print('label ', self.Y_test[i])

            predictions += 1

            if(p == self.Y_test[i]):
                correct_predictions += 1

            print(correct_predictions / predictions)

            print('acc', sklearn.metrics.accuracy_score(l_list, p_list))
            print('macro', sklearn.metrics.f1_score(l_list, p_list, average='macro'))
            print('micro', sklearn.metrics.f1_score(l_list, p_list, average='micro'))


if __name__=='__main__':
    mod = Model()
    mod.build_model()
    mod.fit()
    mod.test()


