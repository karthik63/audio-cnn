# import logging
# logging.basicConfig(level=print)
import numpy as np
import os
# import pomegranate
# from pomegranate import *
# from pomegranate.distributions import *
import librosa as lb
from sklearn.cluster import KMeans
from data_preparation import *
# from hmm.continuous.GMHMM import GMHMM
from hmmlearn.hmm import GMMHMM
from sklearn.mixture import *
import sklearn

np.random.seed(1234)

print('something')

class Audio_HMM():
    def __init__(self, n_states=15, n_mixtures=1, feature='mfcc', n_freq=5, n_mfcc=20,
                 sampling_rate=22050):

        self.n_states = n_states
        self.n_mixtures = n_mixtures

        #! TODO what is the correct error type here ?

        if feature not in ['mfcc', 'mfcc_v', 'mfcc_va', 'stft']:
            raise ValueError('feature type must be among mfcc, mfcc_v, mfcc_va, stf ')

        self.feature = feature
        self.n_freq = n_freq
        self.n_mfcc = n_mfcc
        self.sampling_rate = sampling_rate

    @staticmethod
    def extract_mfcc(X, sampling_rate, n_mfcc):
        sr = sampling_rate
        return np.transpose(lb.feature.mfcc(y=X, sr=sr, n_mfcc=n_mfcc))

    @staticmethod
    def extract_stft(X):
        return np.transpose(np.abs(lb.core.stft(X)))

    def extract_features(self, X):
        print(' * extracting features * ')
        if self.feature == 'stft':
            extract_stft_vectorised = np.vectorize(self.extract_stft, otypes=[np.float64],
                                               signature='(a)->(b,c)')
            X = extract_stft_vectorised(X)

        if self.feature == 'mfcc':
            extract_mfcc_vectorised = np.vectorize(self.extract_mfcc, otypes=[np.float64],
                                               signature='(a),(),()->(d,e)')
            X = extract_mfcc_vectorised(X, self.sampling_rate, self.n_mfcc)

        return X

    def fit(self, X, Y):
        '''
        creates a separate hmm for each label in Y
        '''

        labels = []

        for l in Y:
            if l not in labels:
                labels.append(l)

        labels = sorted(labels)

        self.n_class = len(labels)

        print(' doing something ')

        try:
            X = np.load('X_mfcc.npy')

        except:
            X = self.extract_features(X)
            np.save('X_mfcc.npy', X)

        print(' * finished extracting features * ')

        self.hmm_set = []

        self.hmm_set.append(GMMHMM(n_components=self.n_states,
                              n_mix=self.n_mixtures,
                              verbose=True,
                              n_iter=100) )

        self.hmm_set.append(GMMHMM(n_components=self.n_states,
                              n_mix=self.n_mixtures,
                              verbose=True,
                              n_iter=100) )

        self.hmm_set.append(GMMHMM(n_components=self.n_states,
                              n_mix=self.n_mixtures,
                              verbose=True,
                              n_iter=100) )

        self.hmm_set.append(GMMHMM(n_components=self.n_states,
                              n_mix=self.n_mixtures,
                              verbose=True,
                              n_iter=100) )

        print(self.hmm_set)

        class_data = [[] for _ in range(self.n_class)]
        lengths = [[] for _ in range(self.n_class)]

        print(' * preprocessing * ')

        for i, data in enumerate(X):
            class_data[int(Y[i])].append(data)
            lengths[int(Y[i])].append(data.shape[0])

        print(' * finished preprocessing * ')

        for ci in range(self.n_class):
            print('fitting {}'.format(ci))
            to_fit = np.concatenate(class_data[ci])

            print(to_fit[0].shape)
            print(to_fit.shape)

            self.hmm_set[ci].fit(to_fit, lengths[ci])

            if np.any(self.hmm_set[ci].covars_ <= 0):
                print('some covariances are 0. model might be a poor fit')
                self.hmm_set[ci].covars_ = np.abs(self.hmm_set[ci].covars_) + 1e-10

            if np.any(self.hmm_set[ci].transmat_ == np.nan):
                raise ArithmeticError('transition probabilities are unndefined. '
                                      'Try reducing the number of states')



    def predict(self, X):

        X = self.extract_features(X)

        n_predictions = X.shape[0]
        predictions = np.zeros(n_predictions, np.int32)

        for i, data in enumerate(X):
            print('predicting ',i)
            temp_preds = []

            for ci in range(self.n_class):
                temp_preds.append(self.hmm_set[ci].score(data))

            p = np.argmax(temp_preds)
            predictions[i] = p

        return predictions

    def get_accuracy(self, y_true, y_pred):

        acc = sklearn.metrics.accuracy_score(y_true, y_pred)

        return acc

X_train = np.load('data/indian_4_sana_segmented_X_train.npy')
Y_train = np.load('data/indian_4_sana_segmented_Y_train.npy')

X_test = np.load('data/indian_4_sana_segmented_X_test.npy')
Y_test = np.load('data/indian_4_sana_segmented_Y_test.npy')

print('before instantiating')
f = Audio_HMM()
print('before fitting')
f.fit(X_train, Y_train)
p = f.predict(X_test)
print(p)
print(f.get_accuracy(Y_test, p))

# s = np.array([[1.0, 2,3,4],
#               [5,6,7,8],
#               [9,10,11,12],
#               [2,4,6,8],
#               [3,6,9,12],
#               [4,8,12,16],
#               [5,10,15,20],
#               [6,12,18,24],
#               [7,14,21,28],
#               [8,16,24,32],
#               [9,18,27,36],
#               [10,11,12,13],
#               [1,3,5,7],
#               [11,10,13,14],
#               [15,16,17,18],
#               [19,20,21,22]])
#
#
# mod = HiddenMarkovModel.from_samples(NormalDistribution, n_components=3, X=s)
# mod.fit(s, verbose=True)
#
# print(mod.sample(5, length=6))
# #
# ts, sr = lb.load('../esc/dog/3-136288-A-0.wav', sr=None)
#
# print(np.shape(ts))
# print(sr)
# #
# #
# # def get_data():
#
# def main():
#     X_tr = np.load('X_train.npy')
#     X_te = np.load('X_test.npy')
#     Y_tr = np.load('Y_train.npy')
#     Y_te = np.load('Y_test.npy')
#     sampling_rate = np.load('sampling_rate.npy')
#
#     extract_stft_vectorised = np.vectorize(extract_stft, otypes=[np.float64],
#                                             signature='(a)->(b,c)')
#
#     extract_mfcc_vectorised = np.vectorize(extract_mfcc, otypes=[np.float64],
#                                            signature='(a),b->(c,d)')
#
#     # X_stft = extract_mfcc_vectorised(X_tr)
#     #
#     # X_te = extract_mfcc_vectorised(X_te)
#
#     X_te_1 = extract_stft_vectorised(X_te)
#     # X_te_1 /= np.max(X_te_1)
#     X_te_2 = np.argsort(-X_te_1)[:,:,:20].astype(np.float64)
#     X_te_3 = -X_te_1
#     X_te_3.sort(2)
#     X_te_3 = -X_te_3[:, :, :20].astype(np.float64)
#     # X_te_3 /= np.max(X_te_3)
#     X_te = np.concatenate((X_te_2, X_te_3), axis=2)
#     # X_te = X_te_2
#
#     X_stft = extract_stft_vectorised(X_tr)
#
#     max_freqs = np.argsort(-X_stft)[:,:,:20].astype(np.float64)
#
#     # max_freqs /= np.max(max_freqs)
#
#     print(max_freqs[0])
#
#     print(max_freqs.shape)
#
#     X_stft = -X_stft
#
#     X_stft.sort(2)
#
#     X_stft = -X_stft[:,:,:20].astype(np.float64)
#
#     print(X_stft.shape)
#     print(X_stft[0])
#
#     # X_stft /= np.max(X_stft)
#     #
#     X_stft = np.concatenate((max_freqs, X_stft), axis=2)
#
#     # X_stft = max_freqs
#
#     print(X_stft.shape)
#
#     n_per_class = int(X_stft.shape[0] / n_classes)
#
#     X_dog = X_stft[0:n_per_class,:,:]
#     X_pig = X_stft[n_per_class:n_per_class*2,:,:]
#     X_rooster = X_stft[n_per_class*2:n_per_class*3,:,:]
#
#     # n_states = 10
#     #
#     # X = np.reshape(X_dog, (-1, 10))
#     # y_pred = KMeans(n_states).fit_predict(X)
#     #
#     # means = np.zeros([n_states, 1, 10], dtype=np.float64)
#     # covs = np.zeros([n_states, 1, 10, 10], dtype=np.float64)
#     #
#     # distributions = []
#     # for i in range(n_states):
#     #     X_subset = X[y_pred == i]
#     #     # X_mean = np.expand_dims(np.mean(X_subset, 0), 0)
#     #     # X_cov = np.expand_dims(np.cov(X_subset.T), 0)
#     #
#     #     gmm = GaussianMixture().fit(X_subset)
#     #     X_mean = gmm.means_
#     #     X_cov = gmm.covariances_
#     #
#     #     if i == 4:
#     #         print(X_cov[0])
#     #
#     #     print(np.all(np.linalg.eigvals(X_cov[0].astype(np.float64)) > 0))
#     #
#     #     means[i] = X_mean
#     #     covs[i] = X_cov
#     #
#     # for i in range(n_states):
#     #     for j in range(1):
#     #
#     #         if i == 4:
#     #             print(covs[i][j])
#     #         print(numpy.all(numpy.linalg.eigvals(covs[i][j]) > 0))
#     #         print('boo ', i)
#
#     hmm_dog = GMMHMM(n_components=4,
#                  n_mix=1,
#                  verbose=True,
#                  n_iter=100)
#
#
#     hmm_pig = GMMHMM(n_components=4,
#                  n_mix=1,
#                  verbose=True,
#                  n_iter=100)
#
#     hmm_rooster = GMMHMM(n_components=4,
#                  n_mix=1,
#                  verbose=True,
#                  n_iter=100)
#
#     dog = []
#     pig = []
#     rooster = []
#
#     dog_lengths = []
#     pig_lengths = []
#     rooster_lengths = []
#
#     for data in X_dog:
#         dog.append(data)
#         dog_lengths.append(data.shape[0])
#
#     dog = np.concatenate(dog)
#
#     for data in X_pig:
#         pig.append(data)
#         pig_lengths.append(data.shape[0])
#
#     pig = np.concatenate(pig)
#
#     for data in X_rooster:
#         rooster.append(data)
#         rooster_lengths.append(data.shape[0])
#
#     rooster = np.concatenate(rooster)
#
#     hmm_dog.fit(dog, lengths=dog_lengths)
#     print(hmm_dog.transmat_)
#
#     hmm_pig.fit(pig, lengths=pig_lengths)
#     print(hmm_pig.transmat_)
#
#     hmm_rooster.fit(rooster, lengths=rooster_lengths)
#     print(hmm_rooster.transmat_)
#
#     print(hmm_rooster.covars_)
#
#     hmm_dog.covars_ = np.abs(hmm_dog.covars_) + 1e-10
#     hmm_pig.covars_ = np.abs(hmm_pig.covars_) + 1e-10
#     hmm_rooster.covars_ = np.abs(hmm_pig.covars_) + 1e-10
#
#     predictions = 0
#     correct_predictions = 0
#
#     for i, data in enumerate(X_te):
#
#         dog_val = hmm_dog.score(data)
#         pig_val = hmm_pig.score(data)
#         rooster_val = hmm_rooster.score(data)
#
#         predictions += 1
#
#         if np.max([dog_val, pig_val, rooster_val]) == dog_val:
#             print('dog')
#             if Y_te[i] == 0:
#                 correct_predictions += 1
#
#         if np.max([dog_val, pig_val, rooster_val]) == pig_val:
#             print('pig')
#             if Y_te[i] == 1:
#                 correct_predictions += 1
#
#         if np.max([dog_val, pig_val, rooster_val]) == rooster_val:
#             print('rooster')
#             if Y_te[i] == 2:
#                 correct_predictions += 1
#
#     print(correct_predictions / predictions, ' accuracy')
#
#     print(hmm_dog.score(X_pig[2]))
#     print(hmm_pig.score(X_pig[2]))
#     print(hmm_rooster.score(X_pig[2]))
#
#     print('oy vey')
#
#     # A = np.ones((n_states, n_states), dtype=np.float64) * (1.0 / n_states)
#     # pi = numpy.ones((n_states), dtype=np.float64) * (1.0 / n_states)
#     # w = numpy.ones((n_states, n_states), dtype=np.float64) * (1.0 / n_states)
#     #
#     # gmhmm = GMHMM(n=n_states, m=1, d=10, means=means, covars=covs, A=A, w=w, pi=pi, init_type='user', precision=np.float64)
#     #
#     # gmhmm.train(X_dog[0], 100)
#     #
#     # f = gmhmm.forward_backward(X_dog[1])
#
#     # print(f)
#
#     #     distribution = MultivariateGaussianDistribution(X_mean, X_cov)
#     #     distributions.append(distribution)
#     #
#     # transitions = np.ones((n_states, n_states), dtype='float32') / n_states
#     # starts = np.ones(n_states, dtype='float32') / n_states
#     #
#
#
#
#     # model = HiddenMarkovModel.from_matrix(transitions, distributions, starts)
#     # model.fit(X, verbose=True)
#
#     # mod_dog = HiddenMarkovModel.from_samples(MultivariateGaussianDistribution(np.random.randn(80), np.eye(80)), n_components=10, X=X_dog[:,:,: ])
#     # mod_dog.fit(X_dog, verbose=True)
#     #
#     # mod_pig = HiddenMarkovModel.from_samples(MultivariateGaussianDistribution(np.random.randn(80), np.eye(80)), n_components=10, X=X_pig)
#     # mod_pig.fit(X_pig, verbose=True)
#     #
#     # mod_rooster = HiddenMarkovModel.from_samples(MultivariateGaussianDistribution(np.random.randn(80), np.eye(80)), n_components=10, X=X_rooster)
#     # mod_rooster.fit(X_rooster, verbose=True)
#     #
#     # print(mod_dog.log_probability(X_dog[0]))
#     # print(mod_pig.log_probabilities(X_dog[0]))
#     # print(mod_rooster.log_probabilities(X_dog[0]))
#
#     # extract_mfcc_vectorized = np.vectorize(extract_MFCC, otypes=[np.float64],
#     #                                         signature='(a)->(b,c)')
#     #
#     #
#     # X_mfcc = extract_mfcc_vectorized(X_tr)
#     #
#     # n_per_class = int(X_mfcc.shape[0] / n_classes)
#     #
#     # X_dog = X_mfcc[0:n_per_class,:,:]
#     # X_pig = X_mfcc[n_per_class:n_per_class*2,:,:]
#     # X_rooster = X_mfcc[n_per_class*2:n_per_class*3,:,:]
#     #
#     # X_dog = X_dog / np.max(np.abs(X_dog))
#     # X_pig = X_pig / np.max(np.abs(X_pig))
#     # X_rooster = X_rooster / np.max(np.abs(X_rooster))
#     #
#     # print(X_dog.shape)
#     # print(X_pig.shape)
#     # print(X_rooster.shape)
#     #
#     # # to_cluster = np.reshape(X_mfcc, (-1, 20))
#     # #
#     # # skl
#     #
#     # print(np.max(X_dog))
#     # print(np.min(X_dog))
#     # #
#     # X_dog = np.random.rand(X_dog.shape[0], X_dog.shape[1], X_dog.shape[2])
#     #
#     # print(np.shape(X_dog))
#     #
#     # print(np.max(X_dog))
#     # print(np.min(X_dog))
#
#     # mod_dog = HiddenMarkovModel.from_samples(NormalDistribution, n_components=10, X=X_dog[:,:,: ])
#     # mod_dog.fit(X_dog, verbose=True)
#
#     # mod_pig = HiddenMarkovModel.from_samples(NormalDistribution, n_components=100, X=X_pig)
#     # mod_pig.fit(X_pig, verbose=True)
#     #
#     # mod_rooster = HiddenMarkovModel.from_samples(NormalDistribution, n_components=100, X=X_rooster)
#     # mod_rooster.fit(X_rooster, verbose=True)
#     #
#     # print(mod_dog.log_probability(X_dog[0]))
#     # print(mod_pig.log_probabilities(X_dog[0]))
#     # print(mod_rooster.log_probabilities(X_dog[0]))
#
# main()