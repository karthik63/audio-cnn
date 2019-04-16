import numpy as np
import tensorflow as tf
import librosa as lb
import seaborn as sn
import math
from numba.targets.arraymath import np_all
import sklearn
from tensorflow.python.training import optimizer
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import CuDNNLSTM, Masking, Dense, Input, LSTM


from data_preparation import *
np.random.seed(1234)
tf.set_random_seed(1234)

tracking = True

class GenreCNN:

    def __init__(self, preprocess=False, class_names=None,
                 mel=True, stft=False,
                 batch_size=5,
                 max_itrns=3000,
                 n_classes=4,
                 save_path='saved_models_indian_4_sana_segmented_summary_finding_3',
                 log_path ='saved_models_indian_4_sana_segmented_summary_finding_logs_3',
                 test_songwise=False,
                 lstm_input_size=500,
                 lstm_batch_size=10,
                 max_itrns_lstm=1000):

        self.log_path_train = os.path.join(log_path,'train')
        self.log_path_validation = os.path.join(log_path,'validation')
        self.max_itrns_lstm = max_itrns_lstm
        self.max_sequence_length = None
        self.lstm_batch_size = lstm_batch_size
        self.lstm_input_size = lstm_input_size
        self.test_songwise = test_songwise
        self.mel = mel
        self.stft = stft
        self.batch_size = batch_size
        self.input_h = 128
        self.input_w = 1293
        self.max_itrns = max_itrns
        self.n_classes = n_classes
        self.save_path = save_path
        self.sess = None

        if not os.path.exists(self.log_path_train):
            os.makedirs(self.log_path_train)

        if not os.path.exists(self.log_path_validation):
            os.makedirs(self.log_path_validation)

        if class_names:
            self.index_to_class = sorted(class_names)

    @staticmethod
    def extract_spectrogram(ts_data, ):
        global tracking
        mel_sg = lb.feature.melspectrogram(ts_data)

        if tracking:
            tracking = False
            print(mel_sg.shape)

        return mel_sg

    @staticmethod
    def shuffle(X, Y):

        concat = np.hstack((X, np.reshape(Y, (Y.shape[0], 1))))
        np.random.shuffle(concat)

        return concat[:, 0:-1], concat[:, -1]

    def get_lstm_data_X(self, X, segment_counts):
        print(self.sess)

        if self.sess == None:
            self.sess = tf.Session()
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_path))
            print('* reloaded *')

        if self.mel:
            extract_melsg_vectorized = np.vectorize(self.extract_spectrogram, otypes=[np.float32],
                                                    signature='(a)->(b,c)')
            X = extract_melsg_vectorized(X)

        X = np.expand_dims(X, 3)

        n_predictions = X.shape[0]
        intermediate = np.zeros((n_predictions, self.lstm_input_size), np.float32)

        for xi in range(n_predictions // self.batch_size):
            data = X[xi * self.batch_size: (xi + 1) * self.batch_size]

            pool4 = self.sess.run(self.pool4, {self.input_batch: data})

            intermediate[xi * self.batch_size: (xi + 1) * self.batch_size, :] = pool4

            print(xi * self.batch_size, (xi + 1) * self.batch_size)

        if n_predictions % self.batch_size != 0:
            data = X[n_predictions - self.batch_size: n_predictions]

            pool4 = self.sess.run(self.pool4, {self.input_batch: data})

            intermediate[n_predictions - self.batch_size: n_predictions, :] = pool4

            print(n_predictions - self.batch_size, n_predictions)

        print(intermediate)

        index = 0

        sequence_data = []

        for song_index in range(segment_counts.shape[0]):
            temp = []
            for segment_index in range(int(segment_counts[song_index])):
                temp.append(intermediate[index])
                index += 1
            sequence_data.append(temp)

        return sequence_data

    def get_lstm_data_Y(self, Y, segment_count):

        print(self.sess)

        Y_corrected = []
        index = 0

        for seg_count in segment_count:
            Y_corrected.append(Y[index])
            index += int(seg_count)

        Y_corrected = np.array(Y_corrected, dtype=np.float32)

        return Y_corrected

    def predict_lstm(self, X):
        print(self.sess)

        if self.sess == None:
            self.sess = tf.Session()
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_path + '_lstm'))
            print('* reloaded *')

        n_predictions = X.shape[0]
        predictions = np.zeros(n_predictions, np.int32)

        for xi in range(n_predictions // self.lstm_batch_size):
            data = X[xi * self.lstm_batch_size: (xi + 1) * self.lstm_batch_size]

            class_scores = self.sess.run(self.lstm_output, {self.input_batch_lstm: data})
            class_prediction = np.argmax(class_scores, axis=1)

            predictions[xi * self.lstm_batch_size: (xi + 1) * self.lstm_batch_size] = class_prediction

            print(xi * self.lstm_batch_size, (xi + 1) * self.lstm_batch_size)

        if n_predictions % self.lstm_batch_size != 0:
            data = X[n_predictions - self.lstm_batch_size: n_predictions]

            class_scores = self.sess.run(self.lstm_output, {self.input_batch_lstm: data})
            class_prediction = np.argmax(class_scores, axis=1)

            predictions[n_predictions - self.lstm_batch_size: n_predictions] = class_prediction

            print(n_predictions - self.lstm_batch_size, n_predictions)

        print(predictions)
        return predictions

    def fit_lstm(self, X_train, Y_train, X_test, Y_test, segment_count_train, segment_count_test):

        self.max_sequence_length = int(np.max((np.max(segment_count_test), np.max(segment_count_train))))

        Y_train= self.get_lstm_data_Y(Y_train, segment_count_train)

        Y_train = np.eye(self.n_classes, dtype=np.float32)[Y_train.astype(np.int32)]

        Y_test = self.get_lstm_data_Y(Y_test, segment_count_test)

        X_train = self.get_lstm_data_X(X_train, segment_count_train)

        X_test = self.get_lstm_data_X(X_test, segment_count_test)

        X_train = np.array(pad_sequences(X_train, maxlen=self.max_sequence_length, dtype='float32', padding='post'),
                           dtype=np.float32)
        X_test = np.array(pad_sequences(X_test, maxlen=self.max_sequence_length, dtype='float32', padding='post'),
                          dtype=np.float32)

        self.input_batch_lstm = tf.placeholder(shape=(self.lstm_batch_size, self.max_sequence_length, self.lstm_input_size),
                                     dtype=tf.float32)

        label_batch = tf.placeholder(shape=(self.lstm_batch_size, self.n_classes), dtype=tf.float32)

        with tf.variable_scope('LSTM_audio'):

            input_masked = Masking()(self.input_batch_lstm)

            lstm_out = LSTM(100)(input_masked)

            print(lstm_out)

            dense_1 = Dense(20, activation='relu')(lstm_out)

            dense_2 = Dense(self.n_classes, activation='softmax')(dense_1)

            self.lstm_output = dense_2

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_2, labels=label_batch))

            continuous_loss_summary = tf.summary.scalar('lstm_continuous_loss', loss)

            optimize_step = tf.train.AdamOptimizer().minimize(loss)

            lstm_train_summaries = []
            lstm_validation_summaries = []
            lstm_weight_summaries = []

            accuracy_placeholder = tf.placeholder(tf.float32, shape=())
            microf_placeholder = tf.placeholder(tf.float32, shape=())
            macrof_place_holder = tf.placeholder(tf.float32, shape=())
            loss_placeholder = tf.placeholder(tf.float32, shape=())

            loss_summary = tf.summary.scalar('loss_lstm', loss_placeholder)

            lstm_train_summaries.append(tf.summary.scalar('ac_lstm', accuracy_placeholder))
            lstm_train_summaries.append(tf.summary.scalar('microf_lstm', microf_placeholder))
            lstm_train_summaries.append(tf.summary.scalar('macro_f_lstm', macrof_place_holder))

            lstm_validation_summaries.append(tf.summary.scalar('ac_lstm', accuracy_placeholder))
            lstm_validation_summaries.append(tf.summary.scalar('microf_lstm', microf_placeholder))
            lstm_validation_summaries.append(tf.summary.scalar('macro_f_lstm', macrof_place_holder))
            lstm_validation_summaries.append(loss_summary)

            queue = tf.RandomShuffleQueue(capacity=self.lstm_batch_size * 5,
                                          shapes=[(self.max_sequence_length, self.lstm_input_size), self.n_classes],
                                          dtypes=[tf.float32, tf.float32], min_after_dequeue=self.lstm_batch_size * 2)

            enqueue_op = queue.enqueue_many([X_train, Y_train])

            qr = tf.train.QueueRunner(queue, [enqueue_op] * 6)

            if self.sess == None:
                self.sess = tf.Session()

            coord = tf.train.Coordinator()
            enqueue_threads = qr.create_threads(self.sess, coord=coord, start=True)

            self.sess.run(tf.global_variables_initializer())

            input_b, label_b = queue.dequeue_many(self.lstm_batch_size)

            for batch in range(self.max_itrns_lstm):
                in_b_run, label_b_run = self.sess.run([input_b, label_b])

                _, loss_val = self.sess.run([optimize_step, loss],
                                       feed_dict={self.input_batch_lstm: in_b_run, label_batch: label_b_run})

                ls = self.sess.run(loss_summary, {loss_placeholder: loss})

                self.train_writer.add_summary(ls)

                print("loss: {}, batch {}".format(loss_val, batch))

                if ((batch + 1) % 200 == 0):
                    print(' * saaved * ', batch)
                    self.saver.save(self.sess, os.path.join(self.save_path + '_lstm', 'model.ckpt'), global_step=batch)

                if (batch + 1) % 100 == 0 and np.any(X_test) and np.any(Y_test):

                    prediction = self.predict_lstm(X_test)

                    print(prediction)
                    ac = self.get_accuracy(Y_test, prediction)
                    print(ac)

                    cm = self.get_cm(Y_test, prediction)
                    print(cm)

                    print(sklearn.metrics.f1_score(Y_test, prediction, average='micro'), ' micro')
                    print(sklearn.metrics.f1_score(Y_test, prediction, average='macro'), ' macro')

            coord.request_stop()
            coord.join(enqueue_threads)

    def build_model(self):

        self.input_batch = tf.placeholder(np.float32,
                               [self.batch_size, self.input_h, self.input_w, 1])

        self.label_batch = tf.placeholder(np.float32,
                               [self.batch_size, self.n_classes])

        with tf.variable_scope('CNN_genre'):

            conv1 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.relu, padding='same')(self.input_batch)

            pool1 = tf.keras.layers.MaxPool2D((2, 4), padding='same')(conv1)

            conv2 = tf.keras.layers.Conv2D(384, (3, 3), activation=tf.keras.activations.relu, padding='same')(pool1)

            pool2 = tf.keras.layers.MaxPool2D((4, 5), padding='same')(conv2)

            conv3 = tf.keras.layers.Conv2D(500, (3, 3), activation=tf.keras.activations.relu, padding='same')(pool2)

            pool3 = tf.keras.layers.MaxPool2D((3, 8), padding='same')(conv3)

            conv4 = tf.keras.layers.Conv2D(500, (3, 3), activation=tf.keras.activations.relu,
                                           strides=(3,3), padding='same')(pool3)

            pool4 = tf.keras.layers.MaxPool2D((2,3), padding='same')(conv4)

            pool4 = tf.squeeze(pool4)

            print(pool4.get_shape())

            class_scores = tf.keras.layers.Dense(self.n_classes)(pool4)

            self.pool4 = pool4

            self.class_scores = class_scores

        # with tf.Session() as sess:
        #
        #     sess.run(tf.global_variables_initializer())
        #
        #     a = np.zeros([self.batch_size, self.input_h, self.input_w, 1], np.float32)
        #
        #     ans = sess.run(a_shape, feed_dict={input: a})

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.class_scores, labels=self.label_batch))

        self.global_step = tf.Variable(0, trainable=False)

        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)

        #summaries

        self.train_summaries = []
        self.validation_summaries = []
        self.weight_summaries = []

        self.saver = tf.train.Saver(max_to_keep=4)

        self.train_writer = tf.summary.FileWriter(self.log_path_train)
        self.validation_writer = tf.summary.FileWriter(self.log_path_validation)

        self.summary = tf.Summary()

        self.accuracy_place_holder = tf.placeholder(tf.float32, shape=())

        self.microf_placeholder = tf.placeholder(tf.float32, shape=())
        self.macrof_place_holder = tf.placeholder(tf.float32, shape=())

        self.loss_placeholder = tf.placeholder(tf.float32, shape=())
        self.loss_summary = tf.summary.scalar('loss', self.loss_placeholder)

        for variable in tf.trainable_variables():
            self.weight_summaries.append(tf.summary.histogram(variable.name, variable))

        sacc = tf.summary.scalar('accuracy', self.accuracy_place_holder)
        smac = tf.summary.scalar('macrof', self.macrof_place_holder)
        smic = tf.summary.scalar('microf', self.microf_placeholder)

        self.train_summaries.append(sacc)
        self.train_summaries.append(smac)
        self.train_summaries.append(smic)

        self.validation_summaries.append(sacc)
        self.validation_summaries.append(smac)
        self.validation_summaries.append(smic)
        self.validation_summaries.append(self.loss_summary)


        self.merged_summaries_train = tf.summary.merge(self.train_summaries)
        self.merged_summaries_validation = tf.summary.merge(self.validation_summaries)
        self.merged_summaries_weight = tf.summary.merge(self.weight_summaries)

        print('boo')

    def train(self, X_te=None, Y_te=None, segment_count_te=None,):

        # queue = tf.FIFOQueue(capacity=self.batch_size* 5, shapes=[(self.input_h, self.input_w, 1), self.n_classes],
        #                      dtypes=[tf.float32, tf.float32])

        queue = tf.RandomShuffleQueue(capacity=self.batch_size * 5, shapes=[(self.input_h, self.input_w, 1), self.n_classes],
                             dtypes=[tf.float32, tf.float32], min_after_dequeue=self.batch_size * 2)

        enqueue_op = queue.enqueue_many([self.X_train_sg, self.Y_train])

        input_batch, label_batch = queue.dequeue_many(self.batch_size)

        qr = tf.train.QueueRunner(queue, [enqueue_op] * 6)


        sess = tf.Session()

        coord = tf.train.Coordinator()
        enqueue_threads = qr.create_threads(sess, coord=coord, start=True)

        self.sess = sess

        try:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_path))
            print('*    reloaded *')
        except:
            sess.run(tf.global_variables_initializer())
            print(' * couldnt reload * ')

        start = int(sess.run(self.global_step))

        for ei in range(start, self.max_itrns):

            in_b, l_b = sess.run([input_batch, label_batch])
            loss, weight_summary, _ = sess.run((self.loss,
                                                     self.merged_summaries_weight, self.optimizer),
                                                    {self.input_batch: in_b, self.label_batch: l_b})

            loss_summary = sess.run(self.loss_summary, {self.loss_placeholder: loss})

            self.train_writer.add_summary(weight_summary, ei)
            self.train_writer.add_summary(loss_summary, ei)

            print("loss: {}, batch {}".format(loss, ei))

            if ((ei + 1) % 1000 == 0):
                print(' * saaved * ', ei)
                self.saver.save(sess, os.path.join(self.save_path, 'model.ckpt'),global_step=ei)

            if (ei + 1) % 70 == 0:

                n_test = X_te.shape[0]

                to_find = self.X_train[:n_test]
                to_find_labels = np.argmax(self.Y_train[:n_test], 1)

                prediction = self.predict(to_find)

                # output_tensor = tf.constant(outputs)
                # labels_tensor = tf.constant(np.eye(self.n_classes, dtype=np.float32)[to_find_labels.astype(np.int32)])

                # train_loss = sess.run(-tf.reduce_mean(labels_tensor * tf.log(output_tensor), reduction_indices=[1]))

                print(prediction)
                ac = self.get_accuracy(to_find_labels, prediction)
                print(ac)

                cm = self.get_cm(to_find_labels, prediction)
                print(cm)

                macro = sklearn.metrics.f1_score(to_find_labels, prediction, average='macro')
                micro = sklearn.metrics.f1_score(to_find_labels, prediction, average='micro')

                summaries_train = sess.run(self.merged_summaries_train,
                                                {self.accuracy_place_holder: ac,
                                                 self.macrof_place_holder: macro,
                                                 self.microf_placeholder: micro,
                                                 }
                                            )

                self.train_writer.add_summary(summaries_train, ei)

                print(micro, ' micro')
                print(macro, ' macro')

            if (ei + 1) % 70 == 0 and np.any(X_te) and np.any(Y_te):

                if not self.test_songwise:
                    outputs = self.output(X_te)
                    prediction = np.argmax(outputs, 1)

                if self.test_songwise:
                    outputs = self.output(X_te)
                    prediction = self.get_songwise_prediction(outputs, segment_count_te)

                output_tensor = tf.constant(outputs)
                labels_tensor = tf.constant(np.eye(self.n_classes, dtype=np.float32)[Y_te.astype(np.int32)])

                test_loss = sess.run(-tf.reduce_mean(labels_tensor * tf.log(tf.abs(output_tensor + 1e-5))))

                print(test_loss)

                print(prediction)
                ac = self.get_accuracy(Y_te, prediction)
                print(ac)

                cm = self.get_cm(Y_te, prediction)
                print(cm)

                macro = sklearn.metrics.f1_score(Y_te, prediction, average='macro')
                micro = sklearn.metrics.f1_score(Y_te, prediction, average='micro')

                summaries_validation = sess.run(self.merged_summaries_validation,
                                                {self.accuracy_place_holder: ac,
                                                 self.macrof_place_holder: macro,
                                                 self.microf_placeholder: micro,
                                                 self.loss_placeholder: test_loss,
                                                }
                                                )

                self.validation_writer.add_summary(summaries_validation, ei)

                print(micro, ' micro')
                print(macro, ' macro')

        coord.request_stop()
        coord.join(enqueue_threads)

        self.sess = sess

    def get_songwise_prediction(self, outputs, segment_counts_test):

        n_songs = segment_counts_test.shape[0]
        predictions = np.zeros(n_songs, np.int32)
        out_index = 0

        for song_i in range(n_songs):

            poll = [0] * self.n_classes
            sum = [0] * self.n_classes

            for segment_i in range(int(segment_counts_test[song_i])):

                poll += outputs[out_index]
                best_index = np.argmax(outputs[out_index])
                sum[best_index] += 1
                out_index += 1

            indices = list(range(self.n_classes))
            indices.sort(key=lambda x: sum[x], reverse=True)
            indices.sort(key=lambda x: poll[x], reverse=True)
            predictions[song_i] = indices[0]

        return predictions

    def fit(self, X_train, Y_train, X_te=None, Y_te=None, segment_count_train=None, segment_count_test=None):
        self.X_train = X_train
        self.Y_train = Y_train

        if self.test_songwise:
            self.Y_train = self.get_lstm_data_Y(self.Y_train, segment_count_train)

            self.Y_train = np.eye(self.n_classes, dtype=np.float32)[self.Y_train.astype(np.int32)]

            if Y_te != None:
                Y_te = self.get_lstm_data_Y(Y_te, segment_count_test)

        self.X_train, self.Y_train = self.shuffle(self.X_train, self.Y_train)

        self.Y_train = np.eye(self.n_classes, dtype=np.float32)[self.Y_train.astype(np.int32)]

        time_limit = X_train.shape[0]

        if self.mel:
            extract_melsg_vectorized = np.vectorize(self.extract_spectrogram, otypes=[np.float32],
                                                    signature='(a)->(b,c)')
            self.X_train_sg = extract_melsg_vectorized(self.X_train)

        self.X_train_sg = np.expand_dims(self.X_train_sg, 3)

        self.build_model()
        self.train(X_te, Y_te, segment_count_test)

    def output(self, X):
        print(self.sess)

        if self.sess == None:
            self.sess = tf.Session()
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_path))
            print('* reloaded *')

        if self.mel:
            extract_melsg_vectorized = np.vectorize(self.extract_spectrogram, otypes=[np.float32],
                                                    signature='(a)->(b,c)')
            X = extract_melsg_vectorized(X)

        X = np.expand_dims(X, 3)

        n_predictions = X.shape[0]
        outputs = np.zeros((n_predictions, self.n_classes), np.float32)

        for xi in range(n_predictions//self.batch_size):

            data = X[xi*self.batch_size: (xi + 1) * self.batch_size]

            class_scores = self.sess.run(self.class_scores, {self.input_batch: data})

            outputs[xi*self.batch_size: (xi + 1) * self.batch_size, :] = class_scores

            print(xi*self.batch_size, (xi + 1) * self.batch_size)

        if n_predictions%self.batch_size != 0:
            data = X[n_predictions - self.batch_size: n_predictions]

            class_scores = self.sess.run(self.class_scores, {self.input_batch: data})

            outputs[n_predictions - self.batch_size: n_predictions, :] = class_scores

            print(n_predictions - self.batch_size,  n_predictions)

        print(outputs)
        return outputs

    def predict(self, X):

        print(self.sess)

        if self.sess == None:
            self.sess = tf.Session()
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_path))
            print('* reloaded *')

        if self.mel:
            extract_melsg_vectorized = np.vectorize(self.extract_spectrogram, otypes=[np.float32],
                                                    signature='(a)->(b,c)')
            X = extract_melsg_vectorized(X)

        X = np.expand_dims(X, 3)

        n_predictions = X.shape[0]
        predictions = np.zeros(n_predictions, np.int32)


        for xi in range(n_predictions//self.batch_size):

            data = X[xi*self.batch_size: (xi + 1) * self.batch_size]

            class_scores = self.sess.run(self.class_scores, {self.input_batch: data})
            class_prediction = np.argmax(class_scores, axis=1)

            predictions[xi*self.batch_size: (xi + 1) * self.batch_size] = class_prediction

            print(xi*self.batch_size, (xi + 1) * self.batch_size)

        if n_predictions%self.batch_size != 0:
            data = X[n_predictions - self.batch_size: n_predictions]

            class_scores = self.sess.run(self.class_scores, {self.input_batch: data})
            class_prediction = np.argmax(class_scores, axis=1)

            predictions[n_predictions - self.batch_size: n_predictions] = class_prediction

            print(n_predictions - self.batch_size,  n_predictions)

        print(predictions)
        return predictions

    def get_accuracy(self, y_true, y_pred):


        print(y_true.shape, y_pred.shape)
        acc = sklearn.metrics.accuracy_score(y_true, y_pred)

        return acc

    def get_cm(self, y_true, y_pred):

        cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, cmap=plt.cm.Oranges)
        plt.imsave('c.png', cm, cmap=plt.cm.Oranges)
        plt.show()

        return cm


def main():
    # X_train, X_test, Y_train, Y_test = extract_ts('../mydata')

    # hmm

    bs = 5

    name = 'data/indian_4_sana_segmented'

    X_tr = np.load(name + '_X_train.npy')
    X_te = np.load(name + '_X_test.npy')

    Y_tr = np.load(name + '_Y_train.npy')
    Y_te = np.load(name + '_Y_test.npy')

    segment_count_te = np.load(name + '_segmented_count_test.npy')
    segment_count_tr = np.load(name + '_segment_count_train.npy')

    cn = GenreCNN(batch_size=bs)

    n_te = Y_te.shape[0]

    cn.fit(X_tr, Y_tr, X_te, Y_te, segment_count_tr, segment_count_te)

    # cn.build_model()
    cn.fit_lstm(X_tr, Y_tr, X_te, Y_te, segment_count_tr, segment_count_te)
    prediction = cn.predict(X_te)
    ac = cn.get_accuracy(Y_te, prediction)

    cm = cn.get_cm(Y_te, prediction)

    print(cm)

    print(ac)

if __name__ == '__main__':
    main()
