import numpy as np
import tensorflow as tf
import librosa as lb
import seaborn as sn
import math
from numba.targets.arraymath import np_all
import sklearn
from tensorflow.python.training import optimizer

from data_preparation import *
np.random.seed(1234)
tf.set_random_seed(1234)


class GenreCNN:

    def __init__(self, preprocess=False, class_names=None,
                 mel=True, stft=False,
                 batch_size=5,
                 max_itrns=100,
                 n_classes=4,
                 save_path='saved_models_genre'):

        self.mel = mel
        self.stft = stft
        self.batch_size = batch_size
        self.input_h = 128
        self.input_w = 1293
        self.max_itrns = max_itrns
        self.n_classes = n_classes
        self.save_path = save_path
        self.sess = None

        if class_names:
            self.index_to_class = sorted(class_names)

    @staticmethod
    def extract_spectrogram(ts_data, ):

        mel_sg = lb.feature.melspectrogram(ts_data)
        return mel_sg

    @staticmethod
    def shuffle(X, Y):

        concat = np.hstack((X, np.reshape(Y, (Y.shape[0], 1))))
        np.random.shuffle(concat)

        return concat[:, 0:-1], concat[:, -1]

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

            class_scores = tf.keras.layers.Dense(self.n_classes)(pool4)

            self.class_scores = class_scores

        # with tf.Session() as sess:
        #
        #     sess.run(tf.global_variables_initializer())
        #
        #     a = np.zeros([self.batch_size, self.input_h, self.input_w, 1], np.float32)
        #
        #     ans = sess.run(a_shape, feed_dict={input: a})

        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.class_scores, labels=self.label_batch))

        self.global_step = tf.Variable(0, trainable=False)

        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)

        self.saver = tf.train.Saver(max_to_keep=4)

        print('boo')

    def train(self, X_te=None, Y_te=None):

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
            self.saver.restore(self.sess, tf.train.latest_checkpoint('saved_models/'))
            print('*    reloaded *')
        except:
            sess.run(tf.global_variables_initializer())
            print(' * couldnt reload * ')

        start = int(sess.run(self.global_step))

        for ei in range(start, self.max_itrns):

            in_b, l_b = sess.run([input_batch, label_batch])
            loss, _ = sess.run((self.loss, self.optimizer), {self.input_batch: in_b,
                                                             self.label_batch: l_b})

            print("loss: {}, batch {}".format(loss, ei))

            if ((ei + 1) % 10 == 0):
                print(' * saaved * ', ei)
                self.saver.save(sess, os.path.join(self.save_path, 'model.ckpt'),global_step=ei)


            if (ei + 1) % 50 == 0 and np.any(X_te) and np.any(Y_te):
                prediction = cn.predict(X_te)
                ac = cn.get_accuracy(Y_te, prediction)
                print(ac)

        coord.request_stop()
        coord.join(enqueue_threads)

        self.sess = sess

    def fit(self, X_train, Y_train, X_te=None, Y_te=None):

        self.X_train = X_train
        self.Y_train = Y_train

        self.X_train, self.Y_train = self.shuffle(self.X_train, self.Y_train)

        self.Y_train = np.eye(self.n_classes, dtype=np.float32)[self.Y_train.astype(np.int32)]

        time_limit = X_train.shape[0]

        if self.mel:
            extract_melsg_vectorized = np.vectorize(self.extract_spectrogram, otypes=[np.float32],
                                                    signature='(a)->(b,c)')
            self.X_train_sg = extract_melsg_vectorized(self.X_train)

        self.X_train_sg = np.expand_dims(self.X_train_sg, 3)

        self.build_model()
        self.train(X_te, Y_te)


    def predict(self, X):

        print(self.sess)

        if self.sess == None:
            self.sess = tf.Session()
            self.saver.restore(self.sess, tf.train.latest_checkpoint('saved_models/'))
            print('* reloaded *')

        if self.mel:
            extract_melsg_vectorized = np.vectorize(self.extract_spectrogram, otypes=[np.float32],
                                                    signature='(a)->(b,c)')
            X = extract_melsg_vectorized(X)

        X = np.expand_dims(X, 3)

        n_predictions = X.shape[0]
        predictions = np.zeros(n_predictions - n_predictions%self.batch_size, np.int32)


        for xi in range(n_predictions//self.batch_size):

            data = X[xi*self.batch_size: (xi + 1) * self.batch_size]

            class_scores = self.sess.run(self.class_scores, {self.input_batch: data})
            class_prediction = np.argmax(class_scores, axis=1)

            predictions[xi*self.batch_size: (xi + 1) * self.batch_size] = class_prediction

            print(xi*self.batch_size, (xi + 1) * self.batch_size)

        print(predictions)
        return predictions

    def get_accuracy(self, y_true, y_pred):


        print(y_true.shape, y_pred.shape)
        acc = sklearn.metrics.accuracy_score(y_true, y_pred)

        return acc

if __name__ == '__main__':

    # X_train, X_test, Y_train, Y_test = extract_ts('../mydata')

    bs = 5

    X_tr = np.load('X_train.npy')
    X_te = np.load('X_test.npy')
    Y_tr = np.load('Y_train.npy')
    Y_te = np.load('Y_test.npy')

    cn = GenreCNN(batch_size=bs)

    n_te = Y_te.shape[0]

    Y_te = Y_te[:n_te - n_te%bs]

    cn.fit(X_tr, Y_tr, X_te, Y_te)
    # cn.build_model()
    prediction = cn.predict(X_te)
    ac = cn.get_accuracy(Y_te, prediction)

    print(ac)