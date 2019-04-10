import numpy as np
import librosa as lb


# a = np.load('Y_train_indian_3.npy')
#
# print(a)

g = lb.load('petta.wav')

print(g[1])

lb.output.write_wav('song.wav', np.load('data/indian_fake_X_test.npy')[18], sr=22050)