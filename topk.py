import keras
import tensorflow as tf
import numpy
from keras.models import Sequential
from keras.layers import *

N = 100
L = 1000
def sort(x):
    value,indices = tf.nn.top_k(x,k=400,sorted=True,name=None)
    return value
    
examples_in = numpy.random.normal(0, 1, (N, L))
examples_out = numpy.random.normal(0, 1, (N, 400))

model = Sequential()
model.add(Lambda(sort, input_shape=(L,)))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(examples_in, examples_out, epochs=10, batch_size=10)
model.save('topk_model.dat')
