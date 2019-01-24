import numpy
import time
import functools
import keras
import tensorflow
import os
from keras.models import Sequential
from keras.layers import *
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from matplotlib import pyplot as pyplot

import pulse_sim
import util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def main():
    numpy.random.seed(3)
    
    num_examples = 10000
    default_length = 40000
    num_tests = 5
    
    # generate test examples
    test_vals, test_histos = util.get_examples(num_tests, default_length)
    
    # generate training examples
    examples_vals, examples_histos = util.get_examples(num_examples, default_length)
    
    # load the model
    model = keras.models.load_model('model.dat')
    
    # compile
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    # fit
    model.fit(examples_vals,
        examples_histos, epochs=10, batch_size=1000)
        
    # save weights
    model.save('model2.dat')
    
    # evaluate
    score = model.evaluate(test_vals, test_histos, batch_size=10)
    predictions = model.predict(test_vals, batch_size=10)
    print("evaluation score:", score)
    
    # plot results
    for i in range(num_tests):
        pyplot.figure()
        pyplot.plot(test_histos[i,:])
        pyplot.plot(predictions[i,:])
    pyplot.show()
    
    

if __name__ == "__main__":
    main()
    