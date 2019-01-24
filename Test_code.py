import numpy
import keras
import tensorflow
import os
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU, Conv1D,LocallyConnected1D
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from matplotlib import pyplot as pyplot
import numpy as np


import pulse_sim
import util


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


@util.show_run_time
def get_examples(num_examples, L):
    pulse_shape = pulse_sim.make_pulse_shape()
    histos = []
    results = []
    
    for i in range(num_examples):
        histo = pulse_sim.make_histogram()
        rate = 1/250
        sim = pulse_sim.make_simulation(L, histo, pulse_shape, rate, 0.005)
        k=0
        sim_temp= []
        for j in range(0,3996):
            if(sim[j+4]-sim[j]>=0.2):
                x=sim[j+4]-sim[j]
                sim_temp.append(x)
        sim_temp=np.pad(sim_temp,(0,150-len(sim_temp)),'constant')
        sim_temp=sorted(sim_temp)
        histos.append(histo)
        results.append(sim_temp)
    results = numpy.array(results)[:,:,numpy.newaxis]
    histos = numpy.array(histos)
    print(results.shape)
    return results, histos
    

def main():
    numpy.random.seed(1)
    
    num_examples = 5000
    default_length = 40000
    num_tests = 5
    
    # generate test examples
    test_vals, test_histos = get_examples(num_tests, default_length)
    
    # generate training examples
    examples_vals, examples_histos = get_examples(num_examples, default_length)
    
    # set up the model and its layers
    model = Sequential()
    input_shape = examples_vals.shape[1:]
    num_histo_bins = examples_histos.shape[1]
    
    model.add(Conv1D(1, 5, strides=1, activation='relu',
        input_shape=input_shape))
    model.add(PReLU())
    #model.add(Conv1D(1, 5, strides=5, activation='relu'))
    keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
    model.add(Conv1D(1, 5, strides=2, activation='relu'))
    keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
    model.add(LocallyConnected1D(1, 5, strides=2, activation='relu'))
    keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='valid')
    model.add(LSTM(num_histo_bins, activation='relu'))
    
    # compile
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    # fit
    model.fit(examples_vals,
        examples_histos, epochs=50, batch_size=100)
        
    # save weights
    model.save('model.dat')
    
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
    
    