import numpy
import keras
import tensorflow
import os
from keras.models import Sequential
from keras.layers import *
from keras.layers.advanced_activations import PReLU
from keras.layers.advanced_activations import ThresholdedReLU
from keras.optimizers import Adam
from matplotlib import pyplot as pyplot
import tensorflow as tf
from keras.layers import Lambda



import pulse_sim
import util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def main():
    numpy.random.seed(1)
    
    num_examples = 7500
    default_length = 40000
    num_tests = 20
    
    # generate test examples
    test_vals, test_histos = util.get_examples(num_tests, default_length, True)
    
    # generate training examples
    examples_vals, examples_histos = util.get_examples(num_examples, default_length, True)
    
    # set up the model and its layers
    def sort(x):
        value,indices = tf.nn.top_k(x,k=400,sorted=True,name=None)
        return value
    def expand(x):
        value= tf.expand_dims(x,-1)
        return value
    def bins(x):
        results=tf.scalar_mul(10,x)
        return results

    model = Sequential()
    input_shape = examples_vals.shape[1:]
    num_histo_bins = examples_histos.shape[1]

    model.add(Conv1D(1, 5, strides=1, activation='relu',input_shape=input_shape))


    #model.add(MaxPooling1D(32))

    model.add(Flatten())
   # model.add(ThresholdedReLU(theta=0.15))
    
    model.add(Lambda(sort))
   # model.add(Dense(300, bias_initializer='zeros', activation='relu'))
   # model.add(Dense(250, bias_initializer='zeros', activation='relu'))
   # model.add(Dense(200, bias_initializer='zeros', activation='relu'))
   # model.add(Reshape((1,83)))
   # model.add(Lambda(bins))
    #model.add(Lambda(expand))
   # model.add(Bidirectional(LSTM(100,activation='relu',return_sequences=True)))
   # model.add(LSTM(num_histo_bins, activation='relu'))
    model.add(Dense(400, bias_initializer='zeros', activation='relu'))
    model.add(Dense(500, bias_initializer='zeros', activation='relu'))
    model.add(Dense(500, bias_initializer='zeros', activation='relu'))
    model.add(Dense(300, bias_initializer='zeros', activation='tanh'))
    model.add(Dense(100, bias_initializer='zeros', activation='softmax'))
    #model.add(Lambda(expand))
    #model.add(Bidirectional(LSTM(100,activation='relu',return_sequences=True)))
    #model.add(LSTM(num_histo_bins, activation='relu'))
   
   


    
    
    # compile
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    # fit
    history=model.fit(examples_vals,
        examples_histos, epochs=500, batch_size=100)
        
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

    # Plot training & validation loss values
    pyplot.plot(history.history['loss'])
    pyplot.title('Model loss')
    pyplot.ylabel('Loss')
    pyplot.xlabel('Epoch')
    pyplot.legend(['Train'], loc='upper left')
    pyplot.show()







    
    '''
    model.add(Conv1D(1, 5, strides=3, activation='relu',
        input_shape=input_shape))
    model.add(MaxPooling1D(256))
    model.add(PReLU(shared_axes=[1]))
    #model.add(Flatten())
    #model.add(RepeatVector(num_histo_bins))
    #model.add(PReLU(shared_axes=[1]))
    model.add(Bidirectional(LSTM(100,activation='relu',return_sequences=True)))
    model.add(LSTM(num_histo_bins, activation='relu'))

    # model.add(Dense(300, bias_initializer='zeros', activation='relu'))
   # model.add(Dense(400, bias_initializer='zeros', activation='relu'))
   # model.add(Dense(200, bias_initializer='zeros', activation='relu'))
   # model.add(Dense(300, bias_initializer='zeros', activation='relu'))
   # model.add(Dense(100, bias_initializer='zeros', activation='relu'))



 
    '''

if __name__ == "__main__":
    main()
    
