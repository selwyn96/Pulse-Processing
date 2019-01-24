import faulthandler; faulthandler.enable()
import numpy
import keras
import sys
import os
from matplotlib import pyplot as pyplot
import tensorflow as tf

import util


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def main():
    numpy.random.seed(1)
    
    num_tests = 20
    default_length = 40000
    
    # generate test examples
    test_vals, test_histos = util.get_examples(num_tests, default_length, True)
    
    # load model
    model = keras.models.load_model('model_saved.dat', custom_objects={'tf':tf})
    print(model)
    
    # evaluate
    score = model.evaluate(test_vals, test_histos, batch_size=10)
    predictions = model.predict(test_vals, batch_size=10)
    print("evaluation score:", score)
    
    # plot results
    for i in range(num_tests):
        pyplot.figure()
        pyplot.plot(test_histos[i,:])
        pyplot.plot(predictions[i,:])
        pyplot.savefig("test{}.png".format(i))
    pyplot.show()
    


if __name__ == "__main__":
    main()
    
    