import numpy
import keras
import os
from matplotlib import pyplot as pyplot

import pulse_sim
import util


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def get_intermediate(model, layer_num, test_vals):
    layer = model.layers[layer_num]
    layer_model = keras.models.Model(inputs=model.input,
        outputs=layer.output)
        
    intermediate = layer_model.predict(test_vals)
    print(intermediate.shape)
    intermediate = intermediate.reshape(intermediate.shape[1:2])
    return intermediate


def main():
    numpy.random.seed(1)
    
    default_length = 20000
    num_tests = 1
    
    # generate test examples
    test_vals, test_histos = util.get_examples(num_tests, default_length)
    
    # load the model
    model = keras.models.load_model('model.dat')
    
    def show_layer_output(i):
        intermediate = get_intermediate(model, i, test_vals)
        pyplot.plot(intermediate, label="Layer {} output".format(i))
    
    pyplot.figure()
    pyplot.plot(test_vals[0,:], label="Input")
    for i in [0, 1]:
        show_layer_output(i)
    pyplot.legend()
    
    pyplot.figure(); show_layer_output(1); pyplot.legend()
    #pyplot.figure(); show_layer_output(); pyplot.legend()


    
    pyplot.figure()
    for i in [3]:
        show_layer_output(i)
    pyplot.plot(test_histos[0,:], label="Ideal output")
    pyplot.legend()
    
    pyplot.show()
    

if __name__ == "__main__":
    main()
    

