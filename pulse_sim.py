import numpy
import scipy.signal
from matplotlib import pyplot as pyplot



DECAY_COEFS = numpy.array([-0.01, -0.1])
HISTO_BINS = 100
PULSE_UPPER_LENGTH = int(numpy.log(1e-6)/DECAY_COEFS[0])


def make_pulse_shape():
    N = PULSE_UPPER_LENGTH
    x = numpy.linspace(0, N-1, N)
    y = numpy.exp(x * DECAY_COEFS[0])
    y /= numpy.sum(y)
    return y
    

def make_pulse_shape_dbl():
    N = PULSE_UPPER_LENGTH
    x = numpy.linspace(0, N-1, N)
    y0 = numpy.exp(x * DECAY_COEFS[0])
    y1 = numpy.exp(x * DECAY_COEFS[1])
    y = y0 - y1
    y /= numpy.sum(y)
    return y
    
    
def make_histogram():
    x = numpy.linspace(0, HISTO_BINS - 1, HISTO_BINS)
    h = numpy.zeros(HISTO_BINS)
    num_gaussians = numpy.random.randint(1, 6)
    for i in range(num_gaussians):
        std = 5
        pos = numpy.random.uniform(15, 85)
        height = numpy.random.uniform(1, 5)
        h += numpy.exp(-(x-pos)**2/2/std)*height
    h = h / numpy.sum(h)
    return h


def make_simulation(num_samples, histo, pulse_shape, pulse_rate, noise_std):
    T = PULSE_UPPER_LENGTH + num_samples
    # generate the arrival times
    num_events = numpy.random.poisson(T * pulse_rate)
    events = numpy.random.uniform(0, T, num_events)
    heights = numpy.random.choice(len(histo), size=num_events, p=histo)
    # produce time series with spikes
    spikes = numpy.zeros(T)
    for i in range(len(events)):
        spikes[int(events[i])] += heights[i]
    # convolve with the exponential
    result = scipy.signal.fftconvolve(spikes, pulse_shape)
    # cut to the right size
    result = result[PULSE_UPPER_LENGTH:PULSE_UPPER_LENGTH + num_samples]
    noise = numpy.random.normal(0, 1, num_samples) * noise_std
    result = result + noise
    return result


def main():
    numpy.random.seed(1)
    pulse_shape = make_pulse_shape_dbl()
    histogram = make_histogram()
    sim1 = make_simulation(2000, histogram, pulse_shape, 1/250, 0.005)
    #sim2= sim1[1::5]
    
   # pyplot.plot(pulse_shape)
    pyplot.figure()
    pyplot.plot(histogram)
    pyplot.show()
    #pyplot.figure()
   # pyplot.figure()
    pyplot.plot(sim1)
    pyplot.show()
    
    
if __name__ == "__main__":
    main()
    