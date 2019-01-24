import numpy
import pulse_sim
import time
import functools


def show_run_time(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        before = time.perf_counter()
        result = f(*args, **kwargs)
        after = time.perf_counter()
        duration = after - before
        print("Run time for {}: {}".format(f.__name__, duration))
        return result
    return wrapper


@show_run_time
def get_examples(num_examples, L, use_dbl = False):
    if use_dbl:
        pulse_shape = pulse_sim.make_pulse_shape_dbl()
    else:
        pulse_shape = pulse_sim.make_pulse_shape()
    histos = []
    results = []
    for i in range(num_examples):
        histo = pulse_sim.make_histogram()
        rate = 1/250
        sim = pulse_sim.make_simulation(L, histo, pulse_shape, rate, 0.005)
        sim_1=sim[0::5]
        
        histos.append(histo)
        results.append(sim_1)
    results = numpy.array(results)[:,:,numpy.newaxis]
    histos = numpy.array(histos)
    return results, histos
    
