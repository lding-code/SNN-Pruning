# STDP unsupervised SNN learning model
# Pruned
# Digit recognition (MNIST)

import numpy as np 
import struct
import matplotlib.pyplot as plt

from brian2 import *

# read in test data

batch_tot = 1

data_test = []

n_input = 0
n_sample = 0

for batch_index in range(batch_tot):
    with open("./dataset/MNIST/pst_test_{0}.txt".format(batch_index), "r") as f:
        n_inputNeuron, n_newSample = [int(x) for x in next(f).split()]
        if (batch_index == 0):
            n_input = n_inputNeuron
        n_sample = n_sample + n_newSample
        for line in f:
            data_test.append([float(x) for x in line.split()])


# read in MNIST labels

label_test = []
n_sample = 10000

with open("./dataset/MNIST/test_label.txt", "r") as f:
    for line in f:
        label_test.append(int(line))


# read in neuron class label
neuron_class = []
with open("./neuron_class.txt", "r") as f:
    n_neuron = int(next(f))
    for line in f:
        neuron_class.append(int(line))

prun_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

accuracies = []
accuracies_per_number = []

for prun in prun_ratio:

    print("Current pruning ratio: {0}".format(prun))

    # read in synapse weight
    synapse_weight = []
    with open("./input_synapse_weight_pruned_{0}.txt".format(prun), "r") as f:
        n_synapse = [int(x) for x in next(f).split()]
        for line in f:
            newdata = [float(x) for x in line.split()]
            newdata[:2] = [int(x) for x in newdata[:2]]
            synapse_weight.append(newdata)


    # Set up parameter for modle training
    # Network Construction
    start_scope()


    prefs.codegen.max_cache_dir_size = 0

    # Model parameters ===========================================================
    # size of first layer
    n_input = 784

    # synapse weight threshold for pruning
    w_th = 0.5

    # neuron parameters *******************************************************
    # input neuron  -----------------------------------------------------------
    # input spike rate multiplier
    rate_multiplier = 10*Hz

    # excitatory neuron parameters  ---------------------------------------------
    # number of excitatory neurons
    n_exc = 100

    # membrane time constant
    tau_exc = 50*ms

    # neuron dynamics
    eqs_exc = '''
    dv/dt = -v/tau_exc : 1 (unless refractory)
    vt_exc : 1
    n_fire : 1
    '''

    # inhibitory neuron parameters ---------------------------------------------
    # number of inhibitory neurons
    n_inh = 100

    # membrane time constant
    tau_inh = 50 * ms

    # threshold
    vt_inh = 0.2

    # neuron dynamics
    eqs_inh = '''
    dv/dt = -v/tau_inh : 1
    vt : 1
    '''

    # synapse parameters  ***********************************************

    # input synapses ---------------------------------------------------
    # neuron potential scaler
    scaler = 5 / 2500

    # synapse weight learning rate
    eta = 0

    # weight change offset (how much negative weight change to have)
    # synapse dynamics
    synapse_dynamic = '''
    w : 1
    '''

    synapse_pre_action = '''
    v_post += w * scaler
    '''

    # excitatory synapses ----------------------------------------------------
    # synapse weight
    exc_w_0 = 1

    # synapse dynamics
    exc_dynamic = '''
    exc_w : 1
    '''

    # synapse on pre action
    exc_pre_action = '''
    v_post += exc_w
    '''

    # inhibitory synapses ----------------------------------------------------
    # synapse weight
    inh_w_0 = 0.0

    # synapse dynamics
    inh_dynamic = '''
    inh_w : 1
    '''

    # synapse on pre action
    inh_pre_action = '''
    v_post = 0
    '''

    # Set up neurons ==========================================================
    # set up input layer ******************************************************

    input_layer = PoissonGroup(n_input, data_test[0]*rate_multiplier)

    # set up excitatory layer *************************************************

    exc_layer = NeuronGroup(
        n_exc, 
        eqs_exc, 
        threshold="v>vt_exc", 
        refractory="0*ms", 
        reset="v=0", 
        method="exact", 
        events={"fire_record" : "v>vt_exc"})

    exc_layer.vt_exc = 0.1

    # neuron action on spike firing event
    exc_layer.run_on_event("fire_record", '''n_fire += 1''')

    # set up inhibitory layer **************************************************

    inh_layer = NeuronGroup(
        n_inh,
        eqs_inh,
        threshold = "v>vt_inh",
        refractory = "3*ms",
        reset="v=0",
        method="exact"
    )


    # Set up synapses =============================================================
    # Synapses from input layer to excitatory layer *******************************
    input_s = Synapses(
        input_layer, 
        exc_layer, 
        synapse_dynamic,
        method = "exact",
        on_pre = synapse_pre_action)

    input_s.connect()

    # Initialize / Randomize input synapses
    # for src_index in range(n_input):
    #     for dst_index in range(exc_layer.N):
    #         input_s.w[src_index, dst_index] = eta * uniform(0.1, 0.9)
    for synapse_info in synapse_weight:
        input_s.w[synapse_info[0], synapse_info[1]] = synapse_info[2]

    # Synapses from exc to inh ****************************************************
    exc_s = Synapses(
        exc_layer,
        inh_layer,
        exc_dynamic,
        on_pre = exc_pre_action
    )

    exc_s.connect(condition = "i == j")

    # initialize weight
    exc_s.exc_w = exc_w_0

    # Synapses from inh to exc **************************************************
    inh_s = Synapses(
        inh_layer,
        exc_layer,
        inh_dynamic,
        on_pre = inh_pre_action
    )

    inh_s.connect(condition = "i != j")

    # initialize weight
    inh_s.inh_w = inh_w_0




    # prepare variable vector

    neuron_activity = np.zeros(100)

    pred_label = []

    neuron_activity[:] = -1

    # feature-training model

    step_size = 10000

    start = 0

    stop = start + step_size

    for sample in range(start, stop):    
        # set input layer rates
        input_layer.rates = data_test[sample] * rate_multiplier

        # learning rate decay after first 100 samples
        #if sample > 100:
        #    eta = eta / sqrt(sample - 100 )

        # reset last pre-synaptic neuron fire time to 0
        #input_s.pre_last = 0*ms

        # reset voltage to 0
        exc_layer.v = 0

        # reset fire time
        exc_layer.n_fire = 0

        # reset total_offset
        #input_s.total_offset = 0

        # reset neuron activity
        neuron_activity[:] = 0
        
        run(100*ms)

        for exc_index in range(100):
                neuron_activity[exc_index] = exc_layer.n_fire[exc_index]
        
        pred_label.append(neuron_class[np.argmax(neuron_activity)])

        if sample % 1000 == 0:
            print("{0}% completed".format(100 * sample / 10000))

    tot_per_number = np.zeros(10)
    correct_per_number = np.zeros(10)
    accuracy_per_number = np.zeros(10)

    correct_flag = np.zeros(10000)

    for sample in range(10000):
        tot_per_number[label_test[sample]] = tot_per_number[label_test[sample]] + 1
        if pred_label[sample] == label_test[sample]:
            correct_flag[sample] = 1
            correct_per_number[label_test[sample]] = correct_per_number[label_test[sample]]  + 1
            
    accuracy_per_number = correct_per_number / tot_per_number

    accuracy = np.sum(correct_flag) / 10000

    accuracies.append(accuracy)
    accuracies_per_number.append(accuracy_per_number)

