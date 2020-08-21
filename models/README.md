There are 3 models in this folder.
Please check attached png images for reference
Attached jupyter notebook file has simulation for all 3 networks as well as code that used to generate input and network file for model 2 and model 3. Please check the jupyter notebook for details.

## Input format

I have made one important change to the mode, I didn't use temporally recorded spike train as input, it takes too much space once the input gets larger (6GB for MNIST with very low precision). Although it works for smaller input data, it's not a sustainable method. Instead, I used fire rate model, now the input file simply records fire rate of each pixel baed on its intensity. It follows the equation: fire_rate = intensity / 255 * max_rate. The simulator then generate a Poisson spike train locally at time of simmulating/training/classifying based on the fire rate. For example, if a pixel ahs intensity of 128 and max rate is set to 100Hz, its corresponding neuron then fires spikes at (128/255) * 100Hz ~= 50Hz. This translates to an average of 50 spikes every second / 1 spike every 20ms. The simulating platform then decides where to put these randomly generated spikes based on its simulation time step. I used txt file instead of csv for the same reason, csv take too much space.

## Model 1

Model 1 is the simplest one, it consists of only 3 neurons. They are connected one after another. The specs (input info: spike fire rate, neuron info: potential equation including threshold and leak time constant, synapse info: weight) can be found in the image. This network is too simple to have a file of specs. This is used to experiment with the basics.

## Model 2

Model 2 is 2-layer network. Its input layer has 4 neurons and output layer has 2 neurons. Specs can be seen in the image.
In the folder "model2", there are 3 files: input2.txt, neuron2.txt, synapse2.txt

**input2.txt** contains fire rate of the input neurons (aka Poisson spike train)

The first line has two numbers: number of input neuron, number of input samples.

The rest lines each has fire rates for input neurons for each input sample.

**neuron2.txt** contains information of the rest neurons

The first line contains one number: number of layers (n_layer, including 0th/input layer)

The second line contains n_layer numbers: each representing number of neurons of the corresponding layer (adding them together will give total number of all neurons)

Starting from the third line, each line contains two number for each neuron: tau(time constant), theta(threshold)

Neuron info is listed from first neuron in first layer, excluding input layer because input info is in the input file, to last neuron in last layer. (Maybe I can delete the number of neuron in the input layer, it's confusing and it's already in the input file...)

**synapse2.txt** contains weight information of synapses

The first line contains one number: number of synpases in the whole network

Starting from the second line, each line has 3 numbers: pre-synaptic(srouce) neuron index, post-synaptic(destination) neuron index, weight

Each line describes a synapse connection. Neuron is indexed from input layer to last layer. In model2, there are 4 neurons in input layer, their indices are [0, 1, 2, 3], following the input layer, there are 2 neurons in output layer, their indices are [4, 5].

**Note** In neuron2.txt, the first neuron entry has index of 4 because input neuron (0 to 3) are NOT included in the neuron2.txt

## Model 3

Model 3 has 3 layer. It has 9 input neurons, 5 excitatory neurons and 5 inhibitory neurons.

Model 3 has the same 3 files in the folder "model3" in the same format. In addition to difference in number of neurons and synapses, this model has inhibitory layer, which means the synapses from inhibitory layer has negative weight. You can observe some very interesting suppression phenomenon.

**Note** All parameters of model 3 are randomly generated for now.

## MNIST data set

pst_train_full.txt contains 60000 samples in the format as other input.txt

## PST_generator.ipynb

The first half of the notebook is about encoding MNIST dataset into spike train / fire rate.
The last half of the notebook consists of simulation results for 3 models and code blocks that output spike input and netwokr parameter files. Please use comment as reference.
