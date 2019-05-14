"""
PLEASE DOCUMENT HERE

Usage: python3 project3.py DATASET.csv
"""

import csv, sys, random, math, datetime
from multiprocessing import Pool
import numpy as np

STATIC_ALPHA = 0.01
DYNAMIC_ALPHA_CONSTANT = 1000
EPOCHS = 6000
total_so_far = 0



def read_data(filename, delimiter=",", has_header=True):
    """Reads datafile using given delimiter. Returns a header and a list of
    the rows of data."""
    data = []
    header = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            header = next(reader, None)
        for line in reader:
            example = [float(x) for x in line]
            data.append(example)

        return header, data

def convert_data_to_pairs(data, header):
    """Turns a data list of lists into a list of (attribute, target) pairs."""
    pairs = []
    for example in data:
        x = []
        y = []
        for i, element in enumerate(example):
            if "target" in header[i]:
                y.append(element)
            else:
                x.append(element)
        pair = (x, y)
        pairs.append(pair)

    # If there needs to be more than 2 classes,
    # converts numbers to arrays of zeros except for the corresponding element
    multiclass = False
    min_val = np.inf
    max_val = -np.inf
    for (x, y) in pairs:
        min_val = min(min_val, y[0])
        max_val = max(max_val, y[0])
        if y[0] > 1:
            multiclass = True

    if multiclass:
        num_classes = int(max_val + 1 - min_val)
        for i in range(len(pairs)):
            pair = pairs[i]
            x,y = pair

            y_class = int(y[0] - min_val)
            pairs[i] = (x, [0] * num_classes)
            pairs[i][1][y_class] = 1
    return pairs

def dot_product(v1, v2):
    """Computes the dot product of v1 and v2"""
    assert len(v1) == len(v2)
    _sum = 0
    for i in range(len(v1)):
        _sum += v1[i] * v2[i]
    return _sum

def logistic(x):
    """Logistic / sigmoid function"""
    try:
        denom = (1 + math.e ** -x)
    except OverflowError:
        return 0.0

    return 1/denom
    

def accuracy(nn, pairs):
    """Computes the accuracy of a network on given pairs. Assumes nn has a
    predict_class method, which gives the predicted class for the last run
    forward_propagate. Also assumes that the y-values only have a single
    element, which is the predicted class.

    Optionally, you can implement the get_outputs method and uncomment the code
    below, which will let you see which outputs it is getting right/wrong.

    Note: this will not work for non-classification problems like the 3-bit
    incrementer."""

    true_positives = 0
    total = len(pairs)

    for (x, y) in pairs:
        class_prediction = nn.predict_class(x)
        if len(nn.output_layer) == 1:
            if class_prediction != y[0]:
                true_positives += 1
        # Modification to handle multi-class
        else:
            if y[class_prediction] != 1:
                true_positives += 1

        # outputs = nn.get_outputs()
        # print("y =", y, ",class_pred =", class_prediction, ", outputs =", outputs)

    return 1 - (true_positives / total)

### Helpers
def dynamic_alpha(count, const):
    return const / (const + count)

def static_alpha(count, const):
    return const

def k_groups(data, k):
    """Divides the data into k approximately equal-sized groups"""
    random.shuffle(data)
    groups = []
    for i in range(k):
        groups.append(data[int(i*len(data)/k):int((i+1)*len(data)/k)])
    return groups



class Layer():
    """ Input or hidden layer for a neural network"""
    def __init__(self, size, next_size):
        """Initializes a layer given the size of the layer and the size of the next layer"""
        
        # Create random (size+1) x next_size array of random numbers
        self.weights = np.random.rand(size + 1, next_size)
        self.activations = []
        self.deltas = []

    def __len__(self):
        return len(self.weights)

    def connect(self, layer):
        self.next_layer = layer

    def activate(self, activations):
        """Propagate activations to the next layer"""
        self.activations = activations

        # Compute the propagation to each node in the next layer
        propagation = np.matmul(self.activations, self.weights)

        # Apply logistic function to each
        propagation = np.apply_along_axis(logistic, 0, propagation)
        if not isinstance(self.next_layer, OutputLayer):

            # Add the constant 1 node to the propagation
            propagation = np.append(propagation, [1])
        
        self.next_layer.activate(propagation)

    def compute_deltas(self):
        """Computes deltas for backpropagation.
         Assumes that the next layer's deltas are current."""
        self.deltas = []

        # This does the summation computation in one step
        sums = np.matmul(self.weights, self.next_layer.deltas)
        for j, a in enumerate(self.activations[:-1]):
            delta = a*(1-a)*sums[j]
            self.deltas.append(delta)


class OutputLayer():
    """Output layer. Doesn't need a lot of functionality except for computing deltas."""
    def __init__(self, size):
        self.size = size
        self.activations = []
        self.deltas = []

    def activate(self, activations):
        self.activations = activations

    def compute_deltas(self, y):
        """Compute deltas for output layer"""
        self.deltas = []
        a = self.activations
        for j in range(len(y)):
            delta_j = a[j] * (1-a[j]) * (y[j] - a[j])
            self.deltas.append(delta_j)

    def __len__(self):
        return self.size

    def __getitem__(self, ind):
        return self.activations[ind]

class NeuralNetwork():
    def __init__(self, layer_sizes, alpha_fn, alpha_const):
        """Initializes a neural network given layer sizes and a function and constant used to compute the alpha for each layer"""


        hidden_layer_sizes = layer_sizes[:-1]
        output_layer_size = layer_sizes[-1]
        self.num_layers = len(layer_sizes)
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(size = layer_sizes[i], next_size = layer_sizes[i+1]))
        

        self.input_layer = self.layers[0]
        self.output_layer = OutputLayer(output_layer_size)
        self.layers.append(self.output_layer)


        # Alpha_fn is either static_alpha or dynamic_alpha. 
        # Both take in the epoch number but only dynamic_alpha uses it.
        # I'm doing it this way so that it is picklable
        self.alpha_fn = alpha_fn
        self.alpha_const = alpha_const

        for i in range(len(self.layers) - 1):
            self.layers[i].connect(self.layers[i+1])

    def predict_class(self, _input):
        """Predict the class of an input"""
        self.eval(_input)

        if len(self.output_layer) == 1:
            # If only 2 classes, just round the output node
            return round(self.output_layer[0])
        else:
            max_index = 0
            max_val = 0
            for i, val in enumerate(self.output_layer.activations):
                if val > max_val:
                    max_index = i
                    max_val = val
            return max_index

    def get_alpha(self, count):
        return self.alpha_fn(count, self.alpha_const)

    def forward_propagate(self, data):
        """Given a piece of data, adds constant 1 to it and then sends it to the input layer"""
        data = np.asarray(data)
        data = np.append(data, [1])
        self.input_layer.activate(data)

    def eval(self,data):
        self.forward_propagate(data)
        return self.get_outputs()

    def get_outputs(self):
        return self.output_layer.activations

    def backpropagate(self, y):
        """Backpropagates error deltas"""

        # Do this one separately because it requires a parameter
        self.output_layer.compute_deltas(y)
        for layer in self.layers[::-1][1:]:
            layer.compute_deltas()

    def update_weights(self, alpha):
        """Updates weights based on deltas"""

        for layer in self.layers[:-1]:
            for i, a_i in enumerate(layer.activations):
                for j in range(len(layer.weights[0])):
                    delta_j = layer.next_layer.deltas[j]
                    layer.weights[i,j] += alpha * a_i * delta_j

    def train(self, data, max_epochs, time = np.inf):
        """Trains using a dataset"""
        start_time = datetime.datetime.now()
        num_epochs = 0
        happy = False

        while not happy:

            for x,y in data:
                self.forward_propagate(x)
                self.backpropagate(y)

                # I lost hours of my life because this line of code was unindented
                self.update_weights(alpha = self.get_alpha(num_epochs))
                
            num_epochs += 1
            
            time_elapsed = datetime.datetime.now() - start_time
            if num_epochs > max_epochs or (time != np.inf and time_elapsed > time):
                break
            print("{0:05d}".format(num_epochs), end = '\r')

def map_to_accuracy(args):
    """Helper function to allow me to pass a single parameter to map function for mp"""
    grouped_pair, structure, alpha_fn, alpha_const, epochs = args
    
    nn = NeuralNetwork(structure, alpha_fn, alpha_const)
    training_data, test_data = grouped_pair
    nn.train(training_data, epochs)
    
    return accuracy(nn, test_data)

def eval_k_groups(structure, k, pairs, alpha_fn, alpha_const, epochs):
    """Performs k-fold cross-validation"""
    grouped_params = []
    accs = []
    groups = k_groups(data = pairs, k = k)
    for i in range(len(groups)):
        test_data = groups[i]

        # Flattened list of all data not in the test set
        training_data = [x for group in groups for x in group if group != test_data]
        grouped_params.append(((training_data, test_data), structure, alpha_fn, alpha_const, epochs))

    with Pool(4) as pool:
        acc_list = pool.map(map_to_accuracy, grouped_params)

    avg_accuracy = sum(acc_list)/len(acc_list)
    return avg_accuracy

def main():
    """ Train neuralnetwork(s)"""

    # Run parameters --------------------------------------------
    epoch_counts = [50,1000]

    # This only determines the size of the hidden layer(s) since the input and output layers are created based on the data.
    hidden_structs = [[x] for x in range(2,8)] + [[4,4]]
    static_alpha_consts = [.1,.01]

    # Dynamic alpha: X/(X+count)
    dynamic_alpha_consts = [1000,5000]
    output_filename = "wineoutput.csv"
    filenames = ["wine.csv"]
    num_groups = 10
    # -----------------------------------------------------------
    
    alpha_fns = ([(static_alpha, const) for const in static_alpha_consts] + 
            [(dynamic_alpha, const) for const in dynamic_alpha_consts])

    # This allows the alpha functions to be paired with a nice string representation for the CSV/spreadsheet
    alpha_displays = [str(alpha) for alpha in static_alpha_consts] + ["{}/{}+count".format(d_a,d_a) for d_a in dynamic_alpha_consts]

    alpha_tuples = [(alpha_fns[i][0], alpha_fns[i][1], alpha_displays[i]) for i in range(len(alpha_displays))]    

    possible_runs = [(epoch_count, struct, alpha_tuple) for struct in hidden_structs for alpha_tuple in alpha_tuples for epoch_count in epoch_counts]

    

    with open(output_filename, "a") as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        # writer.writerow(["Dataset", "Structure","Alpha","Epochs", "Accuracy"])
    for file_index,data_filename in enumerate(filenames):
        header, data = read_data(data_filename, ",")
        pairs = convert_data_to_pairs(data, header)
        print('\n')

        for run_index,run_info in enumerate(possible_runs):
            print("          File:{}, run {} of {}".format(data_filename, run_index + 1, len(possible_runs)), end = '\r')
            epochs, struct, alpha_tuple = run_info

            alpha_fn, alpha_const, alpha_display = alpha_tuple

            # Determines the length of the input and output layers from the data
            input_len = len(pairs[0][0])
            output_len = len(pairs[0][1])
            structure = [input_len] + struct + [output_len]
            
            accuracy = eval_k_groups(structure = structure, alpha_fn = alpha_fn, alpha_const = alpha_const,
                                            k = num_groups, pairs = pairs, epochs = epochs)

            # Reopening the file each time so that if my stupid laptop BSOD's while I'm running tests I don't lose the data (I think).
            with open(output_filename, "a") as csvfile:
                writer = csv.writer(csvfile, delimiter = ',')
                writer.writerow([data_filename, structure, alpha_display,epochs, accuracy])
if __name__ == "__main__":
    main()
