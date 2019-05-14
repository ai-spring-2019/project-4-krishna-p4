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

    print(pairs[0])
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
        else:
            if y[class_prediction] != 1:
                true_positives += 1

        # outputs = nn.get_outputs()
        # print("y =", y, ",class_pred =", class_prediction, ", outputs =", outputs)

    return 1 - (true_positives / total)



################################################################################
### Neural Network code goes here

class Layer():
    def __init__(self, size, next_size):
        self.weights = np.random.rand(size + 1, next_size)
        self.activations = []
        self.deltas = []
        self.prediction_info = None
        # self.nodes = [Node() for _ in range(self.size)]
        # self.weights = [node.weights for node in self.nodes]

    def __len__(self):
        return len(self.weights)

    def connect(self, layer):
        self.next_layer = layer

    def activate(self, activations):

        self.activations = activations

        propagation = np.matmul(self.activations, self.weights)

        propagation = np.apply_along_axis(logistic, 0, propagation)


        if not isinstance(self.next_layer, OutputLayer):
            propagation = np.append(propagation, [1])
        
        self.next_layer.activate(propagation)


    def compute_deltas(self):
        self.deltas = []

        sums = np.matmul(self.weights, self.next_layer.deltas)
        for j, a in enumerate(self.activations[:-1]):
            delta = a*(1-a)*sums[j]
            self.deltas.append(delta)



class OutputLayer():
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
        # print("data:{} activations:{}\ndifference:{}".format(y,a,[y[j]-a[j] for j in range(len(y))]))
        for j in range(len(y)):
            delta_j = a[j] * (1-a[j]) * (y[j] - a[j])
            self.deltas.append(delta_j)

    def __len__(self):
        return self.size

    def __getitem__(self, ind):
        return self.activations[ind]


def dynamic_alpha(count, const):
    return const / (const + count)

def static_alpha(count, const):
    return const

def k_groups(data, k):
    random.shuffle(data)
    groups = []
    for i in range(k):
        groups.append(data[int(i*len(data)/k):int((i+1)*len(data)/k)])
    return groups


class NeuralNetwork():
    def __init__(self, layer_sizes, alpha_fn, alpha_const):
        hidden_layer_sizes = layer_sizes[:-1]
        output_layer_size = layer_sizes[-1]
        self.num_layers = len(layer_sizes)
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(size = layer_sizes[i], next_size = layer_sizes[i+1]))

        self.input_layer = self.layers[0]
        self.output_layer = OutputLayer(output_layer_size)
        self.layers.append(self.output_layer)
        self.alpha_fn = alpha_fn
        self.alpha_const = alpha_const


        for i in range(len(self.layers) - 1):
            self.layers[i].connect(self.layers[i+1])

    def predict_class(self, _input):

        self.eval(_input)
        if len(self.output_layer) == 1:
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
        data = np.asarray(data)
        data = np.append(data, [1])
        self.input_layer.activate(data)

    def eval(self,data):
        self.forward_propagate(data)
        return self.get_outputs()

    def get_outputs(self):
        return self.output_layer.activations

    def backpropagate(self, y):
        self.output_layer.compute_deltas(y)
        for layer in self.layers[::-1][1:]:
            layer.compute_deltas()

    def update_weights(self, alpha):

        for layer in self.layers[:-1]:

            for i, a_i in enumerate(layer.activations):

                for j in range(len(layer.weights[0])):

                    delta_j = layer.next_layer.deltas[j]
                    start = layer.weights[i,j]
                    layer.weights[i,j] += alpha * a_i * delta_j
                    end = layer.weights[i,j]


                    assert abs((end - start)- (alpha * a_i * delta_j)) < .00001

    def train(self, data, max_epochs, time = np.inf):
        start_time = datetime.datetime.now()
        num_epochs = 0
        not_happy_yet = True 

        while not_happy_yet:
            

            for x,y in data:
                self.forward_propagate(x)
                self.backpropagate(y)
                self.update_weights(alpha = self.get_alpha(num_epochs))
                
            num_epochs += 1
            
            time_elapsed = datetime.datetime.now() - start_time
            if num_epochs > max_epochs or (time != np.inf and time_elapsed > time):
                break
            print("{0:05d}".format(num_epochs), end = '\r')
    def print_weights(self):
        for layer in self.layers[:-1]:
            print(layer.weights)

def map_to_accuracy(args):
    grouped_pair, structure, alpha_fn, alpha_const, epochs = args
    nn = NeuralNetwork(structure, alpha_fn, alpha_const)
    training_data, test_data = grouped_pair
    nn.train(training_data, epochs)
    return accuracy(nn, test_data)

def eval_k_groups(structure, k, pairs, alpha_fn, alpha_const, epochs):
    grouped_pairs = []
    accs = []
    groups = k_groups(data = pairs, k = k)
    for i in range(len(groups)):
        test_data = groups[i]
        training_data = [x for group in groups for x in group if group != test_data]
        grouped_pairs.append(((training_data, test_data), structure, alpha_fn, alpha_const, epochs))

    with Pool(4) as pool:
        acc_list = pool.map(map_to_accuracy, grouped_pairs)

    avg_accuracy = sum(acc_list)/len(acc_list)
    # print("avg_accuracy:{}".format(avg_accuracy))
    return avg_accuracy

def run_en_masse():
    epoch_counts = [50,1000]

    hidden_structs = [[x] for x in range(2,8)]
    
    static_alpha_consts = [.1,.01]

    dynamic_alpha_consts = [1000,5000]
    
    alpha_fns = ([(static_alpha, const) for const in static_alpha_consts] + 
            [(dynamic_alpha, const) for const in dynamic_alpha_consts])
    
    alpha_displays = [str(alpha) for alpha in static_alpha_consts] + ["{}/{}+count".format(d_a,d_a) for d_a in dynamic_alpha_consts]

    alpha_tuples = [(alpha_fns[i][0], alpha_fns[i][1], alpha_displays[i]) for i in range(len(alpha_displays)) ]

    filenames = ["wine.csv"]

    possible_runs = [(epoch_count, struct, alpha_tuple) for struct in hidden_structs for alpha_tuple in alpha_tuples for epoch_count in epoch_counts]

    output_filename = "wineoutput.csv"

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

            input_len = len(pairs[0][0])
            output_len = len(pairs[0][1])
            structure = [input_len] + struct + [output_len]
             
            accuracy = eval_k_groups(structure = structure, alpha_fn = alpha_fn, alpha_const = alpha_const, k = 10, pairs = pairs, epochs = epochs)
            with open(output_filename, "a") as csvfile:
                writer = csv.writer(csvfile, delimiter = ',')
                writer.writerow([data_filename, structure, alpha_display,epochs, accuracy])

def main():
    run_en_masse()
    return
    epochs = EPOCHS
    hidden_structure = [7]
    using_dynamic_alpha = True
    data_filename = "increment-3-bit.csv" if len(sys.argv) < 2 else sys.argv[1]

    

    header, data = read_data(data_filename, ",")
    pairs = convert_data_to_pairs(data, header)


    input_len = len(pairs[0][0])
    output_len = len(pairs[0][1])
    structure = [input_len] + hidden_structure + [output_len]

    if data_filename == "increment-3-bit.csv":
        training_data = pairs
        test_data = pairs
        nn = NeuralNetwork(structure)
        nn.print_weights()
        groups = k_groups(pairs, 10)

    else:
        accuracy = eval_k_groups(structure = structure, k = 10, pairs = pairs)
        print("\n\nStructure:{}\nOverall accuracy: {}".format(structure, accuracy))
        return




    # # Note: add 1.0 to the front of each x vector to account for the dummy input
    # training = [([1.0] + x, y) for (x, y) in pairs]

    # # Check out the data:
    # for example in training:
    #     print(example)
    

    
    

    nn.train(training_data, dynamic_alpha, max_epochs = epochs)
    if data_filename != "increment-3-bit.csv":
        accs = []
        for i in range(len(groups)):
            test_data = groups[i]
            training_data = [x for x in group for group in groups if group is not test_data]
            accs.append("accuracy", accuracy(nn, test_data))
        avg_accuracy = sum(accs)/len(accs)
        print("avg_accuracy:{}".format(avg_accuracy))
    else:

        for x, y in pairs:
            output = nn.eval(x)
            ftd = [str(p)[:5] for p in output]
            print('input:{} output:{} desired:{}'.format(x, ftd, y))
    nn.print_weights()

    with open("outfile.txt", "a") as outfile:
        alpha_string = "{}/{}+epoch_num".format(DYNAMIC_ALPHA_CONSTANT,DYNAMIC_ALPHA_CONSTANT) if using_dynamic_alpha else str(STATIC_ALPHA)
        outfile.write("Epochs:{}structure:{}alpha:{}".format(epochs, structure, alpha_string))

    ### I expect the running of your program will work something like this;
    ### this is not mandatory and you could have something else below entirely.
    # nn = NeuralNetwork([3, 6, 3])
    # nn.back_propagation_learning(training)

if __name__ == "__main__":
    main()
