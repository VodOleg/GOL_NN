import random
import numpy as np
import pickle
from Utilities import sigmoid, sigmoid_prime, binary_step, binary_step_prime

class Network(object):

    def __init__(self, sizes, config):
        self.num_layers = len(sizes)
        self.config = config
        self.sizes = sizes
        self.epochs = config["epochs"]
        self.mini_batch_size = config["batch_size"]
        self.training_rate = config["training_rate"]
        self.success_threshold = config["success_threshold"]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def classify(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def train(self, training_data, test_data=None):
        epochs = self.epochs
        mini_batch_size = self.mini_batch_size
        eta = self.training_rate
        training_data = list(training_data)
        n = len(training_data)
        n_test = 0
        successfull_classifications = []
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        else:
            print("No Test Data...")

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                successfull_classification = self.evaluate(test_data)
                successfull_classifications.append(successfull_classification)
                print("Epoch {} : {} / {}".format(j,successfull_classification,n_test))
            else:
                print("Epoch {} complete".format(j))
        return successfull_classifications

    def dump_network(self, filename):
        network_data = {
            "config": self.config,
            "weights": self.weights,
            "biases": self.biases
        }
        with open(filename, "wb") as dumpFile:
            pickle.dump(network_data, dumpFile, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_network(self, filename):
        network_data = None
        with open(filename, 'rb') as handle:
            network_data = pickle.load(handle)
        self.weights = network_data["weights"]
        self.biases = network_data["biases"]

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(self.classify(x)[0][0], y[0][0]) for (x, y) in test_data]
        return sum(int(abs(y-x) < 1.0 - (self.success_threshold) ) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)


