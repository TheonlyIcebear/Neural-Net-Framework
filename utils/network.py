import multiprocessing, threading, numpy as np, awkward as ak, utils.layers
from tqdm.auto import tqdm, trange
from utils.optimizers import *
from utils.functions import Loss
from multiprocessing import Manager, Pool

class Network:
    def __init__(self, model=[], loss_function=Loss.mse, optimizer=SGD(), scheduler=None):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.optimizer_values = [None] * len(self.model)

    def compile(self):
        input_shape = self.model[0].output_shape.copy()
        print(input_shape)
        for layer in self.model[1:]:
            layer.initialize(input_shape)
            input_shape = layer.output_shape
            print(input_shape)

    def save(self):
        model_data = []
        for layer in self.model:
            model_data.append(list(layer.save()) + [layer.__class__.__name__])

        return [
            self.optimizer_values,
            model_data
        ]

    def load(self, save_data):
        input_shape = None

        optimizer_values, model_data = save_data
        self.optimizer_values = optimizer_values

        model = []

        for layer_args, layer_data, layer_type in model_data:
            layer_class = getattr(utils.layers, layer_type)
            layer = layer_class(*layer_args)

            layer.load(layer_data)

            model.append(layer)

        self.model = model

    def forward(self, activations, training=True):
        activations = np.array(activations)

        for layer in self.model:
            activations = layer.forward(activations, training=training)

        return activations

    def backward(self, output, expected_output):
        expected_output = np.array(expected_output)
        cost = self.loss_function(output, expected_output).mean()
        node_values = self.loss_function(output, expected_output, deriv=True)

        gradients = [None] * (len(self.model) - 1)

        for idx, layer in enumerate(self.model[::-1][:-1]):
            current_layer_index = -(idx + 1)

            input_activations = self.model[current_layer_index - 1].output_activations
            node_values, gradient = layer.backward(input_activations, node_values)
            gradients[current_layer_index] = gradient

            del gradient, input_activations

        for layer in self.model[::-1][:-1]:
            del layer.output_activations

        return gradients, cost

    def _worker(self, thread_index, index):
        extra = (1 * (thread_index < (self.batch_size % self.threads))) if self.batch_size % self.threads else 0
        tests = (self.batch_size // self.threads) + extra

        return_list = []
        threads = []

        for input_data, expected_output in zip(self.xdata[index-tests:index], self.ydata[index-tests:index]):
            model_output = self.forward(input_data)

            gradient, cost = self.backward(model_output, expected_output)

            return_list.append(cost)
            return_list.append(gradient)

            del gradient, model_output

        return return_list

    def average_gradients(self, gradients):
        summed_array = gradients[0]

        for gradient in gradients[1:]:
            for idx, layer in enumerate(gradient):
                if isinstance(layer, np.ndarray):
                    summed_array[idx] += layer
                else:
                    for count, data in enumerate(layer):
                        summed_array[idx][count] += data

        for idx, layer in enumerate(summed_array):
            if isinstance(layer, np.ndarray):
                summed_array[idx] /= self.batch_size
            else:
                for count, data in enumerate(layer):
                    summed_array[idx][count] /= self.batch_size

        return summed_array

    def fit(self, xdata, ydata, batch_size, learning_rate, epochs, threads=1):
        self.batch_size = batch_size
        self.threads = threads

        xdata = np.array(xdata)
        ydata = np.array(ydata)

        iterations = int(epochs * (len(xdata) / batch_size))

        pool = Pool(processes=self.threads)

        indices = []
        index = 0

        for thread_index in range(threads):
            extra = (1 * (thread_index < (self.batch_size % self.threads))) if self.batch_size % self.threads else 0
            tests = (self.batch_size // self.threads) + extra

            index += tests

            indices.append(index)

        for iteration in range(iterations):

            epoch = (batch_size / len(xdata)) * iteration
            
            if self.scheduler:
                learning_rate = self.scheduler.forward(learning_rate, epoch)

            choices = np.random.choice(xdata.shape[0], size=batch_size, replace=False)
            
            self.xdata = xdata[choices]
            self.ydata = ydata[choices]

            return_data = sum(pool.starmap(self._worker, zip(range(threads), indices)), [])
            
            gradients = return_data[1::2]
            costs = np.array(return_data[::2])

            del return_data

            cost = np.mean(costs)

            yield cost

            gradient = self.average_gradients(gradients)
            del gradients
            for idx, (layer, layer_gradient, descent_values) in enumerate(zip(self.model[1:], gradient, self.optimizer_values)):
                new_descent_values = layer.update(self.optimizer, layer_gradient, descent_values, learning_rate)

                self.optimizer_values[idx] = new_descent_values