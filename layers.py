import scipy, skimage, numpy as np, awkward as ak, time
from utils.functions import Activations

class Layer:
    def __init__(self):
        pass

    def initialize(self, input_shape):
        self.output_shape = input_shape

    def forward(self, input_activations, training=True):
        self.output_activations = input_activations
        return self.output_activations

    def backward(self, input_activations, node_values):
        return node_values, []

    def update(self, optimizer, gradient, descent_values, learning_rate):
        pass

    def save(self):
        return [], None

    def load(self, data):
        pass


class Conv2d(Layer):
    def __init__(self, depth, kernel_shape=[3, 3], stride=1, variance="He"):
        self.kernel_shape = np.array(kernel_shape)
        self.variance = variance
        self.depth = depth
        self.stride = stride

    def forward(self, input_activations, training=True):
        output_activations = self.biases.copy()
        for i, kernels in enumerate(self.kernels):
            for kernel, channel in zip(kernels, input_activations):
                output_activations[i] += scipy.signal.correlate2d(channel, kernel, "valid")
        
        if training:
            self.output_activations = output_activations

        return output_activations

    def backward(self, input_activations, node_values):
        new_node_values = np.zeros(input_activations.shape)
        kernels_gradient = np.zeros(self.kernels.shape)

        for i, (kernels, kernel_node_values) in enumerate(zip(self.kernels, node_values)):
            for j, (image, kernel) in enumerate(zip(input_activations, kernels)):

                kernels_gradient[i, j] = scipy.signal.correlate2d(image, kernel_node_values, "valid")
                new_node_values[j] += scipy.signal.convolve2d(kernel_node_values, kernel, "full")

        end_time = time.perf_counter()

        kernels_biases_gradient = node_values
        return new_node_values, [kernels_gradient, kernels_biases_gradient]

    def initialize(self, input_shape):
        input_channels = input_shape[0]
        output_channels = self.depth

        kernel_width, kernel_height = self.kernel_shape
        fan_in = input_channels * kernel_width * kernel_height
        fan_out = output_channels * kernel_width * kernel_height

        output_shape = input_shape
        output_shape[0] = self.depth
        output_shape[1:] = (input_shape[1:] - self.kernel_shape + 1) // self.stride

        self.output_shape = output_shape

        if (not self.variance) or self.variance == "He":
            variance = np.sqrt(2 / (fan_in))
            self.kernels = np.random.normal(0, variance, (output_channels, input_channels, kernel_width, kernel_height))

        elif self.variance == "lecun":
            variance = np.sqrt(1 / (fan_in))
            self.kernels = np.random.normal(0, variance, (output_channels, input_channels, kernel_width, kernel_height))

        elif self.variance == "xavier":
            variance = np.sqrt(6 / (fan_in + fan_out))
            self.kernels = np.random.uniform(-variance, variance, (output_channels, input_channels, kernel_width, kernel_height))

        else:
            variance = self.variance
            self.kernels = np.random.uniform(-variance, variance, (output_channels, input_channels, kernel_width, kernel_height))

        self.biases = np.zeros(self.output_shape)

    def update(self, optimizer, gradient, descent_values, learning_rate):
        if not descent_values is None:
            kernel_descent_values, bias_descent_values = descent_values

        else:
            kernel_descent_values = None
            bias_descent_values = None

        kernels_gradient, kernels_biases_gradient = gradient

        self.kernels, new_kernel_descent_values = optimizer.apply_gradient(self.kernels, kernels_gradient, kernel_descent_values, learning_rate)
        self.biases, new_bias_descent_values = optimizer.apply_gradient(self.biases, kernels_biases_gradient, bias_descent_values, learning_rate)

        return [new_kernel_descent_values, new_bias_descent_values]

    def save(self):
        return [self.depth, self.kernel_shape, self.stride, self.variance], [self.kernels, self.biases]

    def load(self, data):
        self.kernels, self.biases = data
        self.kernels = np.array(self.kernels)
        self.biases = np.array(self.biases)

class Dense(Layer):
    def __init__(self, depth, variance="He"):
        self.variance = variance
        self.depth = depth

    def forward(self, input_activations, training=True):

        weights = self.layer[:, :-1]
        bias = self.layer[:, -1]

        output = np.dot(weights, input_activations) + bias

        output_activations = output

        if training:
            self.output_activations = output_activations

        return output_activations

    def backward(self, input_activations, old_node_values):
        weights = self.layer[:, :-1]
        biases = self.layer[:, -1]

        gradient = np.zeros(self.layer.shape)
        new_node_values = np.dot(weights.T, old_node_values)

        weights_derivative = old_node_values[:, None] * input_activations
        bias_derivative = 1 * old_node_values

        gradient[:, :-1] += weights_derivative
        gradient[:, -1] += bias_derivative

        return new_node_values, gradient

    def initialize(self, inputs):
        if (not self.variance) or self.variance == "He":
            variance = np.sqrt(2 / (inputs))
            self.layer = np.random.normal(0, variance, (self.depth, inputs + 1))

        elif self.variance == "lecun":
            variance = np.sqrt(1 / (inputs))
            self.layer = np.random.normal(0, variance, (self.depth, inputs + 1))

        elif self.variance == "xavier":
            variance = np.sqrt(6 / (inputs + self.depth))
            self.layer = np.random.uniform(-variance, variance, (self.depth, inputs + 1))

        else:
            variance = self.variance
            self.layer = np.random.uniform(-variance, variance, (self.depth, inputs + 1))

        self.layer[:, -1] = -0
        self.output_shape = self.depth

    def update(self, optimizer, gradient, descent_values, learning_rate):
        self.layer, new_descent_values = optimizer.apply_gradient(self.layer, gradient, descent_values, learning_rate)

        return new_descent_values

    def save(self):
        return [self.depth, self.variance], self.layer

    def load(self, data):
        self.layer = np.array(data)

class BatchNorm(Layer):
    def __init__(self, momentum=0.9, batch_size=4):
        self.momentum = momentum
        self.batch_size = batch_size

    def forward(self, x, training=True):
        epsilon = 1e-5

        # if training:
        if x.ndim == 3: 
            batch_mean = np.mean(x, axis=(1, 2), keepdims=True)
            batch_var = np.var(x, axis=(1, 2), keepdims=True)
        elif x.ndim == 1:
            batch_mean = np.mean(x, axis=0, keepdims=True)
            batch_var = np.var(x, axis=0, keepdims=True)

        self.x_centered = x - batch_mean
        self.stddev_inv = 1 / np.sqrt(batch_var + epsilon)
        x_norm = self.x_centered * self.stddev_inv
        output_activations = self.gamma * x_norm + self.beta

        # else:
        #     x_norm = (x - self.running_mean) / np.sqrt(self.running_var + epsilon)
        #     output_activations = self.gamma * x_norm + self.beta

        if training:
            self.output_activations = output_activations

        return output_activations

    def backward(self, input_activations, old_node_values):
        if input_activations.ndim == 3:  # Convolutional layer (channels, height, width)
            C, H, W = input_activations.shape

            x_norm = self.x_centered * self.stddev_inv
            beta_gradient = np.sum(old_node_values, axis=(1, 2), keepdims=True)
            gamma_gradient = np.sum(old_node_values * x_norm, axis=(1, 2), keepdims=True)

            norm_gradient = old_node_values * self.gamma
            dvar = np.sum(norm_gradient * self.x_centered, axis=(1, 2), keepdims=True) * -0.5 * self.stddev_inv**3
            dmean = np.sum(norm_gradient * -self.stddev_inv, axis=(1, 2), keepdims=True) + dvar * np.mean(-2. * self.x_centered, axis=(1, 2), keepdims=True)
            new_node_values = (norm_gradient * self.stddev_inv) + (dvar * 2 * self.x_centered / (self.batch_size * H * W)) + (dmean / (self.batch_size * H * W))

        elif input_activations.ndim == 1:
            features = input_activations.shape[0]

            x_norm = self.x_centered * self.stddev_inv
            beta_gradient = np.sum(old_node_values, axis=0)
            gamma_gradient = np.sum(old_node_values * x_norm, axis=0)

            norm_gradient = old_node_values * self.gamma
            dvar = np.sum(norm_gradient * self.x_centered, axis=0) * -0.5 * self.stddev_inv**3
            dmean = np.sum(norm_gradient * -self.stddev_inv, axis=0) + dvar * np.mean(-2. * self.x_centered, axis=0)
            new_node_values = (norm_gradient * self.stddev_inv) + (dvar * 2 * self.x_centered / self.batch_size) + (dmean / self.batch_size)

        return new_node_values, [gamma_gradient, beta_gradient, input_activations, input_activations**2]

    def update(self, optimizer, gradient, descent_values, learning_rate):
        dgamma, dbeta, batch_mean, batch_sq_mean = gradient

        batch_var = batch_sq_mean - batch_mean ** 2

        if descent_values is not None:
            gamma_descent_values, beta_descent_values = descent_values
        else:
            gamma_descent_values = None
            beta_descent_values = None

        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

        self.gamma, new_gamma_descent_values = optimizer.apply_gradient(self.gamma, dgamma, gamma_descent_values, learning_rate)
        self.beta, new_beta_descent_values = optimizer.apply_gradient(self.beta, dbeta, beta_descent_values, learning_rate)

        return [np.array(new_gamma_descent_values), np.array(new_beta_descent_values)]

    def initialize(self, input_shape):
        if isinstance(input_shape, tuple):
            features = input_shape[0]

        else:
            features = input_shape

        self.gamma = np.ones(features)
        self.beta = np.zeros(features)
        self.running_mean = np.zeros(features)
        self.running_var = np.ones(features)

        self.output_shape = input_shape

    def save(self):
        return [self.momentum, self.batch_size], [self.gamma, self.beta, self.running_mean, self.running_var]

    def load(self, data):
        self.gamma, self.beta, self.running_mean, self.running_var = data

class Activation(Layer):
    def __init__(self, activation_function):
        self.activation_function = getattr(Activations, activation_function)

    def forward(self, input_activations, training=True):
        output_activations = self.activation_function(input_activations)

        if training:
            self.output_activations = output_activations

        return output_activations

    def backward(self, input_activations, node_values):
        return node_values * self.activation_function(self.output_activations, deriv=True), []

    def save(self):
        return [self.activation_function.__name__], None

class Input(Layer):
    def __init__(self, input_shape):
        self.output_shape = np.array(input_shape)

    def forward(self, input_activations, training=True):
        output_activations = input_activations

        if training:
            self.output_activations = output_activations

        return output_activations

    def save(self):
        return [self.output_shape], None

class Flatten(Layer):
    def __init__(self):
        pass

    def forward(self, input_activations, training=True):
        output_activations = input_activations.flatten()

        if training:
            self.output_activations = output_activations

        self.input_shape = input_activations.shape
        return output_activations

    def backward(self, input_activations, node_values):
        return node_values.reshape(self.input_shape), []

    def initialize(self, input_shape):
        self.output_shape = input_shape.prod()

class Reshape(Layer):
    def __init__(self, output_shape):
        self.output_shape = output_shape

    def forward(self, input_activations, training=True):
        output_activations = input_activations.reshape(output_shape)

        if training:
            self.output_activations = input_activations.reshape(output_shape)

        self.input_shape = input_activations.shape
        return output_activations

    def backward(self, input_activations, node_values):
        return node_values.reshape(self.input_shape), []

    def save(self):
        return [self.output_shape], None

class MaxPool(Layer):
    def __init__(self, pooling_shape = [2, 2]):
        self.pooling_shape = np.array(pooling_shape)

    def forward(self, input_activations, training=True):
        result_dimensions = np.ceil(np.array(input_activations.shape[1:]) / self.pooling_shape).astype(int)
        result_width, result_height = result_dimensions

        depth = input_activations.shape[0]

        output_width, output_height = input_activations.shape[1:]
        pooling_width, pooling_height = self.pooling_shape
        padded_output = np.pad(input_activations, [(0, 0), (0, output_width % pooling_width), (0, output_height % pooling_height)])

        pooling_windows = skimage.util.view_as_blocks(padded_output, (1, pooling_width, pooling_height)).reshape(depth, -1, pooling_width, pooling_height)
        output_activations = np.max(pooling_windows, axis=(2, 3)).reshape(depth, *result_dimensions)

        pooling_indices = np.zeros((depth, pooling_windows.shape[1], 2)).astype(int)

        flat_windows = pooling_windows.reshape(depth, pooling_windows.shape[1], -1)
        flat_indices = np.argmax(flat_windows, axis=2)
        
        window_height, window_width = pooling_windows.shape[2], pooling_windows.shape[3]
        rows, cols = np.divmod(flat_indices, window_width)
        
        pooling_indices[:, :, 0] = rows
        pooling_indices[:, :, 1] = cols

        self.pooling_indices = pooling_indices

        del pooling_windows, pooling_indices, flat_windows, flat_indices, padded_output, rows, cols

        if training:
            self.output_activations = output_activations

        return output_activations

    def backward(self, input_activations, old_node_values):
        channels, height, width = input_activations.shape
        pooling_width, pooling_height = self.pooling_shape

        start_time = time.perf_counter()
        
        unpooled_array = np.zeros_like(input_activations)

        flattened_indices = self.pooling_indices.reshape(channels, -1, 2)
        x_indices = (flattened_indices[:, :, 0] % pooling_width + (flattened_indices[:, :, 0] // pooling_width) * pooling_width).flatten()
        y_indices = (flattened_indices[:, :, 1] % pooling_height + (flattened_indices[:, :, 1] // pooling_height) * pooling_height).flatten()

        unpooled_array = unpooled_array.flatten()
        np.add.at(unpooled_array, x_indices + width * y_indices, old_node_values.flatten())

        end_time = time.perf_counter()
        unpooled_array = unpooled_array.reshape(channels, height, width)

        return unpooled_array, []

    def initialize(self, input_shape):
        output_shape = input_shape
        output_shape[1:] = np.ceil(output_shape[1:] / self.pooling_shape).astype(int)
        self.output_shape = output_shape

    def save(self):
        return [self.pooling_shape], None

class Dropout(Layer):
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        
    def forward(self, input_activations, training=True):
        if training:
            self.mask = (np.random.rand(*input_activations.shape) > self.dropout_rate) / ( 1 - self.dropout_rate)
        else:
            self.mask = np.ones(input_activations.shape)

        output_activations = input_activations * self.mask

        if training:
            self.output_activations = output_activations
            
        return output_activations

    def backward(self, input_activations, node_values):
        return node_values * self.mask, []

    def initialize(self, input_shape):
        self.output_shape = input_shape

    def save(self):
        return [self.dropout_rate], None