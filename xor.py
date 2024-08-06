from utils.layers import *
from utils.network import Network
from utils.optimizers import Adam
from utils.functions import Activations, Loss
import matplotlib.pyplot as plt
import numpy as np, pickle, time

if __name__ == "__main__":
    model = [
        Input((2, 7, 7)),
        Conv2d(2, (3, 3)),
        Activation("selu"),
        Conv2d(2, (3, 3)),
        Activation("selu"),
        Conv2d(2, (3, 3)),
        Activation("softmax"),
        # BatchNorm(momentum = 0.9),
        # Conv2d(2, (3, 3)),
        # Activation("softmax"),
        Flatten(),
        # Dense(16),
        # Activation("selu"),
        # Dense(4),
        # Activation("selu"),
        # Dense(2),
        # Activation("softmax"),
    ]

    print(model)

    network = Network(model, loss_function="cross_entropy", optimizer=Adam(momentum = 0.9, beta_constant = 0.99, learning_rate=0.01))
    network.compile()
    
    training_percent = 1
    batch_size = 4

    save_file = 'model-training-data.json'

    xdata = [[i % 2, i // 2] for i in range(4)]
    ydata = [[1 - (i % 2) ^ (i // 2), (i % 2) ^ (i // 2)] for i in range(4)]

    costs = []
    val_costs = []
    plt.ion()

    start_time = time.perf_counter()

    for idx, cost in enumerate(network.fit(xdata, ydata, batch_size = batch_size, epochs = 400, threads=4)):
        if idx % 10:
            save_data = network.save()

            # network = Network()
            # network.load(save_data)

        end_time = time.perf_counter()

        print(end_time - start_time, "time")

        choice = np.random.randint(len(xdata))

        model_output = network.forward(xdata[choice], training=False)
        val_loss = network.loss_function(model_output, ydata[choice])

        val_costs.append(val_loss)
        costs.append(cost)

        print(cost)

        plt.plot(np.arange(len(costs)) * (batch_size / (len(xdata) * training_percent)), costs, label='training')
        plt.plot(np.arange(len(val_costs)) * (batch_size / (len(xdata) * training_percent)), val_costs, color='orange', label='validation')

        plt.legend()
        plt.draw()
        plt.pause(0.1)
        plt.clf()

        start_time = time.perf_counter()