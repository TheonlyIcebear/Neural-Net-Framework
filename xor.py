from utils.layers import *
from utils.schedulers import *
from utils.network import Network
from utils.optimizers import Adam
from utils.functions import Activations, Loss
import matplotlib.pyplot as plt
import numpy as np, pickle, time

if __name__ == "__main__":
    model = [
        Input(2),
        Dense(3),
        Activation("lrelu"),
        Dense(2),
        Activation("softmax"),
    ]

    print(model)

    network = Network(model, loss_function="cross_entropy", optimizer=Adam(momentum = 0.9, beta_constant = 0.99))
    network.compile()
    
    training_percent = 1
    batch_size = 4

    save_file = 'model-training-data.json'

    xdata = [[i % 2, i // 2] for i in range(4)]
    ydata = [[(i % 2) ^ (i // 2), 1 - ((i % 2) ^ (i // 2))] for i in range(4)]

    costs = []
    plt.ion()

    start_time = time.perf_counter()

    for idx, cost in enumerate(network.fit(xdata, ydata, learning_rate=0.01, batch_size = batch_size, epochs = 1000, threads=4)):
        if idx % 10:
            save_data = network.save()

            # network = Network()
            # network.load(save_data)

        end_time = time.perf_counter()

        print(end_time - start_time, "time")

        costs.append(cost)

        print(cost)

        plt.plot(np.arange(len(costs)) * (batch_size / (len(xdata) * training_percent)), costs, label='training')

        plt.legend()
        plt.draw()
        plt.pause(0.1)
        plt.clf()

        start_time = time.perf_counter()
