import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class PerceptronContinuous:
    def __init__(self):
        self._weights = np.random.normal(size=2)
        self._bias = np.zeros(1)

    def forward(self, x):
        y = np.sum(self._weights * x) + self._bias
        return self._activation(y)

    def _activation(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    def backward(self, alpha, d, y, x):
        self._weights += alpha * (d - y) * y * (1 - y) * x
        self._bias += alpha * (d - y) * y * (1 - y)


def plot_loss(loss):
    l = np.stack(loss)
    t = range(l.shape[0])
    fig, ax = plt.subplots()
    ax.plot(t, l)

    ax.set(xlabel='epoch', ylabel='loss', title='Perceptron training')
    ax.grid()
    plt.show()


if __name__ == '__main__':
    and_data = [
        (np.array([0, 0]), np.array([0])),
        (np.array([0, 1]), np.array([0])),
        (np.array([1, 0]), np.array([0])),
        (np.array([1, 1]), np.array([1])),
    ]

    xor_data = [
        (np.array([0, 0]), np.array([0])),
        (np.array([0, 1]), np.array([1])),
        (np.array([1, 0]), np.array([1])),
        (np.array([1, 1]), np.array([0])),
    ]

    data = xor_data

    perceptron = PerceptronContinuous()
    alpha = .5
    epochs = 500
    loss_list = []

    # Training phase
    for epoch in range(epochs):
        loss = np.zeros(1)
        for x, d in data:
            y = perceptron.forward(x)
            loss += np.power(d - y, 2) / 2
            perceptron.backward(alpha, d, y, x)
        loss_list.append(loss)
        print("Epoch {0:d} loss {1:f}".format(epoch, loss.item()))

    plot_loss(loss_list)

    # Testing phase
    for x, d in data:
        y = perceptron.forward(x)
        print(x, y)
