import numpy as np

class NNet():
    def __init__(self, hidden_shape, learning_rate=0.002, l1_reg=0, l2_reg=0,
                 dropout=0, epsilon=0.0001):
        self.hidden_shape = hidden_shape
        self.alpha = learning_rate
        self.weights = []
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.dropout = dropout
        self.epsilon = epsilon
        self.prev_l2 = [0] * (len(hidden_shape)+1)

    def sigmoid(self, X):
        return 1/(1+np.exp(-1*X))

    def dsigmoid(self, X):
        return self.sigmoid(X) * (1-self.sigmoid(X))

    def softmax(self, X):
        out = X
        for i,x in enumerate(X):
            out[i] = np.exp(x)/np.sum(np.exp(X))
        return out

    def init_weights(self, insize, outsize):
        weights = []
        weights.append(np.random.randn(insize, self.hidden_shape[0]) * 0.01)
        for i,size in enumerate(self.hidden_shape[:-1]):
            weights.append(np.random.randn(size, self.hidden_shape[i+1]) * 0.01)
        weights.append(np.random.randn(self.hidden_shape[-1], outsize) * 0.01)
        self.weights = weights

    def forward(self, X, y, weights):
        a = [X]
        z = [X]
        for i,wl in enumerate(weights[:-1]):
            a.append(np.dot(z[-1], wl))
            z.append(self.sigmoid(a[-1]))
        a.append(np.dot(z[-1], weights[-1]))
        z.append(self.softmax(a[-1]))
        cost = -np.sum(y * np.log(z[-1]))
        return z, cost

    def lossfunc(self, X, y):
        # dropout mask
        if self.dropout != 0:
            drop = []
            drop.append((np.random.randn(*self.weights[0].shape) < self.dropout) / self.dropout)
            for i,size in enumerate(self.hidden_shape[:-1]):
                drop.append((np.random.randn(*self.weights[i].shape) < self.dropout) / self.dropout)
            drop.append((np.random.randn(*self.weights[-1].shape) < self.dropout) / self.dropout)

            weights = []
            for w,d in zip(self.weights, drop):
                weights.append(w * d)
        else:
            weights = self.weights

        # forward pass
        a = [X]
        z = [X]
        for i,wl in enumerate(weights[:-1]):
            a.append(np.dot(z[-1], wl))
            z.append(self.sigmoid(a[-1]))
        a.append(np.dot(z[-1], weights[-1]))
        z.append(self.softmax(a[-1]))

        sweights = 0
        sweights_squared = 0
        for w in weights:
            sweights += np.sum(w)
            sweights_squared += np.sum(np.square(w))

        cost = -np.sum(y * np.log(z[-1])) + self.l2_reg*sweights + 0.5*self.l2_reg*sweights_squared

        # backprop
        d = []
        dw = []
        d.append((z[-1]-y) * self.dsigmoid(a[-1]))
        dw.append(np.dot(z[-2].T, d[-1]))
        j = len(weights)
        for i in range(1, j):
            k = len(weights) - i
            d.append(np.dot(d[-1], weights[k].T) * self.dsigmoid(a[k]))
            dw.append(np.dot(z[k-1].T, d[-1]))
        dwa = []
        for i,g in enumerate(reversed(dw)):
            g += self.l1_reg*weights[i] + self.l2_reg*weights[i]
            dwa.append(g)
        for i,dwd in enumerate(dwa):
            dwa[i] = dwd * drop[i]
        return cost, dwa

    def gradient_check(self, X, y, dw):
        grad = []
        for w in self.weights:
            grad.append(np.zeros_like(w))

        for i in range(len(grad)):
            for x in range(grad[i].shape[0]):
                for y in range(grad[i].shape[1]):
                    weights_i = self.weights[i]
                    e = np.zeros(weights_i.shape)
                    e[x][y] = 1

                    weights_p = self.weights
                    weights_p[i] = weights_i + self.epsilon*e
                    weights_m = self.weights
                    weights_m[i] = weights_i - self.epsilon*e
                    _, g_p = self.forward(X, y, weights_p)
                    _, g_m = self.forward(X, y, weights_m)
                    grad[i][x][y] = (g_p - g_m)/(2*self.epsilon)
        error = float(0)
        for g,dwa in zip(grad, dw):
            error += np.sum(np.absolute(np.subtract(g, dwa))) / \
                np.sum(np.absolute(np.subtract(g, dwa)))
        return error

    def sgd(self, grad, dimension):
        return -self.alpha * grad[dimension]

    def adagrad(self, grad, dimension):
        prev_l2 = self.prev_l2[dimension]
        prev_l2[prev_l2 == 0] = 1
        return -self.alpha * grad[dimension] / prev_l2

    def predict(self, X):
        z = [X]
        for i,wl in enumerate(self.weights[:-1]):
            z.append(self.sigmoid(np.dot(z[i], wl)))
        z.append(self.softmax(np.dot(z[-1], self.weights[-1])))
        return np.argmax(z[-1])

    def train(self, X, y, epochs=100):
        if self.weights == []:
            insize = len(X[0])
            outsize = len(y[0])
            self.init_weights(insize, outsize)
        for e in xrange(epochs):
            cost = 0
            for Xa, ya in zip(X, y):
                inputs = np.array([Xa])
                labels = np.array([ya])
                c, dw = self.lossfunc(inputs, labels)
                for i in range(len(self.prev_l2)):
                    self.prev_l2[i] = np.sqrt(np.square(self.prev_l2[i]) +
                                              np.square(dw[i]))
                cost = c
                for i in range(len(self.weights)):
                    delta_w = self.adagrad(dw, i)
                    self.weights[i] = self.weights[i] + delta_w
            print 'Epoch ' + str(e) + ', cost: ' + str(cost)

