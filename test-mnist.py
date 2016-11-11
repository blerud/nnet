import numpy as np

import mnist
import nnet

if __name__ == '__main__':
    trX,teX,trY,teY = mnist.mnist()
    net = nnet.NNet([50], l1_reg=0.0002, dropout=0.4)
    net.train(trX, trY, epochs=10)
    wrong = 0
    total = 0
    for X, y in zip(teX, teY):
        prediction = net.predict(np.array([X]))
        actual = np.argmax(y)
        if prediction != actual:
            wrong += 1
        total += 1
    print(wrong, total, float(total-wrong)/float(total))
