import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2


class NeuralNetwork:

    def __init__(self, layers, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime

        self.weights = []


        # print len(layers)
        print len(layers)
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            print r,i
            self.weights.append(r)
        # output layer - random((2+1, 1)) : 3 x 1
        print i
        r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)
        print self.weights

        # print self.weights

    def fit(self, X, y, learning_rate=0.2, epochs=10000):

        ones = np.atleast_2d(np.ones(X.shape[0]))


        X = np.concatenate((ones.T, X), axis=1)


         
        error=0
        for k in range(epochs):
            if k % 100== 0: print 'epochs:', k, error
            
            i = np.random.randint(X.shape[0])

            a = [X[i]]
            for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    activation = self.activation(dot_value)
                    a.append(activation)
            
            error = y[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]

            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))
               
            deltas.reverse()

            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)


if __name__ == '__main__':

    nn = NeuralNetwork([2,2,1])

    X = np.array([[0.1, 0.1],
                  [0.1, 0.9],
                  [0.9, 0.1],
                  [0.9, 0.9]])

    y = np.array([0.1, 0.9, 0.9, 0.1])

    nn.fit(X, y)