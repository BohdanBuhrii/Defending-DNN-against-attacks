

import numpy as np
import matplotlib.pyplot as plt


class NeuralNet:

    def __init__(self, layer_dims, normalize=True, learning_rate=0.01,
                 num_iter=30000, precision=None, mini_batch_size=None, T=1,
                 noise_layer=None, noise_layer_sensitivity=None, noise_type='Laplacian',
                 attack_size=None, eps=None, n_expected_scores=1):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.normalize = normalize
        self.layer_dims = layer_dims
        self.precision = precision
        self.mini_batch_size = mini_batch_size
        self.T = T
        
        # PixelDP parameters
        self.noise_layer = noise_layer  # index of noise layer
        self.noise_layer_sensitivity = noise_layer_sensitivity
        self.noise_type = noise_type  # Laplacian or Gaussian
        self.attack_size = attack_size # L_p size of noise expected from attack 
        self.eps = eps
        self.n_expected_scores = n_expected_scores
        
        self.itworks = None

    def apply_normalization(self, X, mean=None, std=None):
        n = X.shape[0]

        if mean is None:
            mean = np.mean(X, axis=1).reshape((n, 1))

        if std is None:
            std = np.std(X, axis=1).reshape((n, 1))

        X_new = (X - mean) / std**2

        return X_new, mean, std

    def denormalize(self, X):
      return X * self.__std**2 + self.__mean

    def __sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def __sigmoid_derivative(self, Z):
        s = self.__sigmoid(Z)
        return s*(1 - s)

    def __relu(self, Z):
        Z = np.array(Z, copy=True)
        Z[Z < 0] = 0
        return Z

    def __relu_derivative(self, Z):
        dZ = np.ones(Z.shape)
        dZ[Z < 0] = 0
        return dZ

    def __tanh(self, Z):
        return np.tanh(Z)

    def __tanh_derivative(self, Z):
        return 1 / np.power(np.cosh(Z), 2)

    def __softmax(self, Z):
        eZ = np.exp((Z - np.max(Z))/self.T)
        return eZ / np.sum(eZ, axis=0, keepdims=True)

    def __initialize_parameters(self):
        layer_dims = self.layer_dims
        parameters = {}
        L = len(layer_dims)

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l],
                                                       layer_dims[l-1]) * 0.01
            print(parameters['W' + str(l)].shape)
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        self.parameters = parameters

    def __forward_linear_activation(self, A_prev, W, b, activation):

        # linear forward
        Z = np.dot(W, A_prev) + b
        linear_cache = (A_prev, W, b)

        # activation forward
        if activation == 'sigmoid':
            A = self.__sigmoid(Z)

        if activation == 'relu':
            A = self.__relu(Z)

        if activation == 'softmax':
            A = self.__softmax(Z)

        if activation == 'tanh':
            A = self.__tanh(Z)

        activation_cache = Z

        cache = (linear_cache, activation_cache)

        return A, cache

    def multilayer_forward(self, X):
        parameters = self.parameters
        caches = []
        A = X
        L = len(parameters) // 2

        for l in range(1, L):
            if(self.noise_layer == l):
                #if(self.itworks == None):
                #print('it works', l)
                #  self.itworks = 1

                A = self.add_noise(A)
            
            A_prev = A
            
            A, cache = self.__forward_linear_activation(
                A_prev, parameters["W"+str(l)], parameters["b"+str(l)], activation='tanh')
            caches.append(cache)

        AL, cache = self.__forward_linear_activation(
            A, parameters["W"+str(L)], parameters["b"+str(L)], activation='softmax')
        caches.append(cache)

        #assert(AL.shape == (10, X.shape[1]))

        return AL, caches

    def add_noise(self, A):
        # Guassian
        #omega = np.sqrt(2*np.log(1.25/self.delta)) * self.noise_layer_sensitivity * self.L / self.eps
        
        # Laplacian
        omega = np.sqrt(2) * self.noise_layer_sensitivity * self.attack_size / self.eps
        
        #print(omega)
        
        B = np.copy(A)
        
        return B + np.random.laplace(scale = omega, size = A.shape) 

    def __backward_linear_activation(self, dA, cache, activation):

        linear_cache, activation_cache = cache

        # activation backward
        Z = activation_cache
        A_prev, W, b = linear_cache

        if activation == 'sigmoid':
            dZ = dA * self.__sigmoid_derivative(Z)

        if activation == 'relu':
            dZ = dA * self.__relu_derivative(Z)

        if activation == 'tanh':
            dZ = dA * self.__tanh_derivative(Z)

        # linear backward
        A_prev, W, b = linear_cache
        m = A_prev.shape[1]
        dW = 1 / m * np.dot(dZ, A_prev.T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def __multilayer_backward(self, X, Y, caches):
        grads = {}
        AL = X
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        linear_cache, activation_cache = caches[L-1]
        A_prev, W, b = linear_cache

        dZ = (AL - Y)# * self.T**2
        m = AL.shape[1]
        grads["dW" + str(L)] = 1 / m * np.dot(dZ, A_prev.T)
        grads["db" + str(L)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        grads["dA" + str(L-1)] = np.dot(W.T, dZ)

        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = \
                self.__backward_linear_activation(
                    grads["dA" + str(l + 1)], current_cache, activation='tanh')
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def compute_cost(self, A, Y):
        J = -np.mean(Y.T * np.log(A.T + 1e-8))
        return J

    def cross_entropy(self, A, Y):
        return - np.sum(Y * np.log(A), axis=1)

    def __update_parameters(self, grads):
        parameters = self.parameters
        learning_rate = self.learning_rate
        L = len(parameters) // 2

        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - \
                learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - \
                learning_rate * grads["db" + str(l+1)]

        self.parameters = parameters

    def __update_parameters_with_momentum(self, grads):
        parameters = self.parameters
        learning_rate = self.learning_rate
        L = len(parameters) // 2

        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - \
                learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - \
                learning_rate * grads["db" + str(l+1)]

        self.parameters = parameters

    def __random_mini_batches(self, X, Y):
        m = X.shape[1]
        mini_batches = []
        mini_batch_size = self.mini_batch_size if self.mini_batch_size != None else m

        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]

        shuffled_Y = Y[:, permutation]

        num_complete_minibatches = int(np.floor(m/mini_batch_size))
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k *
                                      mini_batch_size: (k + 1)*mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k *
                                      mini_batch_size: (k + 1)*mini_batch_size]

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, k*mini_batch_size:]
            mini_batch_Y = shuffled_Y[:, k*mini_batch_size:]

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def __gradient_descent(self, X, Y, print_cost=False):
        m = X.shape[1]
        costs = []

        for i in range(0, self.num_iter):
            mini_batches = self.__random_mini_batches(X, Y)
            #iteration_cost = 0

            for (mini_X, mini_Y) in mini_batches:

                AL, caches = self.multilayer_forward(mini_X)

                #iteration_cost += self.compute_cost(AL, mini_Y)
                cost = self.compute_cost(AL, mini_Y)

                grads = self.__multilayer_backward(AL, mini_Y, caches)

                self.__update_parameters(grads)

            #cost = iteration_cost/m
            costs.append(cost)
            if print_cost and i % 10 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

            if len(costs) > 1 and self.precision != None \
                    and np.abs(costs[-2] - costs[-1]) < self.precision:
                print('Stopping gradient descent ...')
                break

        if print_cost:
            plt.title(str(self.num_iter) + " iterations")
            plt.plot(costs)
            plt.ylabel("Cost")
            plt.xlabel("Iteration")
            plt.show()

    def fit(self, X_vert, Y_vert, print_cost=True):

        X, Y = X_vert.T.copy(), Y_vert.T.copy()

        if self.normalize:
            X, self.__mean, self.__std = self.apply_normalization(X)

        self.__initialize_parameters()

        self.__gradient_descent(X, Y, print_cost)

    def keep_fitting(self, X_vert, Y_vert, iterations, print_cost=True):

        X, Y = X_vert.T, Y_vert.T

        prev_iter = self.num_iter
        self.num_iter = iterations

        self.__gradient_descent(X, Y, print_cost)

        self.num_iter = prev_iter + self.num_iter

    def predict_proba(self, X_vert):
        # n_expected_scores is for evaluation under PixelDP defence
        
        X = X_vert.T
        if self.normalize:
            X, _, _ = self.apply_normalization(X, self.__mean, self.__std)

        probs = np.zeros((10, X_vert.shape[0]))  # TODO shape
        
        for _ in range(self.n_expected_scores):
            probs += self.multilayer_forward(X)[0]

        return probs.T / self.n_expected_scores

    def predict(self, X_vert):
        positive_probs = self.predict_proba(X_vert)
        return np.argmax(positive_probs, axis=1)
