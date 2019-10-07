import numpy as np
import pickle


class NN:
    def __init__(self, layer_sizes, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.weights = [np.zeros((m, n)) for m, n in zip(layer_sizes[1:], layer_sizes)]
        self.layer_sizes = layer_sizes
        self.seed = seed
    
    def predict(self, X):
        out = X
        for W in self.weights:
            Z = out @ W.T
            out = np.tanh(Z)
        if out.shape[0] == 1 and len(out.shape) == 1:
            return out.item()
        return out

    def set_weights(self, weights, copy=False):
        if copy:
            self.weights = [np.copy(l) for l in weights]
        else:
            self.weights = weights
        
    def get_weights(self, copy=False):
        if copy:
            return [np.copy(l) for l in self.weights]
        return self.weights


class NNBackProp:
    def __init__(self):
        np.random.seed(12334)

        self.input_size = 24
        self.learning_rate = 0.000001

        self.layers_size = [self.input_size, 32, 4]
        self.nn = []
        self.bias = []

        self.activations = []
        self.derivatives = []

    def tanh(self, X):
        return (np.exp(X) - np.exp(X * -1)) / (np.exp(X) + np.exp(X * -1))

    def tanh_derivative(self, X):
        return 1 - (self.tanh(X) ** 2)

    def relu(self, X):
        return np.maximum(X, 0)

    def relu_derivative(self, X):
        X[X <= 0] = 0
        X[X > 0] = 1

        return X

    def init_nn(self):
        self.activations = [self.relu, self.tanh]
        self.derivatives = [self.relu_derivative, self.tanh_derivative]

        for i in range(len(self.layers_size)-1):
            self.nn.append(np.random.randn(self.layers_size[i], self.layers_size[i+1]))
            self.bias.append(np.random.randn(self.layers_size[i+1], 1))

    def get_nn_prop_history(self, X):
        prev_dim = 1 if len(X.shape) < 2 else X.shape[1]
        prev = X.reshape( (X.shape[0], prev_dim) )
        history = []

        for i in range(len(self.nn)):
            curr_layer = np.transpose(self.nn[i])
            curr_bias = self.bias[i]

            current_state = self.activations[i](curr_layer.dot(prev) + curr_bias)
            prev = current_state
            history.append(current_state)

        return history

    def nn_backprop_one(self, observation, cum_reward):
        observation = np.array(observation).reshape((24,))
        history = self.get_nn_prop_history(observation)

        losses = np.transpose(np.array([cum_reward] * 4))
        o_0 = losses * self.derivatives[-1](history[-2].T.dot(self.nn[-1]) + self.bias[-1].reshape((self.bias[-1].shape[0],)))
        o_0 = o_0[0]

        o_1 = []

        for i in range(self.layers_size[-2]):
            o_1_tmp = 0
            for j in range(self.layers_size[-1]):
                delta = self.learning_rate * o_0[j] * history[-2][i]
                o_1_tmp += self.nn[-1][i][j] * o_0[j]
                self.nn[-1][i][j] += delta

            o_1.append(o_1_tmp)

        o_1 = np.array(o_1)
        o_1 = o_1 * self.derivatives[-1](observation.T.dot(self.nn[-2]) + self.bias[-2].reshape((self.bias[-2].shape[0],)))

        for i in range(self.layers_size[-3]):
            for j in range(self.layers_size[-2]):
                delta = self.learning_rate * o_1[j] * observation[i]
                self.nn[-2][i][j] += delta

    def nn_backprop(self, observations, rewards):
        cum_rewards = []
        cum_rew = 0

        for reward in rewards:
            cum_rew = reward - cum_rew
            cum_rewards.append(cum_rew)

        for i in range(len(observations)):
            self.nn_backprop_one(observations[i], cum_rewards[i])

    def predict(self, X):
        prev_dim = 1 if len(X.shape) < 2 else X.shape[1]
        prev = X.reshape( (X.shape[0], prev_dim) )

        for i in range(len(self.nn)):
            curr_layer = np.transpose(self.nn[i])
            curr_bias = self.bias[i]

            current_state = self.activations[i](curr_layer.dot(prev) + curr_bias)
            prev = current_state

        return prev

    def save_weights(self, file):
        model = [self.nn, self.bias]
        pickle.dump(model, open(file, 'wb'))

    def load_weights(self, file):
        try:
            model = pickle.load(open(file, 'rb'))
        except FileNotFoundError:
            print("File not found:", file)
            exit(1)
        self.nn = model[0]
        self.bias = model[1]


