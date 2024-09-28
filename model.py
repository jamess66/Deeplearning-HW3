import torch

from linear import Linear
from relu import ReLU


class ShallowNeuralNet:
    """
    Feedforward Neural Network
    """

    def __init__(self, layer_sizes):
        """
        Constructor
        :param layer_sizes: Tuple of shape (n_feature, n_hidden1, n_hidden2, ..., n_output)
        """

        self.dim_in = layer_sizes[0]
        self.dim_out = layer_sizes[-1]
        self.n_layers = len(layer_sizes) - 1   # number of layers in the neural network

        layers = []
        for i in range(self.n_layers):
            D_i = layer_sizes[i]
            D_o = layer_sizes[i+1]
            layers.append(Linear(D_i, D_o))
            if i != self.n_layers-1:
                layers.append(ReLU())
        self.layers = layers


    def __repr__(self):
        """
        Print the model architecture
        :return: str
        """

        n_weights, n_biases = 0, 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                n_weights += layer.W.view(-1).shape[0]
                n_biases  += layer.b.view(-1).shape[0]

        return f'Model has {n_weights+n_biases} parameters ({n_weights} weights, {n_biases} biases).'


    def __call__(self, X, training=True):
        """
        Forward propagation from input layer to the pre-activation of the output layer
        :param X: input (tensor of shape (n_features, n_samples))
        :return: Pre-activation of the output layer
        """

        n_samples = X.shape[1]
        assert(X.shape[0] == self.dim_in)

        for layer in self.layers:
            ### START CODE HERE ### (≈ 1 line of code)
            X = layer(X, training=training)
            ### END CODE HERE ###

        return X

    def backward(self, dF):
        """
        Backpropagation from the pre-activation of the output layer
        :param dF: Gradient of loss with respect to the pre-activation of the output layer
        :return: None
        """

        for layer in reversed(self.layers):
            ### START CODE HERE ### (≈ 1 line of code)
            dF = layer.backward(dF)
            ### END CODE HERE ###

        return None



def debug():


    nn = ShallowNeuralNet((5, 4, 3, 2))
    print(nn)
    for layer in nn.layers:
        print(layer)

    X = torch.randn((5,3))  # 5 features, 3 samples
    F = nn(X)

    print('------')

    dF = torch.randn((2,3))
    nn.backward(dF)

    for layer in reversed(nn.layers):
        if isinstance(layer, Linear):
            print('dW =', layer.dW)
            print('db =', layer.db)



if __name__ == '__main__':
    debug()
