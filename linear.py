import torch
import numpy as np


class Linear:
    """
    Linear function
    """

    def __init__(self, D_i, D_o):
        """

        :param D_i: Dimension of input
        :param D_o: Dimension of output (Number of neurons)
        """

        # Parameters
        # W is weight matrix: array of shape (D_o, D_i)
        # b is bias vector: array of shape (D_o, 1)
        ### START CODE HERE ### (≈ 1 line of code)
        # He initialization (gain=2, mode=fan_in)
        self.W = torch.sqrt(torch.tensor(2.0 / D_i)) * torch.randn((D_o, D_i))
        ### END CODE HERE ###
        self.b = torch.zeros(D_o, 1)

        # Forward propagation cache
        self.H = None

        # Computed gradients
        self.dW = None
        self.db = None

    def __call__(self, H, training=True):
        """
        Forward propagation for linear function
        :param H: Post-activation from previous layer (or input data): array of shape (D_i, N) where N is batch size
        :return: Output of linear unit
        """

        ### START CODE HERE ### (≈ 1 line of code)
        F = torch.matmul(self.W, H) + self.b
        ### END CODE HERE ###

        assert(F.shape == (self.W.shape[0], H.shape[1]))

        if training:
            self.H = H.clone()

        return F


    def backward(self, dF):
        """
        Backpropagation for linear function
        :param dF: Gradient of the loss with respect to the output of linear function (pre-activation)
        :return:
        """

        ### START CODE HERE ### (≈ 3 lines of code)
        dW = torch.matmul(dF, self.H.t())
        db = torch.sum(dF, dim=1, keepdim=True)
        dH = torch.matmul(self.W.t(), dF)
        ### END CODE HERE ###

        self.dW = dW.clone()
        self.db = db.clone()

        return dH


def debug():

    D_i, D_o, N = 3, 2, 5
    linear = Linear(D_i, D_o)

    H = torch.randn((D_i, N))
    F = linear(H)

    print('F =', F)

    dF = torch.randn_like(F)
    dH = linear.backward(dF)
    print('dH =', dH)
    print('dW =', linear.dW)
    print('db =', linear.db)



if __name__ == '__main__':

    debug()
