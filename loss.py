import torch

class CELoss:
    """
    Multi-class Cross Entropy Loss
    """

    def __init__(self):
        """
        Constructor
        """

        self.Y = None
        self.F = None

    def __call__(self, Y, F, training=True):
        """
        Forward propagation of CE Loss
        :param Y: Ground truth, array of shape (n_output, batch size)
        :param F: Model prediction, array of shape (n_output, batch size)
        :return:
        """

        ### START CODE HERE ### (≈ 1-2 lines of code)
        loss = -torch.sum(Y * torch.log_softmax(F, dim=0)) / F.shape[1]
        ### END CODE HERE ###

        assert(Y.shape == F.shape)
        if training:
            self.Y = Y.clone()
            self.F = F.clone()

        return loss

    def backward(self):
        """
        Backward propagation for CE Loss
        :return: pre-activation gradient (of loss w.r.t. F)
        """

        ### START CODE HERE ### (≈ 1 lines of code)
        dF = (torch.softmax(self.F, dim=0) - self.Y) / self.F.shape[1]
        ### END CODE HERE ###

        return dF


def debug():

    (D_i, N) = 3, 4

    Y = torch.tensor([
        [1, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
    ], dtype=torch.float32)
    print('Y =', Y)
    F = torch.tensor([
        [-2, 3, 8, -7],
        [10, 2, 8, 4],
        [5, 7, 1, 3],
    ], dtype=torch.float32)
    print('F =', F)

    ce_loss = CELoss()
    loss = ce_loss(Y, F)
    print('Loss =', loss)

    dF = ce_loss.backward()
    print('dF =', dF)



if __name__ == '__main__':
    debug()
