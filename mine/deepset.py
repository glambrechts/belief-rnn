import torch.nn as nn


class DeepSet(nn.Module):
    """Deep set (see arXiv:1703.06114). The implementation follows the one from
    <https://github.com/manzilzaheer/DeepSets>.

    Arguments:
    - input_size: int
        The input size
    - representation_size: int
        The vector size for the set representation
    """

    def __init__(self, input_size, representation_size):
        super().__init__()

        self.phi = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, representation_size))

        self.rho = nn.Linear(representation_size, representation_size)

    def forward(self, input):
        """
        Computes the forward pass in the encoder and outputs the set
        representation.

        Arguments:
        - inputs: a batch of inputs of size [B, N, D] where B is the batch
            size, N, the number of elements in the set, and D the
            dimension.

        Returns:
        - representation: tensor
            The intermediate representation of the set.
        """
        features = self.phi(input)
        return self.rho(features.sum(dim=1))
