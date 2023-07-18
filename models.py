import torch
from torch import nn


class TetrisModel(nn.Module):
    """Predicts the next state of the cells.

    Inputs:
        x: Tensor of float32 of shape (batch_size, channels, height, width). channels = 2 is the one-hot encoding of cell types, with
           0 for empty cells and 1 for filled cells. height = 22 and width = 10 are the dimensions of the game board.
        z: Tensor of float32 of shape (batch_size, 4). The entries should be random numbers sampled from a uniform distribution.

    Returns: Tensor of float32 of shape (batch_size, height, width), logits for the new cells. Probabilities close to 0 (negative logits)
             correspond to empty cells, and probabilities close to 1 (positive logits) correspond to filled cells.
    """

    def __init__(self):
        super().__init__()

        self.loc = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        for i in [0, 3]:
            m = self.loc[i]
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

        self.glob = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(160, 10),
            nn.ReLU(),
        )

        for i in [0, 4]:
            m = self.glob[i]
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

        for i in [9]:
            m = self.glob[i]
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            nn.init.constant_(m.bias, 0.01)

        self.head = nn.Sequential(
            nn.Conv2d(26, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 2, 1),
            nn.Softmax(dim=1),
        )

        for i in [0, 3, 6]:
            m = self.head[i]
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        z = torch.rand(batch_size, 4)
        z = z[:, :, None, None]  # Expand dims to match x
        z = z.repeat(1, 1, height, width)  # Upscale to image size
        x = torch.cat((x, z), dim=1)

        x = self.loc(x)

        x_glob = self.glob(x)
        x_glob = x_glob[:, :, None, None]  # Expand dims
        x_glob = x_glob.repeat(1, 1, height, width)  # Upscale to image size
        x = torch.cat((x, x_glob), dim=1)

        y = self.head(x)
        return y


class TetrisDiscriminator(nn.Module):
    """A discriminator for the cell state predictions. Assesses the output of the generator.

    Inputs:
        x: Tensor of float32 of shape (batch_size, channels, height, width). channels = 2 is the one-hot encoding of cell types, with
           0 for empty cells and 1 for filled cells. height = 22 and width = 10 are the dimensions of the game board.
        y: Tensor of float32 of shape (batch_size, channels, height, width), as with x. This should be either the output of the
           generator or the one-hot encoding of the ground truth of the next cell states.

    Returns: Tensor of float32 of shape (batch_size, 1), decisions on whether the data are real or fake. Probabilities close to 0 (negative logits)
             correspond to fake data, and probabilities close to 1 (positive logits) correspond to real data.
    """

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(112, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Flatten(start_dim=0),
        )

        for i in [0, 3, 5, 8]:
            m = self.body[i]
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            nn.init.constant_(m.bias, 0.01)

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        logits = self.body(x)
        return logits
