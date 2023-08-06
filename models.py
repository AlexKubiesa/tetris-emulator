import torch
from torch import nn


NUM_CELL_TYPES = 8
NUM_EVENT_TYPES = 5
NUM_RANDOM_INPUTS = 4


class TetrisModel(nn.Module):
    """Predicts the next state of the cells.

    Inputs:
        b: Tensor of float32 of shape (batch_size, channels, height, width). channels is the one-hot encoding of cell types. height = 22
            and width = 10 are the dimensions of the game board.
        e: Tensor of float32 of shape (batch_size, channels). channels is the one-host encoding of event types.

    Returns: Tensor of float32 of shape (batch_size, height, width), logits for the new cells. Probabilities close to 0 (negative logits)
             correspond to empty cells, and probabilities close to 1 (positive logits) correspond to filled cells.
    """

    def __init__(self):
        super().__init__()

        self.loc = nn.Sequential(
            nn.Conv2d(
                NUM_CELL_TYPES + NUM_EVENT_TYPES + NUM_RANDOM_INPUTS,
                16,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(26, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, NUM_CELL_TYPES, kernel_size=1),
            nn.Softmax(dim=1),
        )

        for i in [0, 3, 6]:
            m = self.head[i]
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

    def forward(self, b, e):
        batch_size, cell_channels, height, width = b.shape

        # Upscale events to board size
        e = e[:, :, None, None].repeat(1, 1, height, width)

        # Generate random inputs
        z = torch.rand(batch_size, NUM_RANDOM_INPUTS)
        z = z[:, :, None, None].repeat(1, 1, height, width)

        # Combine cells, events and random inputs
        x = torch.cat((b, e, z), dim=1)

        x = self.loc(x)

        x_glob = self.glob(x)
        x_glob = x_glob[:, :, None, None].repeat(1, 1, height, width)
        x = torch.cat((x, x_glob), dim=1)

        y = self.head(x)
        return y


class TetrisDiscriminator(nn.Module):
    """A discriminator for the cell state predictions. Assesses the output of the generator.

    Inputs:
        b: Tensor of float32 of shape (batch_size, channels, height, width). channels is the one-hot encoding of cell types. height = 22
            and width = 10 are the dimensions of the game board.
        e: Tensor of float32 of shape (batch_size, channels). channels is the one-host encoding of event types.
        y: Tensor of float32 of shape (batch_size, channels, height, width), as with x. This should be either the output of the
            generator or the one-hot encoding of the ground truth of the next cell states.

    Returns: Tensor of float32 of shape (batch_size, 1), decisions on whether the data are real or fake. Probabilities close to 0 (negative logits)
             correspond to fake data, and probabilities close to 1 (positive logits) correspond to real data.
    """

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                2 * NUM_CELL_TYPES + NUM_EVENT_TYPES, 16, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3),
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

    def forward(self, b, e, y):
        batch_size, cell_channels, height, width = b.shape

        # Upscale events to board size
        e = e[:, :, None, None].repeat(1, 1, height, width)

        # Combine cells, events and random inputs
        x = torch.cat((b, e, y), dim=1)

        logits = self.body(x)
        return logits


class Conv2dLeakyReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        use_batch_norm=False,
        negative_slope=0.0,
    ):
        super().__init__()

        self.use_batch_norm = use_batch_norm

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=(not use_batch_norm),
        )
        nn.init.kaiming_uniform_(self.conv.weight, a=negative_slope)
        if not use_batch_norm:
            nn.init.constant_(self.conv.bias, 0.01)

        if use_batch_norm:
            self.norm = nn.BatchNorm2d(out_channels)

        self.relu = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        x = self.conv(x)
        if self.use_batch_norm:
            x = self.norm(x)
        x = self.relu(x)
        return x


class ConvTranspose2dLeakyReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        use_batch_norm=False,
        negative_slope=0.0,
    ):
        super().__init__()

        self.use_batch_norm = use_batch_norm

        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=(not use_batch_norm),
        )
        nn.init.kaiming_uniform_(self.conv.weight, a=negative_slope)
        if not use_batch_norm:
            nn.init.constant_(self.conv.bias, 0.01)

        if use_batch_norm:
            self.norm = nn.BatchNorm2d(out_channels)

        self.relu = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        x = self.conv(x)
        if self.use_batch_norm:
            x = self.norm(x)
        x = self.relu(x)
        return x


class LinearLeakyReLU(nn.Module):
    def __init__(self, in_features, out_features, negative_slope=0.0):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)
        nn.init.kaiming_uniform_(self.linear.weight, a=negative_slope)
        nn.init.constant_(self.linear.bias, 0.01)

        self.relu = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x


class GameganAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        use_batch_norm = True
        leak = 0.2

        self.board_encoder = nn.Sequential(
            Conv2dLeakyReLU(
                NUM_CELL_TYPES,
                64,
                kernel_size=2,
                stride=2,
                use_batch_norm=use_batch_norm,
                negative_slope=leak,
            ),
            Conv2dLeakyReLU(
                64,
                64,
                kernel_size=3,
                use_batch_norm=use_batch_norm,
                negative_slope=leak,
            ),
            Conv2dLeakyReLU(
                64,
                64,
                kernel_size=3,
                use_batch_norm=use_batch_norm,
                negative_slope=leak,
            ),
            nn.Flatten(start_dim=1),
            LinearLeakyReLU(448, 256, negative_slope=leak),
        )

        self.renderer = nn.Sequential(
            LinearLeakyReLU(256, 448, negative_slope=leak),
            nn.Unflatten(dim=1, unflattened_size=(64, 7, 1)),
            ConvTranspose2dLeakyReLU(
                64,
                64,
                kernel_size=3,
                use_batch_norm=use_batch_norm,
                negative_slope=leak,
            ),
            ConvTranspose2dLeakyReLU(
                64,
                64,
                kernel_size=3,
                use_batch_norm=use_batch_norm,
                negative_slope=leak,
            ),
            nn.ConvTranspose2d(64, NUM_CELL_TYPES, kernel_size=2, stride=2),
            nn.Softmax(dim=1),
        )

    def forward(self, b):
        # Encode board state
        s = self.board_encoder(b)

        # Render new board
        y = self.renderer(s)
        return y
