import torch
from torch import nn


NUM_CELL_TYPES = 8
NUM_EVENT_TYPES = 5
NUM_RANDOM_INPUTS = 4


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


class GameganBoardEncoder(nn.Sequential):
    def __init__(self):
        use_batch_norm = True
        leak = 0.2

        super().__init__(
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


class GameganRenderer(nn.Sequential):
    def __init__(self):
        use_batch_norm = True
        leak = 0.2

        super().__init__(
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


class GameganAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.board_encoder = GameganBoardEncoder()
        self.renderer = GameganRenderer()

    def forward(self, b):
        # Encode board state
        s = self.board_encoder(b)

        # Render new board
        y = self.renderer(s)
        return y


class GameganGenerator(nn.Module):
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

        self.device = None
        leak = 0.2

        self.board_encoder = GameganBoardEncoder()

        self.event_encoder = nn.Sequential(
            LinearLeakyReLU(
                NUM_EVENT_TYPES + NUM_RANDOM_INPUTS, 32, negative_slope=leak
            ),
            LinearLeakyReLU(32, 32, negative_slope=leak),
            LinearLeakyReLU(32, 32, negative_slope=leak),
        )

        self.dynamics = nn.Sequential(
            LinearLeakyReLU(256 + 32, 256, negative_slope=leak),
            LinearLeakyReLU(256, 256, negative_slope=leak),
            LinearLeakyReLU(256, 256, negative_slope=leak),
        )

        self.renderer = GameganRenderer()

    def forward(self, b, e):
        batch_size, cell_channels, height, width = b.shape

        # Encode board state
        s = self.board_encoder(b)

        # Generate random inputs
        z = torch.rand(batch_size, NUM_RANDOM_INPUTS, device=self.device)

        # Encode events and random inputs
        v = self.event_encoder(torch.cat((e, z), dim=1))

        # Combine encodings
        h = torch.cat((s, v), dim=1)

        # Apply game dynamics
        h = self.dynamics(h)

        # Render new board
        y = self.renderer(h)
        return y

    def to(self, device):
        super().to(device)
        self.device = device
