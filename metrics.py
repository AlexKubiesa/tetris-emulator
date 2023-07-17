import torch
import numpy as np
from scipy.stats import entropy
from math import exp


NUM_SPAWN_TYPES = 7

BLOCKS = [
    torch.tensor(
        [
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.int,
    ),  # I
    torch.tensor(
        [
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.int,
    ),  # O
    torch.tensor(
        [
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.int,
    ),  # J
    torch.tensor(
        [
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.int,
    ),  # T
    torch.tensor(
        [
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.int,
    ),  # S
    torch.tensor(
        [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.int,
    ),  # L
    torch.tensor(
        [
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.int,
    ),  # Z
]


def get_block_spawn_type(classes_x, classes_y):
    """Determines the type of block spawn from one frame to the next, if any.

    Inputs:
        classes_x: Tensor of int32 of shape (height, width), the first frame (with argmax applied on cell types).
        classes_y: Tensor of int32 of shape (height, width), the second frame (with argmax applied on cell types).

    Returns: (int | None) The block spawn type, if any. A spawn type of None means either no block spawned or
        the spawned shape was invalid.
    """

    # Take difference to see which cells are full but weren't before.
    diff = classes_y - classes_x

    # Each example in the batch will be matched by at most one spawn type, and when it matches, we return that spawn type.
    for type, block in enumerate(BLOCKS):
        if (diff[:3, :] == block).all(-1).all(-1).item():
            return type

    return None


class SpawnDiversity:
    """Roughly measures what proportion of block spawn types are represented with equal probability by the emulator.

    Only examples where a valid block spawn is predicted are considered. A model that predicts n spawn types with equal
    probability would score (n / NUM_SPAWN_TYPES) in this metric, where NUM_SPAWN_TYPES = 7.
    """

    def __init__(self):
        self.predicted_spawn_type_counts = np.zeros(NUM_SPAWN_TYPES)

    def reset_state(self):
        self.predicted_spawn_type_counts.fill(0)

    def update_state(self, classes_x, classes_y_pred):
        """Accumulates the metric based on a batch of data and predictions.

        Inputs:
            classes_x: Tensor of int of shape (batch_size, height, width), with 0 for empty cells and 1 for filled cells. height = 22 and
                width = 10 are the dimensions of the game board.
            classes_y_pred: Tensor of int of shape (batch_size, height, width), as with x. This should be the argmax (dim=1) of the output
                of the generator.
        """
        batch_size = classes_x.size(0)
        for i in range(batch_size):
            spawn_type = get_block_spawn_type(classes_x[i], classes_y_pred[i])
            if spawn_type is not None:
                self.predicted_spawn_type_counts[spawn_type] += 1

    def result(self):
        num_predicted_spawns = np.sum(self.predicted_spawn_type_counts)
        probs = self.predicted_spawn_type_counts / num_predicted_spawns
        H = entropy(probs)
        # A uniform random variable with n states has an entropy of log(n). We want to get n.
        equiv_num_types = exp(H)
        return equiv_num_types / NUM_SPAWN_TYPES


if __name__ == "__main__":
    metric = SpawnDiversity()

    classes_x = torch.zeros(NUM_SPAWN_TYPES, 22, 10, dtype=torch.int)
    classes_y_pred = torch.zeros(NUM_SPAWN_TYPES, 22, 10, dtype=torch.int)
    classes_y_pred[:, :3, :] = BLOCKS[0]

    metric.update_state(classes_x, classes_y_pred)
    val = metric.result()
    # Expected: 14.29%
    print(f"Minimum diversity: {val:.2%}")

    metric.reset_state()
    for i in range(NUM_SPAWN_TYPES):
        classes_y_pred[i, :3, :] = BLOCKS[i]
    metric.update_state(classes_x, classes_y_pred)
    val = metric.result()
    # Expected: 100.00%
    print(f"Maximum diversity: {val:.2%}")

    metric.reset_state()
    classes_y_pred[(NUM_SPAWN_TYPES // 2) :, :3, :] = 0
    metric.update_state(classes_x, classes_y_pred)
    val = metric.result()
    # Expected: 42.86%
    print(f"Middling diversity: {val:.2%}")
