import torch
import numpy as np
from scipy.stats import entropy
from math import exp


NUM_SPAWN_TYPES = 7

BLOCKS = [
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
            [0, 0, 0, 0, 2, 2, 0, 0, 0, 0],
            [0, 0, 0, 2, 2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.int,
    ),  # S
    torch.tensor(
        [
            [0, 0, 0, 3, 3, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 3, 3, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.int,
    ),  # Z
    torch.tensor(
        [
            [0, 0, 0, 4, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 4, 4, 4, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.int,
    ),  # J
    torch.tensor(
        [
            [0, 0, 0, 0, 0, 5, 0, 0, 0, 0],
            [0, 0, 0, 5, 5, 5, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.int,
    ),  # L
    torch.tensor(
        [
            [0, 0, 0, 6, 6, 6, 6, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.int,
    ),  # I
    torch.tensor(
        [
            [0, 0, 0, 0, 7, 7, 0, 0, 0, 0],
            [0, 0, 0, 0, 7, 7, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.int,
    ),  # O
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
    diff = (classes_y - classes_x)[:3, :]

    # Each example in the batch will be matched by at most one spawn type, and when it matches, we return that spawn type.
    for type, block in enumerate(BLOCKS):
        # If the first frame overlaps with the block that should spawn, then it's not a spawn
        if ((classes_x[:3, :] > 0) & (block > 0)).any():
            continue
        # Given the first frame is zero wherever the block is nonzero, check that the specified block appeared and that
        # no other cells in the first three rows changed.
        if (diff == block).all(-1).all(-1).item():
            return type

    return None


class BoardAccuracy:
    """Measures the proportion of boards where all cells were predicted correctly."""

    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self.num_correct = 0
        self.dataset_size = 0

    def update_state(self, classes_y_pred, classes_y):
        self.num_correct += (
            (classes_y_pred == classes_y).all(-1).all(-1).type(torch.int).sum().item()
        )
        self.dataset_size += classes_y_pred.size(0)

    def result(self):
        return self.num_correct / self.dataset_size


class BoardPlausibility:
    """Measures the proportion of predictions that would be plausible according to a perfect discriminator."""

    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self.num_plausible = 0
        self.dataset_size = 0

    def update_state(self, classes_x, classes_y_pred, classes_y):
        for i in range(classes_x.size(0)):
            y_spawn_type = get_block_spawn_type(classes_x[i], classes_y[i])
            if y_spawn_type is None:
                # If it's a block fall, expect the boards to match exactly.
                self.num_plausible += int(
                    (classes_y_pred[i] == classes_y[i]).all(-1).all(-1).item()
                )
            else:
                # If it's a block spawn, allow any spawn type, but check the rows below the top 3 to make sure
                # they match exactly.
                y_pred_spawn_type = get_block_spawn_type(
                    classes_x[i], classes_y_pred[i]
                )
                self.num_plausible += int(
                    (y_pred_spawn_type is not None)
                    and (
                        (classes_y_pred[i, 3:, :] == classes_y[i, 3:, :])
                        .all(-1)
                        .all(-1)
                        .item()
                    )
                )

        self.dataset_size += classes_x.size(0)

    def result(self):
        return self.num_plausible / self.dataset_size


class CellAccuracy:
    """Measures the proportion of cells predicted correctly."""

    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self.num_correct = 0
        self.dataset_size = 0

    def update_state(self, classes_y_pred, classes_y):
        self.num_correct += (
            (classes_y_pred == classes_y)
            .type(torch.float)
            .mean(dim=(1, 2))
            .sum()
            .item()
        )
        self.dataset_size += classes_y_pred.size(0)

    def result(self):
        return self.num_correct / self.dataset_size


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


class SpawnPrecision:
    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self.num_true_positives = np.float32(0.0)
        self.num_spawns_pred = np.float32(0.0)

    def update_state(self, classes_x, classes_y_pred, classes_y):
        spawns = (classes_x[:, 0, :] == 0).all(-1) & (classes_y[:, 0, :] > 0).any(-1)
        spawns_pred = (classes_x[:, 0, :] == 0).all(-1) & (
            classes_y_pred[:, 0, :] > 0
        ).any(-1)

        self.num_true_positives += (
            (spawns & spawns_pred).type(torch.float).sum().numpy()
        )
        self.num_spawns_pred += spawns_pred.type(torch.float).sum().numpy()

    def result(self):
        return self.num_true_positives / self.num_spawns_pred


class SpawnRecall:
    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self.num_true_positives = 0
        self.num_spawns = 0

    def update_state(self, classes_x, classes_y_pred, classes_y):
        spawns = (classes_x[:, 0, :] == 0).all(-1) & (classes_y[:, 0, :] > 0).any(-1)
        spawns_pred = (classes_x[:, 0, :] == 0).all(-1) & (
            classes_y_pred[:, 0, :] > 0
        ).any(-1)

        self.num_true_positives += (spawns & spawns_pred).type(torch.int).sum().item()
        self.num_spawns += spawns.type(torch.int).sum().item()

    def result(self):
        return self.num_true_positives / self.num_spawns


class SpawnValidity:
    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self.num_valid_spawns_pred = np.float32(0.0)
        self.num_spawns_pred = np.float32(0.0)

    def update_state(self, classes_x, classes_y_pred):
        spawns_pred = (classes_x[:, 0, :] == 0).all(-1) & (
            classes_y_pred[:, 0, :] > 0
        ).any(-1)

        num_valid_spawns_pred = np.float32(0.0)
        for i in range(classes_x.size(0)):
            if not spawns_pred[i]:
                # Avoid computing spawn validity if there is no spawn
                continue
            valid_spawn = (
                get_block_spawn_type(classes_x[i], classes_y_pred[i]) is not None
            )
            self.num_valid_spawns_pred += np.float32(valid_spawn)

        self.num_valid_spawns_pred += num_valid_spawns_pred
        self.num_spawns_pred += spawns_pred.type(torch.float).sum().numpy()

    def result(self):
        return self.num_valid_spawns_pred / self.num_spawns_pred


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
