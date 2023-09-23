from abc import ABC, abstractmethod
import random

import numpy.typing as npt
import numpy as np
import torch
import torch.nn.functional as F

import recording
import models


class EventTypes:
    # Block should drop one space - timer ticked or user pressed Down
    DROP = 0
    # User wants to move block left
    LEFT = 1
    # User wants to move block right
    RIGHT = 2
    # User wants to rotate block
    ROTATE = 3
    # The user wants to instantly drop a block so it lands
    INSTA_DROP = 4


EVENT_NAMES = ["Drop", "Left", "Right", "Rotate", "Insta-drop"]

NUM_CELL_TYPES = 8
NUM_EVENT_TYPES = 5


class TetrisEngine(ABC):
    """The Tetris engine runs the main body of the game. It is responsible for predicting the next state of
    the board at each time step.
    """

    @abstractmethod
    def reset(self) -> npt.NDArray[np.int32]:
        """Starts or resets the game and returns the initial state of the board.

        Returns:
            `board`: An array of shape `(width, height)` representing the state of the board at the next time step.
        """
        pass

    @abstractmethod
    def step(self, event_type: int) -> tuple[npt.NDArray[np.int32], bool]:
        """Advances one time step and returns the new state of the board.

        Arguments:
            `event_type`: An (integer) event taken from the class `EventTypes`.

        Returns:
            `board`: An array of shape `(width, height)` representing the state of the board at the next time step.
        """
        pass


TETRIS_SHAPES = [
    np.array([[1, 1, 1], [0, 1, 0]], dtype=np.int32),
    np.array([[0, 2, 2], [2, 2, 0]], dtype=np.int32),
    np.array([[3, 3, 0], [0, 3, 3]], dtype=np.int32),
    np.array([[4, 0, 0], [4, 4, 4]], dtype=np.int32),
    np.array([[0, 0, 5], [5, 5, 5]], dtype=np.int32),
    np.array([[6, 6, 6, 6]], dtype=np.int32),
    np.array([[7, 7], [7, 7]], dtype=np.int32),
]


class Block:
    def __init__(self, shape, x, y):
        self.shape = shape
        self.x = x
        self.y = y

    @property
    def width(self):
        return self.shape.shape[1]

    @property
    def height(self):
        return self.shape.shape[0]

    @property
    def left(self):
        return self.x

    @property
    def right(self):
        return self.x + self.shape.shape[1]

    @property
    def top(self):
        return self.y

    @property
    def bottom(self):
        return self.y + self.shape.shape[0]


class RuleBasedTetrisEngine(TetrisEngine):
    def __init__(self, cols, rows):
        self.cols = cols
        self.rows = rows
        self.block = None
        self.board = None
        self.gameover = False

    def reset(self) -> npt.NDArray[np.int32]:
        self.board = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.gameover = False
        return self.board

    def step(self, event_type: int) -> tuple[npt.NDArray[np.int32], bool]:
        if self.board is None:
            raise RuntimeError("`reset` must be called before `step`.")

        if self.gameover:
            raise RuntimeError("Game is over.")

        if event_type == EventTypes.DROP:
            if self.block is None:
                self.board = self.clear_rows(self.board)
                self.new_block()
                if self.check_collision(self.board, self.block):
                    self.gameover = True
            else:
                self.block.y += 1
            if self.check_collision(self.board, self.block, offset=(0, 1)):
                self.add_block_to_board(self.board, self.block)
                self.block = None

        elif event_type == EventTypes.LEFT:
            if self.block is not None:
                if not self.check_collision(self.board, self.block, offset=(-1, 0)):
                    self.block.x -= 1
                    if self.check_collision(self.board, self.block, offset=(0, 1)):
                        self.land_block()

        elif event_type == EventTypes.RIGHT:
            if self.block is not None:
                if not self.check_collision(self.board, self.block, offset=(1, 0)):
                    self.block.x += 1
                    if self.check_collision(self.board, self.block, offset=(0, 1)):
                        self.land_block()

        elif event_type == EventTypes.ROTATE:
            if self.block is not None:
                rotated_block = Block(
                    np.rot90(self.block.shape, -1), self.block.x, self.block.y
                )

                if not self.check_collision(self.board, rotated_block):
                    self.block = rotated_block
                elif not self.check_collision(
                    self.board, rotated_block, offset=(-1, 0)
                ):
                    rotated_block.x -= 1
                    self.block = rotated_block
                elif not self.check_collision(self.board, rotated_block, offset=(1, 0)):
                    rotated_block.x += 1
                    self.block = rotated_block
                elif not self.check_collision(
                    self.board, rotated_block, offset=(0, -1)
                ):
                    rotated_block.y -= 1
                    self.block = rotated_block

                if self.check_collision(self.board, self.block, offset=(0, 1)):
                    self.land_block()

        elif event_type == EventTypes.INSTA_DROP:
            if self.block is not None:
                while not self.check_collision(self.board, self.block, offset=(0, 1)):
                    self.block.y += 1
                self.land_block()

        board = self.board.copy()
        if self.block is not None:
            self.add_block_to_board(board, self.block)
        return board, self.gameover

    def new_block(self):
        shape = random.choice(TETRIS_SHAPES)
        self.block = Block(
            shape,
            int(self.cols / 2 - shape.shape[1] / 2),
            0,
        )

    def clear_rows(self, board):
        """Clears any filled rows on the board and moves the remaining rows down."""
        idxs_to_clear = board.min(axis=-1).nonzero()[0]
        new_board = list(board)
        for idx in reversed(idxs_to_clear):
            del new_board[idx]
        new_board = [np.zeros(self.cols)] * len(idxs_to_clear) + new_board
        new_board = np.array(new_board, dtype=np.int32)
        return new_board

    def check_collision(self, board, block, offset=(0, 0)):
        """Checks whether the block overflows the board boundary or overlaps filled cells on the board, when offset by the given amount."""
        off_x, off_y = offset
        if (
            (block.left + off_x < 0)
            or (block.right + off_x > board.shape[1])
            or (block.top + off_y < 0)
            or (block.bottom + off_y > board.shape[0])
        ):
            return True

        board_view = board[
            block.top + off_y : block.bottom + off_y,
            block.left + off_x : block.right + off_x,
        ]

        overlap = (block.shape != 0) & (board_view != 0)
        return overlap.any()

    def land_block(self):
        self.add_block_to_board(self.board, self.block)
        self.block = None

    def add_block_to_board(self, board, block):
        board_view = board[block.top : block.bottom, block.left : block.right]
        np.add(board_view, block.shape, where=(board_view == 0), out=board_view)


class RecordingTetrisEngine(TetrisEngine):
    """A Tetris engine decorator that captures gameplay data."""

    def __init__(self, engine: TetrisEngine, folder: str, filter=None):
        self.engine = engine
        self.db = recording.FileBasedDatabase(folder)
        self.filter = filter
        self.board_1 = None
        self.board_2 = None
        self.event = None

    def reset(self) -> npt.NDArray[np.int32]:
        self.board_1 = None
        self.event = None
        board = self.engine.reset()
        self.board_2 = board
        return board

    def step(self, event_type: int) -> tuple[npt.NDArray[np.int32], bool]:
        board, gameover = self.engine.step(event_type)
        self.board_1 = self.board_2
        self.board_2 = board
        self.event = event_type
        boards = np.array([self.board_1, self.board_2])
        events = np.array([self.event])
        if (self.filter is None) or self.filter.should_store(boards, events):
            self.db.insert(boards, events)
        return board, gameover


class ProbabilisticRecordingFilter:
    def __init__(self, prob):
        self.prob = prob

    def should_store(self, boards, events):
        return random.random() < self.prob


class ModelBasedTetrisEngine(TetrisEngine):
    def __init__(self, cols, rows, mode="normal"):
        self.cols = cols
        self.rows = rows
        if mode not in ["normal", "prob"]:
            raise ValueError()
        self.mode = mode
        self.model = models.GameganGenerator()
        self.model.load_state_dict(torch.load("gamegan_emulator.pth"))
        self.model.eval()
        self.rng = np.random.default_rng()
        self.board = None

    def reset(self) -> npt.NDArray[np.int32]:
        self.board = np.zeros((self.rows, self.cols), dtype=np.int32)
        return self.board

    def step(self, event_type: int) -> tuple[npt.NDArray[np.int32], bool]:
        if self.board is None:
            raise RuntimeError("`reset` must be called before `step`.")

        with torch.no_grad():
            b = torch.tensor(self.board, dtype=torch.long)
            b = F.one_hot(b, NUM_CELL_TYPES)
            b = b.type(torch.float)
            b = b.permute((2, 0, 1))
            b = b.unsqueeze(0)

            e = torch.tensor(event_type, dtype=torch.long)
            e = F.one_hot(e, NUM_EVENT_TYPES)
            e = e.type(torch.float)
            e = e.unsqueeze(0)

            probs = self.model(b, e).squeeze(0)
            probs = probs.numpy()

        if self.mode == "prob":
            channels, height, width = probs.shape
            thresholds = np.cumsum(probs, axis=0)
            randoms = self.rng.random((height, width))
            np.sum(randoms > thresholds, axis=0, out=self.board)
        else:
            np.argmax(probs, axis=0, out=self.board)

        return self.board, False
