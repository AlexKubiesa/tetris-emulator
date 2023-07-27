from abc import ABC, abstractmethod
import random

import numpy.typing as npt
import numpy as np
import torch
import torch.nn.functional as F

import recording
from models import TetrisModel


class CellTypes:
    # A cell beyond the edge of the board.
    BOUNDARY = -1
    # An empty cell.
    EMPTY = 0
    # A cell with a block in it.
    BLOCK = 1


class EventTypes:
    # Block should drop one space - timer ticked or user pressed Down
    DROP = 0
    # User wants to move block left
    LEFT = 1
    # User wants to move block right
    RIGHT = 2


class TetrisEngine(ABC):
    """The Tetris engine runs the main body of the game. It is responsible for predicting the next state of
    the board at each time step.
    """

    @abstractmethod
    def reset(self) -> tuple[npt.NDArray[np.int32], bool]:
        """Starts or resets the game and returns the initial state of the board.

        Returns:
            `board`   : An array of shape `(width, height)` representing the state of the board at the next time step.
            `gameover`: Whether the game has ended.
        """
        pass

    @abstractmethod
    def step(self, event_type: int) -> tuple[npt.NDArray[np.int32], bool]:
        """Advances one time step and returns the new state of the board.

        Arguments:
            `event_type`: An (integer) event taken from the class `EventTypes`.

        Returns:
            `board`   : An array of shape `(width, height)` representing the state of the board at the next time step.
            `gameover`: Whether the game has ended.
        """
        pass


tetris_shapes = [
    np.array([[1, 1, 1], [0, 1, 0]], dtype=np.int32),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=np.int32),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int32),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=np.int32),
    np.array([[0, 0, 1], [1, 1, 1]], dtype=np.int32),
    np.array([[1, 1, 1, 1]], dtype=np.int32),
    np.array([[1, 1], [1, 1]], dtype=np.int32),
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
        self.gameover = None

    def reset(self) -> tuple[npt.NDArray[np.int32], bool]:
        self.board = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.gameover = False
        return self.board, self.gameover

    def step(self, event_type: int) -> tuple[npt.NDArray[np.int32], bool]:
        if self.board is None:
            raise RuntimeError("`reset` must be called before `step`.")

        if event_type == EventTypes.DROP:
            if (self.board[0, :] != 0).any():
                self.gameover = True
                return self.board, self.gameover
            if self.block is None:
                self.new_block()
            else:
                self.block.y += 1
            if self.check_collision_bottom(self.board, self.block):
                self.add_block_to_board(self.board, self.block)
                self.block = None
            board = self.board.copy()
            if self.block is not None:
                self.add_block_to_board(board, self.block)
            return board, self.gameover

        if event_type == EventTypes.LEFT:
            board = self.board.copy()
            if self.block is not None:
                if not self.check_collision_left(board, self.block):
                    self.block.x -= 1
                self.add_block_to_board(board, self.block)
            return board, self.gameover

        if event_type == EventTypes.RIGHT:
            board = self.board.copy()
            if self.block is not None:
                if not self.check_collision_right(board, self.block):
                    self.block.x += 1
                self.add_block_to_board(board, self.block)
            return board, self.gameover

    def new_block(self):
        shape = random.choice(tetris_shapes)
        self.block = Block(
            shape,
            int(self.cols / 2 - shape.shape[1] / 2),
            0,
        )

    def check_collision_left(self, board, block):
        if block.left <= 0:
            return True
        board_view = board[block.top : block.bottom, block.left - 1 : block.right - 1]
        overlap = block.shape & board_view
        return overlap.any()

    def check_collision_right(self, board, block):
        if block.right >= board.shape[1]:
            return True
        board_view = board[block.top : block.bottom, block.left + 1 : block.right + 1]
        overlap = block.shape & board_view
        return overlap.any()

    def check_collision_bottom(self, board, block):
        if block.bottom >= board.shape[0]:
            return True
        board_view = board[block.top + 1 : block.bottom + 1, block.left : block.right]
        overlap = block.shape & board_view
        return overlap.any()

    def add_block_to_board(self, board, block):
        board_view = board[block.top : block.bottom, block.left : block.right]
        np.add(board_view, block.shape, where=(board_view == 0), out=board_view)


class RecordingTetrisEngine(TetrisEngine):
    """A Tetris engine decorator that captures gameplay data."""

    def __init__(self, engine: TetrisEngine, folder: str):
        self.engine = engine
        self.db = recording.FileBasedDatabaseWithActions(folder)
        self.board_1 = None
        self.board_2 = None
        self.event = None

    def reset(self) -> tuple[npt.NDArray[np.int32], bool]:
        self.board_1 = None
        self.event = None
        board, gameover = self.engine.reset()
        self.board_2 = board
        return board, gameover

    def step(self, event_type: int) -> tuple[npt.NDArray[np.int32], bool]:
        board, gameover = self.engine.step(event_type)
        self.board_1 = self.board_2
        self.board_2 = board
        self.event = event_type
        boards = np.array([self.board_1, self.board_2])
        events = np.array([self.event])
        self.db.insert(boards, events)
        return board, gameover


class ModelBasedTetrisEngine(TetrisEngine):
    def __init__(self, cols, rows, mode="normal"):
        self.cols = cols
        self.rows = rows
        if mode not in ["normal", "prob"]:
            raise ValueError()
        self.mode = mode
        self.model = TetrisModel()
        self.model.load_state_dict(torch.load("tetris_emulator.pth"))
        self.model.eval()
        self.rng = np.random.default_rng()
        self.board = None
        self.gameover = None

    def reset(self) -> tuple[npt.NDArray[np.int32], bool]:
        self.board = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.gameover = False
        return self.board, self.gameover

    def step(self, event_type: int) -> tuple[npt.NDArray[np.int32], bool]:
        if self.board is None:
            raise RuntimeError("`reset` must be called before `step`.")
        with torch.no_grad():
            X = torch.tensor(self.board, dtype=torch.long)
            X = F.one_hot(X, 2)
            X = X.type(torch.float)
            X = X.permute((2, 0, 1))
            X = X.unsqueeze(0)
            probs = self.model(X).squeeze(0)
            probs = probs.numpy()
        if self.mode == "prob":
            thresholds = self.rng.random(size=probs.shape[1:], dtype=probs.dtype)
            np.greater(probs[1], thresholds, out=self.board)
        else:
            np.argmax(probs, axis=0, out=self.board)
        return self.board, False
