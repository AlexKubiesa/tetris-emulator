from abc import ABC, abstractmethod
import random

import numpy.typing as npt
import numpy as np
import torch
import torch.nn.functional as F

from recording import RecordingFolder
from models import TetrisModel


class CellTypes:
    # A cell beyond the edge of the board.
    BOUNDARY = -1
    # An empty cell.
    EMPTY = 0
    # A cell with a block in it.
    BLOCK = 1


class EventTypes:
    # The game has just started.
    GAME_START = 0
    # It is time for a block to drop one space.
    DROP = 1


class TetrisEngine(ABC):
    """The Tetris engine runs the main body of the game. It is responsible for predicting the next state of
    the board at each time step.
    """

    @abstractmethod
    def step(self, event_type: int) -> tuple[npt.NDArray[np.int32], bool]:
        """Advances one time step and returns the new state of the board.

        Arguments:
            `board`: Array of shape `(width, height)`, where each entry is the integer value of a cell type. The
                     cell type zero is reserved for "boundary" cells that are beyond the edge of the board.

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

    def step(self, event_type: int) -> tuple[npt.NDArray[np.int32], bool]:
        if event_type == EventTypes.GAME_START:
            self.board = np.zeros((self.rows, self.cols), dtype=np.int32)
            self.gameover = False
            return self.board, self.gameover
        if event_type == EventTypes.DROP:
            if (self.board[0, :] != 0).any():
                self.gameover = True
                return self.board, self.gameover
            if self.block is None:
                self.new_block()
            else:
                self.block.y += 1
            if self.check_collision(self.board, self.block):
                self.add_block_to_board(self.board, self.block)
                self.block = None
            board = self.board.copy()
            if self.block is not None:
                self.add_block_to_board(board, self.block)
            return board, self.gameover

    def new_block(self):
        shape = random.choice(tetris_shapes)
        self.block = Block(
            shape,
            int(self.cols / 2 - shape.shape[1] / 2),
            0,
        )

    def check_collision(self, board, block):
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

    buffer_length = 2

    def __init__(self, engine: TetrisEngine, folder: str):
        self.engine = engine
        self.folder = RecordingFolder(folder)
        self.board_buffer = []

    def step(self, event_type: int) -> tuple[npt.NDArray[np.int32], bool]:
        board, gameover = self.engine.step(event_type)
        self.board_buffer.append(board)
        if len(self.board_buffer) >= self.buffer_length:
            path = self.folder.next_file()
            np.save(path, np.array(self.board_buffer))
            self.board_buffer.clear()
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

    def step(self, event_type: int) -> tuple[npt.NDArray[np.int32], bool]:
        if event_type == EventTypes.GAME_START:
            self.board = np.zeros((self.rows, self.cols), dtype=np.int32)
            return self.board, False
        if event_type == EventTypes.DROP:
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
