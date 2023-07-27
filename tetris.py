# Original Tetris implementation: https://gist.github.com/silvasur/565419

# Control keys:
#       Down - Drop block faster
# Left/Right - Move block (not implemented)
#         Up - Rotate block clockwise (not implemented)
#     Escape - Quit game
#          P - Pause game
#     Return - Instant drop (not implemented)


import sys
import argparse

import pygame
import numpy as np

from engines import (
    EventTypes,
    ModelBasedTetrisEngine,
    RecordingTetrisEngine,
    RuleBasedTetrisEngine,
    TetrisEngine,
)

# The configuration
CELL_SIZE = 18
COLS = 10
ROWS = 22
MAX_FPS = 30

CELL_COLORS = [
    # Background 1
    (0, 0, 0),
    # Block colors
    (255, 85, 85),
    (100, 200, 115),
    (120, 108, 245),
    (255, 140, 50),
    (50, 120, 52),
    (146, 202, 73),
    (150, 161, 218),
    # Background 2
    (35, 35, 35),
]


class TetrisApp(object):
    def __init__(self, engine: TetrisEngine):
        self.engine = engine

        pygame.init()
        pygame.key.set_repeat(250, 25)
        self.width = CELL_SIZE * (COLS + 6)
        self.height = CELL_SIZE * ROWS
        self.rlim = CELL_SIZE * COLS
        self.bground_grid = np.zeros((ROWS, COLS), dtype=np.int32)
        self.bground_grid[::2, ::2] = 8
        self.bground_grid[1::2, 1::2] = 8

        self.default_font = pygame.font.Font(pygame.font.get_default_font(), 12)

        self.screen = pygame.display.set_mode((self.width, self.height))
        # We do not need mouse movement events, so we block them.
        pygame.event.set_blocked(pygame.MOUSEMOTION)
        self.init_game()

    def init_game(self):
        self.board, self.gameover = self.engine.reset()
        self.level = 1
        self.score = 0
        self.lines = 0
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000)

    def disp_msg(self, msg, topleft):
        x, y = topleft
        for line in msg.splitlines():
            self.screen.blit(
                self.default_font.render(line, False, (255, 255, 255), (0, 0, 0)),
                (x, y),
            )
            y += 14

    def center_msg(self, msg):
        for i, line in enumerate(msg.splitlines()):
            msg_image = self.default_font.render(
                line, False, (255, 255, 255), (0, 0, 0)
            )

            msgim_center_x, msgim_center_y = msg_image.get_size()
            msgim_center_x //= 2
            msgim_center_y //= 2

            self.screen.blit(
                msg_image,
                (
                    self.width // 2 - msgim_center_x,
                    self.height // 2 - msgim_center_y + i * 22,
                ),
            )

    def draw_matrix(self, matrix, offset):
        off_x, off_y = offset
        for y, row in enumerate(matrix):
            for x, val in enumerate(row):
                if val:
                    pygame.draw.rect(
                        self.screen,
                        CELL_COLORS[val],
                        pygame.Rect(
                            (off_x + x) * CELL_SIZE,
                            (off_y + y) * CELL_SIZE,
                            CELL_SIZE,
                            CELL_SIZE,
                        ),
                        0,
                    )

    def add_cl_lines(self, n):
        linescores = [0, 40, 100, 300, 1200]
        self.lines += n
        self.score += linescores[n] * self.level
        if self.lines >= self.level * 6:
            self.level += 1
            newdelay = 1000 - 50 * (self.level - 1)
            newdelay = 100 if newdelay < 100 else newdelay
            pygame.time.set_timer(pygame.USEREVENT + 1, newdelay)

    def quit(self):
        self.center_msg("Exiting...")
        pygame.display.update()
        sys.exit()

    def drop(self, manual):
        if not self.gameover and not self.paused:
            self.score += 1 if manual else 0
            self.board, self.gameover = self.engine.step(EventTypes.DROP)

    def step_model(self, event):
        if not self.gameover and not self.paused:
            self.board, self.gameover = self.engine.step(event)

    def insta_drop(self):
        raise NotImplementedError()

    def rotate_block(self):
        raise NotImplementedError()

    def toggle_pause(self):
        self.paused = not self.paused

    def start_game(self):
        if self.gameover:
            self.init_game()
            self.gameover = False

    def run(self):
        key_actions = {
            "ESCAPE": self.quit,
            "LEFT": lambda: self.step_model(EventTypes.LEFT),
            "RIGHT": lambda: self.step_model(EventTypes.RIGHT),
            "DOWN": lambda: self.drop(True),
            "UP": lambda: self.step_model(EventTypes.ROTATE),
            "p": self.toggle_pause,
            "SPACE": self.start_game,
            "RETURN": self.insta_drop,
        }

        self.gameover = False
        self.paused = False

        dont_burn_my_cpu = pygame.time.Clock()
        while 1:
            self.screen.fill((0, 0, 0))
            if self.gameover:
                self.center_msg(
                    f"Game Over!\nYour score: {self.score}\nPress space to continue"
                )
            else:
                if self.paused:
                    self.center_msg("Paused")
                else:
                    pygame.draw.line(
                        self.screen,
                        (255, 255, 255),
                        (self.rlim + 1, 0),
                        (self.rlim + 1, self.height - 1),
                    )
                    self.disp_msg("Next:", (self.rlim + CELL_SIZE, 2))
                    self.disp_msg(
                        f"Score: {self.score}\n\nLevel: {self.level}\nLines: {self.lines}",
                        (self.rlim + CELL_SIZE, CELL_SIZE * 5),
                    )
                    self.draw_matrix(self.bground_grid, (0, 0))
                    self.draw_matrix(self.board, (0, 0))
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.USEREVENT + 1:
                    self.drop(False)
                elif event.type == pygame.QUIT:
                    self.quit()
                elif event.type == pygame.KEYDOWN:
                    for key in key_actions:
                        if event.key == eval("pygame.K_" + key):
                            key_actions[key]()

            dont_burn_my_cpu.tick(MAX_FPS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record",
        action="store_true",
        help="If set, this flag records game data for use in model training.",
    )
    parser.add_argument(
        "--engine",
        default="model",
        choices=["model", "rule"],
        help="The engine drives the Tetris game. 'model' is a machine learning model-based engine."
        + " 'rule' is an explicit rule-based engine with the Tetris rules explicitly coded in.",
    )

    args = parser.parse_args()

    if args.record and (args.engine == "model"):
        raise Exception(
            "Training and test data should be recorded with the rule-based engine."
        )

    engine: TetrisEngine
    if args.engine == "rule":
        engine = RuleBasedTetrisEngine(COLS, ROWS)
    else:
        engine = ModelBasedTetrisEngine(COLS, ROWS, mode="normal")

    if args.record:
        engine = RecordingTetrisEngine(engine, "recordings")

    App = TetrisApp(engine)
    App.run()
