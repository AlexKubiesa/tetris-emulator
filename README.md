# Tetris Emulator

An experiment in game emulation through machine learning. The goal was to emulate a relatively simple game using a machine learning model.

At each time step, the state of the game grid is fed into a "Tetris engine" which predicts the new state of the grid. The game grid is then updated to this new state.

The engine can be configured to be either a hard-coded one which explicitly follows the usual Tetris rules, or a machine learning-based one which draws its predictions from an ML model. By default, the ML-based predictor is used.


# Running the game

Ensure you have Python (3.10) installed and a virtual environment created and activated. Call your virtual environment `venv` so it gets gitignored. Install dependencies with:

> `pip install -r requirements.txt`

You can run the game with:

> `python tetris.py`

Command-line help is available with:

> `python tetris.py --help`

You can also run the emulation in Visual Studio Code, where some launch profiles have been set up.


## Command-line options

**`--help`**: Shows command-line help.

**`--record`**: If set, this flag records and saves game data to disk to be used in model training. The data is saved in the `recordings` folder. This flag cannot be used when `--engine` is set to `model`.

**`--engine`**: The engine drives the Tetris game. Default value: `model`.
* `model` is a machine learning model-based predictor.
* `rule` has the Tetris rules explicitly coded in.


## Current capabilities

* Spawn new blocks when no block is falling.
* Make blocks fall on timer tick.


## Planned work

* Block spawns are currently always of the same type. Instead, the model should spawn all 7 types of block with equal probability.
* Take into account user key presses: Down to drop block faster, Left/Right to move block, Up to rotate block and Return to drop block instantly.
* Clear rows when filled.
* Blocks are currently all of the same colour, but it would look nicer if they all had different colours. To support this, the model should support 8 cell types (empty plus the 7 colours) instead of the current 2.
* Increase the score when a user clears rows or manually drops blocks.


## Limitations

The limitations below will likely not be addressed in this project as they would significantly increase the size of the model and the complexity of training it.

* Instead of being run once every game frame, the model only gets called when an "event" happens that should trigger a change in the game state. If instead we ran the model once per frame, it would need to learn to "do nothing" on most frames but still reliably "do something" when needed.
* The model gets fed the states of all cells on the board, rather than raw pixel data. This means we can focus on predicting the underlying game dynamics instead of learning to map many pixels to a relatively small amount of internal state.


# Retraining the model

The ML model is the generator of a GAN trained using `train_tetris_emulator.ipynb`. The model weights are in `tetris_emulator.pth`. Run the notebook to see the model training and evaluation. Training run metrics are logged to the `runs` folder and are viewable in TensorBoard.

The notebooks prefixed with `experiment_XXX` were experiments used to improve the model.

# Acknowledgements

The original inspiration for this can be traced back to Ollin Boer Bohan's blog post here: https://madebyoll.in/posts/game_emulation_via_dnn/

The Tetris implementation is based on a gist by `silvasur` here: https://gist.github.com/silvasur/565419
