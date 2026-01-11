# LUDOpy - AI Ludo Game Player

This project implements a Ludo game environment in Python and develops an Artificial Intelligence (AI) player using Q-learning. It is built on the [LUDOpy](https://github.com/SimonLBSoerensen/LUDOpy) library.

## Abstract

An artificial intelligent (AI) method to play the game ludo is developed. The method used is called Q-learning and it uses a special representation of the game so that the method is compatible with it. The representation is composed of states that the pieces can be in and actions which can be performed by the pieces. 
For the method to understand and be able to win the game training is needed. A reward function is used when training to determine if the Q-learning player is making a good or bad decision. Training is done against 3 random acting players and it achieves an average win rate of 59.5% after tuning the hyperparameters of the Q-learning algorithm. 

The algorithm is evaluated in two different test. The first test is against an alternative AI method and the second test is against the alternative AI method and 2 random acting players. In the first test the presented method achieves a win rate of 56.43% compared to 43.56% of the alternative method. In the second test it achieves a win rate of 46.89% compared to 37.5 % for the alternative method and the random acting players with a win rate or around 7.8%.

## Project Structure

- `ludopy/`: Core package containing the game logic and AI implementation.
  - `game.py`: Main game engine.
  - `Q_Learning.py`: Q-learning agent implementation.
  - `player.py`: Player logic and utilities.
  - `visualizer.py`: Tools for rendering the game board and creating videos.
  - `resources/`: Assets used by the visualizer (icons, etc.).
- `main.py`: Refactored script to train and evaluate the AI with CLI support.
- `test.py`: Basic game example and video generation.
- `ludopy/Data/`: Generated training data, Q-tables, and win rate plots.

## Installation

```sh
pip install -r requirements.txt
```

## Usage

### Training the AI
The `main.py` script now supports various command-line arguments for flexible training:

```bash
python3 main.py --games 1000 --eval-games 25 --plot
```

**Available Arguments:**
- `--lr`: Learning rate (default: 0.1)
- `--df`: Discount factor (default: 0.4)
- `--er`: Exploration rate (default: 0.05)
- `--games`: Total number of training games (default: 1000)
- `--eval-games`: Number of games per evaluation phase (default: 25)
- `--eval-after`: Start evaluation after N games (default: 800)
- `--plot`: If set, generates a win rate plot after training.
- `--verbose`: Enable detailed logging.

Training results, including timestamped Q-tables and win rate data, are automatically saved in `ludopy/Data/`.

### Basic Game Example
```python
import ludopy
import numpy as np

g = ludopy.Game()
there_is_a_winner = False

while not there_is_a_winner:
    (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = g.get_observation()

    if len(move_pieces):
        piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
    else:
        piece_to_move = -1

    _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)

g.save_hist_video("game_video.mp4")
```

## Documentation
For detailed documentation on the `ludopy` library, visit: [https://ludopy.readthedocs.io/en/latest/index.html](https://ludopy.readthedocs.io/en/latest/index.html)

## License
MIT
