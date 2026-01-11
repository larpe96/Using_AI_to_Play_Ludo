# LUDOpy - AI Ludo Game Player

This project implements a Ludo game environment in Python and develops an Artificial Intelligence (AI) player using Q-learning. It is built on the [LUDOpy](https://github.com/SimonLBSoerensen/LUDOpy) library.

## Abstract

An artificial intelligent (AI) method to play the game ludo is developed. The method used is called Q-learning and it uses a special representation of the game so that the method is compatible. The representation is composed of states that the player piece can be in and actions which can be performed by the player pieces. For the method to understand and be able to win the game training is need to help the Q-learning player learn the game. A reward function is also used when training to determine if the Q-learning player is making a good or bad decision. 

The training is done against 3 random acting players and it achieves an average win rate of 59.5%. The algorithm is evaluated in two different test, against an alternative AI method and against the alternative method and 2 random acting players. In the first test the presented method achieves a win rate of 56.43% compared to 43.56% of the alternative method. In the second test it achieves a win rate of 46.89% compared to 37.5 % with the alternative method and the random acting players with a win rate or around 7.8%.

## Project Structure

- `ludopy/`: Core package containing the game logic and AI implementation.
  - `game.py`: Main game engine.
  - `Q_Learning.py`: Q-learning agent implementation.
  - `main_ludo.py`: Script to train and evaluate the AI.
  - `player.py`: Player logic and utilities.
  - `visualizer.py`: Tools for rendering the game board and creating videos.
  - `resources/`: Assets used by the visualizer (icons, etc.).
- `Data/`: Generated training data and Q-tables.

## Installation

```sh
pip install -r requirements.txt
```

## Usage

### Training the AI
To start the training process, run:
```bash
python3 ludopy/main_ludo.py
```

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
