import numpy as np
import ludopy
import argparse
import logging
import os
import matplotlib.pyplot as plt
from datetime import datetime


class LudoTrainer:
    def __init__(self, q_player=0, training_games=1000, eval_games=25, eval_after=800):
        """
        Initializes the LudoTrainer.

        :param q_player: The index of the player to be controlled by Q-Learning (0-3).
        :param training_games: Total number of games for training.
        :param eval_games: Number of games to play during each evaluation phase.
        :param eval_after: Start evaluating after this many training games.
        """
        self.q_player = q_player
        self.training_games = training_games
        self.eval_games = eval_games
        self.eval_after = eval_after

        self.logger = logging.getLogger(__name__)
        self.q_agent = ludopy.QLearning(self.q_player)

        # Win rate storage: games -> win_rate
        self.win_rates = {}

    def run_game(self, game, agent, is_training=True):
        """
        Runs a single game of Ludo.
        """
        agent.training = 1 if is_training else 0
        stop_while = False

        while not stop_while:
            (
                dice,
                move_pieces,
                player_pieces,
                enemy_pieces,
                player_is_a_winner,
                there_is_a_winner,
            ), player_i = game.get_observation()

            if player_i == self.q_player:
                piece_to_move = agent.update_q_table(
                    player_pieces, enemy_pieces, dice, game, there_is_a_winner
                )
                if there_is_a_winner:
                    stop_while = True
            else:
                if len(move_pieces):
                    piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
                else:
                    piece_to_move = -1

            _, _, _, _, _, there_is_a_winner = game.answer_observation(piece_to_move)
            if there_is_a_winner:
                stop_while = True

        winner = game.first_winner_was
        agent.reset_game()
        return winner

    def evaluate(self):
        """
        Evaluates the current performance of the agent.
        """
        wins = 0
        for _ in range(self.eval_games):
            game = ludopy.Game()
            winner = self.run_game(game, self.q_agent, is_training=False)
            if winner == self.q_player:
                wins += 1

        win_rate = wins / self.eval_games
        return win_rate

    def train(self, learning_rate=0.1, discount_factor=0.4, explore_rate=0.05):
        """
        Starts the training process.
        """
        self.q_agent.learning_rate = learning_rate
        self.q_agent.discount_factor = discount_factor
        self.q_agent.explore_rate = explore_rate

        self.logger.info(
            f"Starting training with LR={learning_rate}, DF={discount_factor}, ER={explore_rate}"
        )

        for k in range(self.training_games):
            if k % 100 == 0:
                self.logger.info(f"Training game {k}/{self.training_games}...")

            game = ludopy.Game()
            self.run_game(game, self.q_agent, is_training=True)

            if k > self.eval_after:
                win_rate = self.evaluate()
                self.win_rates[k] = win_rate
                self.logger.info(f"Game {k}: Win Rate = {win_rate:.2f}")

        self.save_results()

    def save_results(self):
        """
        Saves the Q-table and win rate data.
        """
        os.makedirs("ludopy/Data", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        q_table_path = f"ludopy/Data/Q_table_{timestamp}.npy"
        win_rate_path = f"ludopy/Data/Win_rates_{timestamp}.npy"

        self.q_agent.save_Q_table(q_table_path)
        np.save(win_rate_path, self.win_rates)

        self.logger.info(f"Q-table saved to {q_table_path}")
        self.logger.info(f"Win rates saved to {win_rate_path}")

    def plot_win_rates(self):
        """
        Plots the win rate over time.
        """
        if not self.win_rates:
            self.logger.warning("No win rate data to plot.")
            return

        sorted_games = sorted(self.win_rates.keys())
        rates = [self.win_rates[g] for g in sorted_games]

        plt.figure(figsize=(10, 6))
        plt.plot(sorted_games, rates, marker="o", linestyle="-", color="b")
        plt.title("Win Rate Improvement Over Training")
        plt.xlabel("Game Number")
        plt.ylabel("Win Rate")
        plt.grid(True)

        plot_path = "ludopy/Data/win_rate_plot.png"
        plt.savefig(plot_path)
        self.logger.info(f"Win rate plot saved to {plot_path}")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Ludo AI using Q-Learning.")
    parser.add_argument(
        "--lr", type=float, default=0.1, help="Learning rate (default: 0.1)"
    )
    parser.add_argument(
        "--df", type=float, default=0.4, help="Discount factor (default: 0.4)"
    )
    parser.add_argument(
        "--er", type=float, default=0.05, help="Exploration rate (default: 0.05)"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1000,
        help="Number of training games (default: 1000)",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=25,
        help="Number of evaluation games (default: 25)",
    )
    parser.add_argument(
        "--eval-after",
        type=int,
        default=800,
        help="Start evaluation after N games (default: 800)",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Plot win rates after training"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    trainer = LudoTrainer(
        training_games=args.games,
        eval_games=args.eval_games,
        eval_after=args.eval_after,
    )

    trainer.train(learning_rate=args.lr, discount_factor=args.df, explore_rate=args.er)

    if args.plot:
        trainer.plot_win_rates()
