import os
import pickle
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor


ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
COIN_COUNT = 50
N_COINS = 1
FEAT_SIZE = 4

# from the blog post
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 1.0
EXPLORATION_DECAY = 0.96


def policy(self, Q_values):
    if np.random.rand() < self.exploration_rate or not self.isFit:
        return np.random.randint(4)
    else:
        return np.argmax(Q_values)


def tabular_Q_func(self, features):
    Q_values = []

    for Qs in self.Q_dicts:
        if hash(str(features)) in Qs.keys():
            Q_values.append(Qs[hash(str(features))])
        else:
            Qs[hash(str(features))] = 0
            Q_values.append(0)
    return Q_values


def lr_Q_func(self, feat):
    q_values = []
    for beta in self.weights:
        q_values.append(np.dot(feat, beta))
    return q_values


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    if self.train or not os.path.isfile("my-saved-model.pt"):

        self.Q_values = []

        # for tabular Q learning
        self.Q_dicts = [{}, {}, {}, {}, {}, {}]

        weights = [[], [], [], [], [], []]
        self.weights = [np.random.random(FEAT_SIZE) for _ in enumerate(weights)]
        self.model = MultiOutputRegressor(LGBMRegressor(n_estimators=6, n_jobs=1))
        self.isFit = False

        self.exploration_rate = EXPLORATION_MAX
    else:
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    features = state_to_features(game_state)
    Q_values = lr_Q_func(self, features)
    self.Q_values = Q_values
    # print('features: ', features)
    # print('Q_values: ', Q_values)
    # print('weights: ', self.weights)

    return ACTIONS[policy(self, Q_values)]


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # self feat
    _, score, bombs_left, (me_x, me_y) = game_state["self"]
    pos = np.asarray((me_y, me_x))

    # coins feat
    coins = np.asarray(game_state["coins"])

    # for all coins
    # coins = np.array(coins).flatten()
    # num_missing_coins = 2 * COIN_COUNT - coins.size
    # coins_feat = np.concatenate((coins, np.zeros(num_missing_coins)))

    # for single closest coin
    closest_coin_idx = np.argmin(np.linalg.norm(pos - coins, axis=1))

    return np.concatenate([pos, coins[closest_coin_idx]])
