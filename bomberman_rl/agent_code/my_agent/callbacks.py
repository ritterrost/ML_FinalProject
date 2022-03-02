from collections import deque
import os
import pickle
import random

import numpy as np

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "WAIT"]
SIGHT = 2
BOMBS_FEAT_SIZE = 12
COIN_COUNT = 50
A_TO_NUM = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3, "WAIT": 4, "BOMB": 5}
EPSILON = 0.9


def policy(q_res):
    r = np.random.random(1)
    if r > EPSILON:
        return np.random.randint(6)
    else:
        return np.argmax(q_res)


def Q_func(self, feat):
    result = []
    for beta in self.betas:
        result.append(np.dot(feat, beta))
    # self.logger.info(f"Q_func result: {result}")

    return result


def state_to_features(game_state):
    # Gather information about the game state
    arena = game_state["field"]
    explosion_map = game_state["explosion_map"]
    _, score, bombs_left, (x, y) = game_state["self"]
    bombs = game_state["bombs"]
    # not used yet
    others = [xy for (n, s, b, xy) in game_state["others"]]

    # turn bombs into constant sized feature
    bomb_xys = [[bomb_x, bomb_y, bomb_t] for ((bomb_x, bomb_y), bomb_t) in bombs]
    bomb_xys = np.array(bomb_xys).flatten()
    num_missing_bombs = BOMBS_FEAT_SIZE - bomb_xys.size
    bomb_feat = np.concatenate((bomb_xys, np.zeros(num_missing_bombs)))

    # turn coins into constant sized feature
    coins = game_state["coins"]
    coins = np.array(coins).flatten()
    num_missing_coins = 2 * COIN_COUNT - coins.size
    coins_feat = np.concatenate((coins, np.zeros(num_missing_coins)))

    # construct field feature
    xx, yy = np.clip(
        np.mgrid[x - SIGHT : x + SIGHT, y - SIGHT : y + SIGHT], 0, arena.shape[0] - 1
    )
    rel_field = arena[yy, xx].flatten().astype("int32")

    # construct explosion map feature
    rel_exp_map = explosion_map[yy, xx].flatten().astype("int32")

    # turn bombs left into int32 numpy array with
    bombs_left = (np.asarray(bombs_left)).astype("int32")

    # construct feature vector
    feature_vec = np.concatenate(
        (rel_field, rel_exp_map, np.array([x, y]), bomb_feat, coins_feat)
    )
    feature_vec = np.append(feature_vec, score)
    feature_vec = np.append(feature_vec, bombs_left)

    return feature_vec


def setup(self):

    self.betas = [[], [], [], [], [], []]
    self.feat_history = [[], [], [], [], [], []]
    self.target_history = [[], [], [], [], [], []]

    self.betas = [np.ones(148) for _ in enumerate(self.betas)]
    self.Q_pred = 1

    if self.train or not os.path.isfile("my-saved-model.pt"):
        # self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        # self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:

    feat = state_to_features(game_state)
    self.Q_pred = Q_func(self, feat)
    a = policy(self.Q_pred)

    self.logger.info(f"action a in act: {a}")
    return ACTIONS[a]
