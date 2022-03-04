from collections import deque
import os
import pickle
import random

import numpy as np

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "WAIT"]
SIGHT = 1
BOMBS_FEAT_SIZE = 12
COIN_COUNT = 50
A_TO_NUM = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3, "WAIT": 4, "BOMB": 5}
EPSILON = 0.9
BATCH_SIZE = 10
BUFFER_SIZE = 200
FEAT_SIZE = 9


class Buffer:
    def __init__(self):
        self.storage = [[], [], [], [], [], []]

    def append(self, idx, val):
        self.storage[idx].append(val)
        if len(self.storage[idx]) > BUFFER_SIZE:
            del self.storage[idx][0]
        # print("Buffer: ", self.storage)

    def get_by_list(self, idx, list):
        return np.array(self.storage[idx])[list]

    def get_by_idx(self, idx):
        return np.array(self.storage[idx])

    def get_storage(self):
        return self.storage

    def get_storage_size(self, idx):
        return len(self.storage[idx])

    def get_batch(self, idx):
        selection_size = self.get_storage_size(idx)
        selection_mask = np.random.permutation(np.arange(selection_size))[0:BATCH_SIZE]
        return self.get_by_list(idx, selection_mask)


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

    new_vec = np.concatenate((coins_feat[0:5], rel_field))
    print(new_vec)

    return new_vec


def setup(self):

    self.betas = [[], [], [], [], [], []]
    self.feat_history = Buffer()
    self.target_history = Buffer()

    self.betas = [np.random.random(FEAT_SIZE) for _ in enumerate(self.betas)]
    self.Q_pred = 0

    if self.train or not os.path.isfile("my-saved-model.pt"):
        # self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        # self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.betas = pickle.load(file)


def act(self, game_state: dict) -> str:
    feat = state_to_features(game_state)
    self.Q_pred = Q_func(self, feat)
    # print("self.Q_pred: ", self.Q_pred)
    # print("self.betas: ", self.betas)
    a = policy(self.Q_pred)
    return ACTIONS[a]
