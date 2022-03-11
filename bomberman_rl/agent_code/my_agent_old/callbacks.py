from collections import deque
import os
import pickle
import random

import numpy as np

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "WAIT"]
SIGHT = 1
BOMBS_FEAT_SIZE = 12
COIN_COUNT = 50
COIN_K = 1
A_TO_NUM = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3, "WAIT": 4, "BOMB": 5}
EPSILON = 0.3


def policy(q_res):
    r = np.random.random(1)
    prob = np.array([0.2, 0.2, 0.2, 0.2, 0, 0.2])  # exclude bombs for now
    if r < EPSILON:
        return np.random.choice(np.arange(0, 6, 1), p=prob)
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
    coins = game_state["coins"]
    _, score, bombs_left, (x, y) = game_state["self"]
    bombs = game_state["bombs"]
    # not used yet
    others = [xy for (n, s, b, xy) in game_state["others"]]

    # turn bombs into constant sized feature
    bomb_xys = [
        [bomb_x - x, bomb_y - y, bomb_t] for ((bomb_x, bomb_y), bomb_t) in bombs
    ]
    bomb_xys = np.array(bomb_xys).flatten()
    num_missing_bombs = BOMBS_FEAT_SIZE - bomb_xys.size
    bomb_feat = np.concatenate((bomb_xys, np.zeros(num_missing_bombs)))

    ## turn coins into constant sized feature
    # coins = np.array(coins).flatten()
    # num_missing_coins = 2 * COIN_COUNT - coins.size
    # coins_feat = np.concatenate((coins, np.zeros(num_missing_coins)))

    # Alternatively only include the nearest K coins
    # only give direction of coin (for linear regression)
    coin_xy_rel = np.array([[coin_x - x, coin_y - y] for (coin_x, coin_y) in coins])
    if len(coins) < COIN_K:
        sorted_index = np.argsort(np.linalg.norm(coin_xy_rel, axis=1, ord=1))
        coins_feat = np.concatenate(
            (np.sign(coin_xy_rel[sorted_index]), np.zeros((COIN_K - len(coins), 2)))
        ).flatten()
    else:
        sorted_index = np.argsort(np.linalg.norm(coin_xy_rel, axis=1))[:COIN_K]
        coins_feat = np.sign(coin_xy_rel[sorted_index]).flatten()

    # construct field feature
    xx, yy = np.clip(
        np.mgrid[x - SIGHT : x + SIGHT, y - SIGHT : y + SIGHT], 0, arena.shape[0] - 1
    )

    ##take absolute value so walls and crates are treated equally
    # rel_obstacle_map = np.abs(arena[xx, yy]).flatten().astype("int32")

    # relative field
    rel_field = np.array(arena[xx, yy]).flatten().astype("int32")

    ##construct map of coins in SIGHT
    # rel_coin_map = np.zeros((len(yy), len(xx)))
    # coin_xy = [[coin_x,coin_y] for (coin_x,coin_y) in coins]
    # for coin in coin_xy:
    #    for xc in xx:
    #        for yc in yy:
    #            if (xc,yc) == coin:
    #                rel_coin_map[xc, yc] = 1
    # rel_coin_map = rel_coin_map.flatten().astype("int32")
    ##construct map of crates likewise
    # rel_crate_map = arena[xx, yy]>0
    # rel_crate_map = rel_crate_map.flatten().astype("int32")

    # construct explosion map feature
    rel_exp_map = explosion_map[xx, yy].flatten().astype("int32")

    # turn bombs left into int32 numpy array with
    bombs_left = (np.asarray(bombs_left)).astype("int32")

    # construct feature vector
    feature_vec = np.concatenate(
        #    (rel_field, rel_exp_map, np.array([x, y]), bomb_feat, coins_feat)
        (rel_field, coins_feat)
        #    (rel_obstacle_map, rel_coin_map, rel_crate_map)
    )
    # feature_vec = np.append(feature_vec, score)
    # feature_vec = np.append(feature_vec, bombs_left)

    return feature_vec


def setup(self):

    self.betas = [[], [], [], [], [], []]
    self.feat_history = [[], [], [], [], [], []]
    self.target_history = [[], [], [], [], [], []]

    # self.betas = [np.ones(58) for _ in enumerate(self.betas)]
    self.betas = [np.ones(6) for _ in enumerate(self.betas)]
    self.Q_pred = 1

    if self.train or not os.path.isfile("my-saved-model.pt"):
        # self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        # self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict):

    feat = state_to_features(game_state)
    self.Q_pred = Q_func(self, feat)
    a = policy(self.Q_pred)

    # self.logger.info(f"action a in act: {a}")
    return ACTIONS[a]
