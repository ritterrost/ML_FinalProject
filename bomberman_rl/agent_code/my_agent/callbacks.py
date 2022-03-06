from collections import deque
import os
import pickle
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

import numpy as np

#for convenience
ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
A_TO_NUM = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3, "WAIT": 4, "BOMB": 5}
A_IDX = np.arange(0,6,1,dtype='int')
BOMBS_FEAT_SIZE = 12
FEAT_DIM = 10 + BOMBS_FEAT_SIZE

#hyperparameters
MAX_DEPTH = 10
N_ESTIMATORS = 10
SIGHT = 1
COIN_K = 1
RANGE = 6
EPSILON_TRAIN = 0.1
EPSILON = 0.01

def policy(self):
    r = np.random.random(1)
    prob = np.array([0.2,0.2,0.2,0.2,0,0.2]) #exclude bombs for now
    if r < self.epsilon:
        return np.random.choice(np.arange(0,6,1), p=prob)
    else:
        return np.argmax(self.Q_pred)


def Q_func(self, feat):
    #takes features and index of action, returns predicted Q-value from forest of action, a-vector
    regr = self.forests
    return np.array([regr[idx].predict([feat]) for idx in A_IDX])


def state_to_features(game_state):
    # Gather information about the game state
    arena = game_state["field"]
    explosion_map = game_state["explosion_map"]
    coins = game_state["coins"]
    _, score, bombs_left, (x, y) = game_state["self"]
    bombs = game_state["bombs"]
    # not used yet
    others = [xy for (n, s, b, xy) in game_state["others"]]

    #surrounding walls
    wall_above = arena[x, y+1] == 1
    wall_below = arena[x, y-1] == 1
    wall_right = arena[x+1, y] == 1
    wall_left = arena[x-1, y] == 1

    # turn bombs into constant sized feature
    # only consider bombs in 'range'
    bomb_xys = [[bomb_x-x, bomb_y-y, bomb_t] for ((bomb_x, bomb_y), bomb_t) in bombs]
    if len(bomb_xys) > 1:
        bomb_mask = np.logical_and(np.abs(bomb_xys[:,0])<RANGE, np.abs(bomb_xys[:,1])<RANGE)
        bomb_xys = np.array(bomb_xys[:,bomb_mask]).flatten()
        num_missing_bombs = BOMBS_FEAT_SIZE - bomb_xys.size
        bomb_feat = np.concatenate((bomb_xys, np.zeros(num_missing_bombs)))

    else:
        bomb_feat = np.zeros(BOMBS_FEAT_SIZE)

    ## turn coins into constant sized feature
    #coins = np.array(coins).flatten()
    #num_missing_coins = 2 * COIN_COUNT - coins.size
    #coins_feat = np.concatenate((coins, np.zeros(num_missing_coins)))

    #Alternatively only include the nearest K coins
    #only give direction of coin (for simplicity)
    NO_COINS = False
    coin_xy_rel = np.array([[coin_x-x,coin_y-y] for (coin_x,coin_y) in coins])
    if len(coins)<COIN_K:
        if len(coins)==1:
            coins_feat = np.concatenate((np.sign(coin_xy_rel), np.zeros((COIN_K-len(coins),2))))
        if len(coins)==0:
            NO_COINS = True
            coins_feat = np.zeros((COIN_K-len(coins),2)).flatten()
        else:
            sorted_index = np.argsort(np.linalg.norm(coin_xy_rel, axis=1, ord=1))
            coins_feat = np.concatenate((np.sign(coin_xy_rel[sorted_index]), np.zeros((COIN_K-len(coins),2))))
    else:
        sorted_index = np.argsort(np.linalg.norm(coin_xy_rel, axis=1))[:COIN_K]
        coins_feat = np.sign(coin_xy_rel[sorted_index])
    
    #exclude directions with walls
    if len(coins)>0:
        if wall_above:
            mask = coins_feat[1,:] == 1
            coins_feat[1, mask] = 0
        if wall_below:
            mask = coins_feat[1,:] == -1
            coins_feat[1, mask] = 0
        if wall_right:
            mask = coins_feat[0,:] == 1
            coins_feat[0, mask] = 0
        if wall_left:
            mask = coins_feat[0,:] == -1
            coins_feat[0, mask] = 0

    if not NO_COINS and np.all(coins_feat == 0):
        #choose random direction without wall
        available_directions = []
        if not wall_above: available_directions.append([0,1])
        if not wall_below: available_directions.append([0,-1])
        if not wall_right: available_directions.append([1,0])
        if not wall_left: available_directions.append([-1,0])
        if len(available_directions)>0:
            chosen_direction = np.random.randint(len(available_directions))
            coins_feat[0,0], coins_feat[0,1] = np.array(available_directions)[chosen_direction, 0],np.array(available_directions)[chosen_direction, 1]
            
    coins_feat=coins_feat.flatten()
    # construct field feature
    xx, yy = np.clip(
        np.mgrid[x - SIGHT : x + SIGHT, y - SIGHT : y + SIGHT], 0, arena.shape[0] - 1
    )

    ##take absolute value so walls and crates are treated equally
    #rel_obstacle_map = np.abs(arena[xx, yy]).flatten().astype("int32")

    #relative field
    rel_field = np.array(arena[xx,yy]).flatten().astype("int32")

    ##construct map of coins in SIGHT
    #rel_coin_map = np.zeros((len(yy), len(xx)))
    #coin_xy = [[coin_x,coin_y] for (coin_x,coin_y) in coins]
    #for coin in coin_xy:
    #    for xc in xx:
    #        for yc in yy:
    #            if (xc,yc) == coin:
    #                rel_coin_map[xc, yc] = 1
    #rel_coin_map = rel_coin_map.flatten().astype("int32")
    ##construct map of crates likewise
    #rel_crate_map = arena[xx, yy]>0
    #rel_crate_map = rel_crate_map.flatten().astype("int32")

    # construct explosion map feature
    rel_exp_map = explosion_map[xx, yy].flatten().astype("int32")

    # turn bombs left into int32 numpy array with
    bombs_left = (np.asarray(bombs_left)).astype("int32")

    # construct feature vector
    feature_vec = np.concatenate(
    #    (rel_field, rel_exp_map, np.array([x, y]), bomb_feat, coins_feat)
        (rel_field, coins_feat, rel_exp_map, bomb_feat)
    #    (rel_obstacle_map, rel_coin_map, rel_crate_map)
    )
    #feature_vec = np.append(feature_vec, score)
    #feature_vec = np.append(feature_vec, bombs_left)

    return feature_vec


def setup(self):

    self.forests = [RandomForestRegressor(n_estimators = N_ESTIMATORS, max_depth = MAX_DEPTH, bootstrap = True) for a in ACTIONS]
    self.feat_history = [[], [], [], [], [], []]
    self.target_history = [[], [], [], [], [], []]

    self.Q_pred = np.ones(FEAT_DIM)
    if self.train:
        self.epsilon = EPSILON_TRAIN
    else:  
        self.epsilon = EPSILON
    if self.train or not os.path.isfile("my-saved-model.pt"):
        # self.logger.info("Setting up model from scratch.")
        #initial forest, create random model of right dimension and make fit
        X, y = make_regression(n_features = FEAT_DIM, random_state=0)
        for idx in A_IDX:
            self.forests[idx].fit(X, y)
    else:
        # self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.forests = pickle.load(file)


def act(self, game_state: dict):

    feat = state_to_features(game_state)
    self.Q_pred = Q_func(self, feat)
    a = policy(self)

    #self.logger.info(f"action a in act: {a}")
    return ACTIONS[a]
