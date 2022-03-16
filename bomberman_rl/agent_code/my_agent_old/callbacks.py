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
BOMBS_FEAT_SIZE = 2
FEAT_DIM = 11 + BOMBS_FEAT_SIZE

#hyperparameters
MAX_DEPTH = 10
N_ESTIMATORS = 10
SIGHT = 1
HISTORY_SIZE = 100000
#RANGE = 6
EPSILON_TRAIN = 0.2
EPSILON = 0.01
RHO_TRAIN = 1
RHO = 0.01

def policy_alt(self):
    prob = np.exp(self.Q_pred/self.rho)
    #sometimes probabilities are NaN
    mask = np.isnan(prob)
    prob[mask]=0

    prob = prob/np.sum(prob)
    return np.random.choice(np.arange(0,6,1), p=prob.reshape(6))

def policy(self):
    r = np.random.random(1)
    prob = np.array([0.2,0.2,0.2,0.2,0,0.2]) #include bombs for now
    if r < self.epsilon:
        return np.random.choice(np.arange(0,6,1), p=prob)
    else:
        return np.argmax(self.Q_pred)


def Q_func(self, feat):
    #takes features and index of action, returns predicted Q-value from forest of action, a-vector
    regr = self.forests
    return np.array([regr[idx].predict([feat]) for idx in A_IDX])


def state_to_features(self, game_state):
    
    #find closest coin, ... first
    find(self, game_state)
    # Gather information about the game state
    arena = game_state["field"]
    explosion_map = game_state["explosion_map"]
    coins = game_state["coins"]
    _, score, bombs_left, (x, y) = game_state["self"]
    bombs = game_state["bombs"]
    others = [xy for (n, s, b, xy) in game_state["others"]]
    player_pos = np.array([x,y])

    # construct field feature, somethings wrong here??
    ################################check if this is ok
    xx, yy = np.clip(
        np.mgrid[x - SIGHT : x + SIGHT, y - SIGHT : y + SIGHT], 0, arena.shape[0] - 1
    )

    #relative field
    rel_field = np.array(arena[xx,yy]).flatten().astype("int32")

    # construct explosion map feature
    rel_exp_map = explosion_map[xx, yy].flatten().astype("int32")

    unpassable = np.logical_or(rel_field != 0, rel_exp_map)

    ##surrounding walls
    wall_above = arena[x, y+1] == -1 or explosion_map[x, y+1] != 0
    wall_below = arena[x, y-1] == -1 or explosion_map[x, y-1] != 0
    wall_right = arena[x+1, y] == -1 or explosion_map[x+1, y] != 0
    wall_left = arena[x-1, y] == -1 or explosion_map[x-1, y] != 0

    "Coin, Barrel, Player features"
    "only give direction of closest coin/barrel/player (for simplicity)"
    "If path is blocked by wall give random lateral direction"

    coins_feat = self.closest_coin-player_pos
    barrel_feat = self.closest_barrel-player_pos
    enemy_feat = self.closest_player-player_pos
    closest_bomb = self.closest_bomb
    if np.all(closest_bomb == 0):
        bomb_feat = closest_bomb.flatten()
    else:
        bomb_feat = np.array([0,0])

    #exclude directions with walls
    for feature in [coins_feat, barrel_feat, enemy_feat]:
        feature = np.sign(feature)
        if len(feature)>0:
            if wall_above and feature[1] > 0:
                feature[1] = 0
            if wall_below and feature[1] < 0:
                feature[1] = 0
            if wall_right and feature[0] > 0:
                feature[0] = 0
            if wall_left and feature[0] < 0:
                feature[0] = 0

        if len(feature)>0 and np.all(feature == 0):
            #choose random direction without wall
            available_directions = []
            if not wall_above: available_directions.append([0,1])
            if not wall_below: available_directions.append([0,-1])
            if not wall_right: available_directions.append([1,0])
            if not wall_left: available_directions.append([-1,0])
            if len(available_directions)>0:
                chosen_direction = np.random.randint(len(available_directions))
                feature[0], feature[1] = np.array(available_directions)[chosen_direction, 0],np.array(available_directions)[chosen_direction, 1]
            
        feature = feature.flatten()

    # construct feature vector
    feature_vec = np.concatenate(
        (coins_feat, barrel_feat, enemy_feat, bomb_feat, unpassable, np.array([bombs_left])) #unpassable seems to be necessary, bombs left sensible when using crates (note tree for placing bomb will heavily use this)
    )

    return feature_vec


def setup(self):

    self.forests = [RandomForestRegressor(n_estimators = N_ESTIMATORS, max_depth = MAX_DEPTH, bootstrap = True) for a in ACTIONS]
    self.feat_history = [deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE)]
    self.target_history = [deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE)]
    self.next_feat_history = [deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE)]
    self.reward_history = [deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE)]
    self.closest_coin = 'None'
    self.closest_barrel = 'None'
    self.closest_player = 'None'

    self.Q_pred = np.ones(FEAT_DIM)
    if self.train:
        self.epsilon = EPSILON_TRAIN
        self.rho = RHO_TRAIN
    else:  
        self.epsilon = EPSILON
        self.rho = RHO

    if not os.path.isfile("my-saved-model.pt") or not os.path.isfile("feature_history.pt")\
    or not os.path.isfile("next_feature_history.pt") or not os.path.isfile("reward_history.pt"):
        # self.logger.info("Setting up model from scratch.")
        #initial forest, create random model of right dimension and make fit
        X, y = make_regression(n_features = FEAT_DIM, random_state=0)
        for idx in A_IDX:
            self.forests[idx].fit(X, y)
    else:
        # self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.forests = pickle.load(file)
        with open("feature_history.pt", "rb") as file:
            self.feat_history = pickle.load(file)
        with open("next_feature_history.pt", "rb") as file:
            self.next_feat_history = pickle.load(file)
        with open("reward_history.pt", "rb") as file:
            self.reward_history = pickle.load(file)


def act(self, game_state: dict):
    
    feat = state_to_features(self, game_state)
    self.Q_pred = Q_func(self, feat)
    a = policy_alt(self)

    #self.logger.info(f"action a in act: {a}")
    return ACTIONS[a]


def find(self, game_state):
    "finds closest coin, barrel and player(, bomb) to target and saves their position"
    arena = game_state["field"]
    explosion_map = game_state["explosion_map"]
    coins = game_state["coins"]
    _, score, bombs_left, (x, y) = game_state["self"]
    others = np.array([xy for (n, s, b, xy) in game_state["others"]])
    player_pos = np.array([x,y])
    bombs = game_state["bombs"]

    if game_state is not None:
        self.old_closest_coin = self.closest_coin
        self.old_closest_barrel = self.closest_barrel
        self.old_closest_player = self.closest_player

    #coins
    coin_xy = np.array([[coin_x,coin_y] for (coin_x,coin_y) in coins])
    if len(coins)>0:
        closest_index = np.argmin(np.linalg.norm(coin_xy - player_pos, axis=1))
        self.closest_coin = coin_xy[closest_index]

    else:
        self.closest_coin = player_pos

    #barrels
    barrel_xy = np.argwhere(arena>0)
    if len(barrel_xy)>0:
        closest_index = np.argmin(np.linalg.norm(barrel_xy - player_pos, axis=1))
        self.closest_barrel = barrel_xy[closest_index]

    else:
        self.closest_barrel = player_pos

    #players
    if len(others)>0:
        closest_index = np.argmin(np.linalg.norm(others - player_pos, axis=1))
        self.closest_player = others[closest_index]

    else:
        self.closest_player = player_pos

    #bombs
    if len(bombs)>0:
        bomb_xy = np.array([[x,y] for [(x,y),t] in bombs])
        closest_index = np.argmin(np.linalg.norm(bomb_xy - player_pos, axis=1))
        self.closest_bomb = bomb_xy - player_pos
       
    else:
        self.closest_bomb = np.array([0,0])
