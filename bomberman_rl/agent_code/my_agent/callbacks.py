import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from collections import deque
from agent_code.my_agent import feature_functions

# Game Parameter
ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
A_TO_NUM = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3, "WAIT": 4, "BOMB": 5}
A_IDX = np.arange(0, 6, 1, dtype="int")
#FEAT_DIM = 16
FEAT_DIM = 12#+1
#hyperparameters
MAX_DEPTH = None #better: less than 15
MAX_LEAF_NODES = 100000
MIN_SAMPLES_SPLIT = 5
N_ESTIMATORS = 50
HISTORY_SIZE = 10000000
EPSILON_TRAIN = 0.2
RHO_TRAIN = 50


def policy_alt(self):
    prob = np.exp(self.Q_pred/self.rho)
    #sometimes probabilities are NaN
    mask = np.isnan(prob)
    prob[mask]=0

    prob = prob/np.sum(prob)
        
    return np.random.choice(np.arange(0,6,1), p=prob.reshape(6))


def policy(self):
    r = np.random.random(1)
    prob = np.ones(6)/6  # include bombs but not waiting
    if r < self.epsilon:
        return np.random.choice(np.arange(0, 6, 1), p=prob)
    else:
        return np.argmax(self.Q_pred)


def Q_func(self, feat):
    # takes features and index of action, returns predicted Q-value from forest of action, a-vector
    model = self.forests
    return np.array([model[idx].predict([feat]) for idx in A_IDX])

def setup(self):

    self.forests = [
        RandomForestRegressor(
            n_estimators = N_ESTIMATORS, max_depth = MAX_DEPTH, bootstrap = True, max_leaf_nodes = MAX_LEAF_NODES, min_samples_split = MIN_SAMPLES_SPLIT
            )
            for a in ACTIONS
    ]

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
        self.epsilon = 0

    if not os.path.isfile("current_model/my-saved-model.pt") or not os.path.isfile("current_model/feature_history.pt")\
    or not os.path.isfile("current_model/next_feature_history.pt") or not os.path.isfile("current_model/reward_history.pt"):
        # self.logger.info("Setting up model from scratch.")
        #initial forest, create random model of right dimension and make fit
        X, y = make_regression(n_features = FEAT_DIM, random_state=0)
        for idx in A_IDX:
            self.forests[idx].fit(X, y)
    else:
        # self.logger.info("Loading model from saved state.")
        with open("current_model/my-saved-model.pt", "rb") as file:
            self.forests = pickle.load(file)
        with open("current_model/feature_history.pt", "rb") as file:
            self.feat_history = pickle.load(file)
        with open("current_model/next_feature_history.pt", "rb") as file:
            self.next_feat_history = pickle.load(file)
        with open("current_model/reward_history.pt", "rb") as file:
            self.reward_history = pickle.load(file)

def act(self, game_state: dict):
    feat = feature_functions.state_to_features_bfs_2(self, game_state)
    self.Q_pred = Q_func(self, feat)
    if self.train:
        #a = policy_alt(self)
        a = policy(self)
    else:
        a = policy(self)
    # self.logger.info(f"action a in act: {a}")
    return ACTIONS[a]
