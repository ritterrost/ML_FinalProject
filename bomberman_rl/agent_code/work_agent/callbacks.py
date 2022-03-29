import os
import sys
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from collections import deque
from .feature_functions import state_to_features, FEAT_DIM

# Game Parameter
ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
A_TO_NUM = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3, "WAIT": 4, "BOMB": 5}
A_IDX = np.arange(0, 6, 1, dtype="int")

#hyperparameters
MAX_DEPTH = None #better: less than 15
MAX_LEAF_NODES = 1000
MIN_SAMPLES_SPLIT = 5
N_ESTIMATORS = 20
HISTORY_SIZE = 10000
EPSILON_TRAIN = 0.2
RHO_TRAIN = 50


def setup(self):
    self.keep_training = False
    for arg in sys.argv:
        if arg=="--keep-training":
            self.keep_training = True
            break
    if self.train:
        self.forests = [
            RandomForestRegressor(
                n_estimators = N_ESTIMATORS, max_depth = MAX_DEPTH, bootstrap = True, max_leaf_nodes = MAX_LEAF_NODES, min_samples_split = MIN_SAMPLES_SPLIT
                )
                for a in ACTIONS
        ]
        self.feat_history = [deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE)]
        self.next_feat_history = [deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE)]
        self.reward_history = [deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE)]
        self.epsilon = EPSILON_TRAIN
    else:  
        self.epsilon = 0

    if (self.train and not self.keep_training) or not os.path.isfile("my-saved-model.pt"):
        # initial forest, create random model of right dimension and make fit
        X, y = make_regression(n_features=FEAT_DIM)
        for idx in A_IDX:
            self.forests[idx].fit(X, y)
    else:
        with open("my-saved-model.pt", "rb") as file:
            self.forests = pickle.load(file)


def policy_alt(self):
    prob = np.exp(self.Q_pred/self.rho)
    #sometimes probabilities are NaN
    mask = np.isnan(prob)
    prob[mask]=0

    prob = prob/np.sum(prob)
        
    return np.random.choice(np.arange(0,6,1), p=prob.reshape(6))


def policy(self, Qs):
    r = np.random.random(1)
    prob = np.ones(6)/6
    if r < self.epsilon:
        return np.random.choice(np.arange(0, 6, 1), p=prob)
    else:
        return np.argmax(Qs)

def Q(self, s, a):
    return self.forests[A_TO_NUM[a]].predict(s)

def Q_func(self, s):
    # takes features and index of action, returns predicted Q-value from forest of action, a-vector
    return np.array([Q(self, [s], a) for a in ACTIONS])


def act(self, game_state: dict):
    feat = state_to_features(game_state)
    Qs = Q_func(self, feat)
    a = policy(self, Qs)
    return ACTIONS[a]
