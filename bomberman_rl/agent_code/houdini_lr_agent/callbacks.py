import os
import sys
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from collections import deque
from .feature_functions import state_to_features_bfs_2 as state_to_features, FEAT_DIM, A_TO_NUM

# Game Parameter
ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
A_IDX = np.arange(0, 6, 1, dtype="int")

#hyperparameters
MAX_DEPTH = 15 #better: less than 15
MAX_LEAF_NODES = int(1e5)
MIN_SAMPLES_SPLIT = 5
N_ESTIMATORS = 50
HISTORY_SIZE = int(1e6)
EPSILON_TRAIN = 0.1
RHO_TRAIN = 50


def setup(self):
    self.keep_training = False
    for arg in sys.argv:
        if arg=="--keep-training":
            self.keep_training = True
            break
    if self.train:
        self.model = [
            LinearRegression()
                for a in ACTIONS
            ]
        self.feat_history = [deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE)]
        self.next_feat_history = [deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE)]
        self.reward_history = [deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE)]
        self.epsilon = EPSILON_TRAIN
    else:  
        self.epsilon = 0

    if (self.train and not self.keep_training) or not os.path.isfile("current_model/my-saved-model.pt"):
        X, y = make_regression(n_features=FEAT_DIM)
        for idx in A_IDX:
            self.model[idx].fit(X, y)
    else:
        self.logger.info("Loading model from saved state.")
        with open("current_model/my-saved-model.pt", "rb") as file:
                self.model = pickle.load(file)
    if self.keep_training and os.path.isfile("current_model/feature_history.pt") \
    and os.path.isfile("current_model/next_feature_history.pt") \
    and os.path.isfile("current_model/reward_history.pt"):
        with open("current_model/feature_history.pt", "rb") as file:
            self.feat_history = pickle.load(file)
        with open("current_model/next_feature_history.pt", "rb") as file:
            self.next_feat_history = pickle.load(file)
        with open("current_model/reward_history.pt", "rb") as file:
            self.reward_history = pickle.load(file)


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
    return self.model[A_TO_NUM[a]].predict(s)

def Q_func(self, s):
    return np.array([Q(self, [s], a) for a in ACTIONS])


def act(self, game_state: dict):
    feat = state_to_features(game_state)
    Qs = Q_func(self, feat)
    a = policy(self, Qs)
    return ACTIONS[a]
