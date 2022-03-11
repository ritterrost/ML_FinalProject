import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

from agent_code.my_agent import feature_functions

# Game Parameter
ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
A_TO_NUM = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3, "WAIT": 4, "BOMB": 5}
A_IDX = np.arange(0, 6, 1, dtype="int")

# Hyperparameters
MAX_DEPTH = 10
N_ESTIMATORS = 20
# RANGE = 6
EPSILON_TRAIN = 0.2
EPSILON = 0.05
RHO_TRAIN = 1
RHO = 0.1
FEAT_DIM = 7


def policy_alt(self):
    prob = np.exp(self.Q_pred / self.rho)
    prob = prob / np.sum(prob)
    return np.random.choice(np.arange(6), p=prob.reshape(6))


def policy(self):
    r = np.random.random(1)
    prob = np.array([0.2, 0.2, 0.2, 0.2, 0, 0.2])  # include bombs for now
    if r < self.epsilon:
        return np.random.choice(np.arange(0, 6, 1), p=prob)
    else:
        return np.argmax(self.Q_pred)


def Q_func(self, feat):
    # takes features and index of action, returns predicted Q-value from forest of action, a-vector
    model = self.forests
    return np.array([model[idx].predict([feat]) for idx in A_IDX])


def setup(self):
    if self.train:
        self.forests = [
            RandomForestRegressor(
                n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, bootstrap=True
            )
            for a in ACTIONS
        ]
        self.feat_history = [[], [], [], [], [], []]
        self.target_history = [[], [], [], [], [], []]
        self.Q_pred = np.ones(50)
        self.epsilon = EPSILON_TRAIN
        self.rho = RHO_TRAIN
    else:
        self.epsilon = EPSILON
        self.rho = RHO
    if self.train or not os.path.isfile("my-saved-model.pt"):
        # initial forest, create random model of right dimension and make fit
        X, y = make_regression(n_features=FEAT_DIM, random_state=0)
        for idx in A_IDX:
            self.forests[idx].fit(X, y)
    else:
        # self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.forests = pickle.load(file)


def act(self, game_state: dict):
    feat = feature_functions.state_to_features_bfs_2(game_state)
    self.Q_pred = Q_func(self, feat)
    a = policy(self)
    # self.logger.info(f"action a in act: {a}")
    return ACTIONS[a]
