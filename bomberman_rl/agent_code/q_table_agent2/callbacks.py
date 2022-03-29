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
EPSILON_TRAIN = 0.2

def Q_func(self, features):
    Q_values = []

    for Qs in self.Q_dicts:
        if tuple(features) in Qs.keys():
            Q_values.append(Qs[tuple(features)])
        else:
            Qs[tuple(features)] = 0
            Q_values.append(0)
    return Q_values


def setup(self):
    if self.train:
        self.Q_dicts = [{}, {}, {}, {}, {}, {}]
        self.epsilon = EPSILON_TRAIN
    else:  
        with open("my-saved-model.pt", "rb") as file:
            self.epsilon = 0
            self.Q_dicts = pickle.load(file)


def policy(self, Qs):
    r = np.random.random(1)
    prob = np.ones(6)/6
    if r < self.epsilon:
        return np.random.choice(np.arange(0, 6, 1), p=prob)
    else:
        return np.argmax(Qs)


def act(self, game_state: dict):
    feat = state_to_features(game_state)
    Qs = Q_func(self, feat)
    self.Q_values = Qs
    a = policy(self, Qs)
    return ACTIONS[a]
