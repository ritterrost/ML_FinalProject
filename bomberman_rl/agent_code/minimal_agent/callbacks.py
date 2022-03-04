import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
RHO=1
#for dimension reduction
NN_OFFSET=np.array([[0,0],[0,1], [0,-1], [1,0], [-1,0]])
REL_ZONE_OFFSET=[]

#area with maximal distance of 4
for i1 in NN_OFFSET:
    for i2 in NN_OFFSET:
        for i3 in NN_OFFSET:
            for i4 in NN_OFFSET:
                REL_ZONE_OFFSET.append(i1+i2+i3+i4)
#only keep unique values
REL_ZONE_OFFSET=np.unique(REL_ZONE_OFFSET, axis=0)


def Q(self, game_state: dict):
    """
    ACTIONS-Vector of Q-value as function of game_state, np.array
    """
    return Q_vector

def decision_rule(self, game_state:dict, rho):
    """
    Softmax decision rule, rho is hyperparameter
    """
    Q_vector=Q(game_state)
    weight=np.exp(Q_vector/rho)
    #choose action according to probability distribution given by weight
    return np.random.choice(ACTIONS, p=weight/np.sum(weight))


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

def act(self, game_state: dict):
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    return decision_rule(game_state, RHO)

def state_to_features(game_state: dict):
    """
    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    #own position
    own_position=np.array(game_state['self'][-1])
    bombs_left=game_state['self'][-2]
    #indices of next neighbor and relevant field zone relative to own_position
    next_neighbor=own_position+NN_OFFSET
    relevant_zone=own_position+REL_ZONE_OFFSET
    
    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(game_state['field'])[relevant_zone]
    channels.append(game_state['others'])
    channels.append(game_state['bombs'])[0][relevant_zone]
    channels.append(game_state['bombs'])[1]
    channels.append(game_state['coins'])[relevant_zone]
    channels.append(game_state['explosion_map'])[next_neighbor]
    # and return them as a vector
    return np.array(channels).flatten
