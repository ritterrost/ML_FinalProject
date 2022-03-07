import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Hyper-parameters
# Dimension reduction
N_BOMBS = 1
N_COINS = 2


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
        self.logger.info("Setting up beta and rho from scratch.")
        self.betas = [np.random.rand(psi(self, None, True)[0]) for _ in range(6)]
        self.rho = 1  # initial rho can be low as initialisation is random
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(dict(betas=self.betas, rho=self.rho), file)

    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            pickledict = pickle.load(file)
        self.betas = pickledict['betas']
        self.rho = pickledict['rho']


def psi(self, game_state=None, give_length=False) -> np.array:
    if game_state is None:
        if give_length:
            pos = np.array([0,0])
        else:
            self.logger.debug(f"game_state is None or not specified. Return None")
            return None
    else:
        pos = game_state['self'][3]

    env = pos + np.array([[-1,-1], [-1,0], [-1,1], [0,-1], [0,1], 
                                 [1,-1], [1,0], [1,1]])
    explosion = pos + np.array([[-1,0], [1,0], [0,-1], [0,1]])
    
    l_env, l_exp, l_b, l_c = len(env), len(explosion), 2*N_BOMBS +1, 2*N_COINS
    if give_length:
        return l_env + l_exp + l_b + l_c, l_env, l_exp, l_b, l_c

    bombs = np.zeros(l_b)
    bombs[-1] = game_state['self'][2]
    for i, bomb in enumerate(game_state['bombs']):
        if i>=int(l_b/2):
            break
        v = pos - np.array(bomb[0])
        dist = max(.1, np.linalg.norm(v, ord=1)**2)
        bombs[2*i:2*(i+1)] = v / dist  # let vector scale with 1/||v||_1
#        self.logger.debug(f"bomb {i}: direction={v}, distance={dist}, entry={bombs[2*i:2*(i+1)]}")

    coins = np.zeros(l_c)
    for i, coin in enumerate(game_state['coins']):
        if i>=int(l_c/2):
            break
        v = pos - np.array(coin)
        dist = max(1, np.linalg.norm(v, ord=1)**2)
        coins[2*i:2*(i+1)] = v / dist  # let vector scale with 1/||v||_1
    
    reduced_state = np.concatenate([game_state['field'][tuple(env.T)],
                                    game_state['explosion_map'][tuple(explosion.T)],
                                    bombs, coins])
#    self.logger.debug(f"reduced state: {reduced_state}")
    return reduced_state


def Qs(self, game_state):
    return self.betas @ psi(self, game_state)
    

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    Q_hats = Qs(self, game_state)
    self.logger.debug(f"Q_hats = {Q_hats}")
    softmax = np.exp(Q_hats/self.rho)
    softmax[np.isinf(softmax)] = 1e4
#    self.logger.debug(f"softmax = {softmax}")
    
    return np.random.choice(ACTIONS, p=softmax/np.sum(softmax))


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

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

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
