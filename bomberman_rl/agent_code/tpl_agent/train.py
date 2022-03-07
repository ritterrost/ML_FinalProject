from collections import namedtuple, deque

import pickle
from typing import List
import numpy as np

import events as e
from .callbacks import psi, Qs

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 5000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
GAMMA = .5
ALPHA = 1
BATCH_SIZE = 5
INITIAL_TEMP = 100
FINAL_TEMP = 1
N_ROUNDS = 1000

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
WALKED_TO_COIN = 'WALKED_TO_COIN'
WALKED_FROM_COIN = 'WALKED_FROM_COIN'
WALKED_TO_BOMB = 'WALKED_TO_BOMB'
WALKED_FROM_BOMB = 'WALKED_FROM_BOMB'


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS_TO_VECTOR = dict(zip(ACTIONS, np.array([[0,-1], [1,0], [0,1], [-1,0], [0,0], [0,0]])))
#VECTOR_TO_ACTIONS = {v: k for k, v in ACTIONS_TO_VECTOR.items()}

O2 = {
      'e': np.array([[ 1, 0],
                     [ 0, 1]]),
      'pi/2': np.array([[ 0,-1],
                        [ 1, 0]]),
      'pi': np.array([[-1, 0],
                      [ 0,-1]]),
      '3pi/2': np.array([[ 0, 1],
                         [-1, 0]]),
      'sigma_x': np.array([[ 1, 0],
                           [ 0,-1]]),
      'sigma_y': np.array([[-1, 0],
                           [ 0, 1]]),
      'sigma_d1': np.array([[ 0, 1],
                            [ 1, 0]]),
      'sigma_d2': np.array([[ 0,-1],
                            [-1, 0]])
      }

O2_func = {
    'e': lambda m: m,
    'pi/2': np.rot90,
#    'pi': np.rot180,
#    '3pi/2': np.rot270,
    'sigma_x': np.flipud,
    'sigma_y': np.fliplr,
    'sigma_d1': lambda m: np.rot90(np.fliplr(m)),
    'sigma_d2': lambda m: np.rot90(np.flipud(m))
    }

origin = np.array([8,8])


def symmetry(game_state):
    if game_state is None:
        return None
    states = {}
    for g in O2.keys():
        sym_state = game_state.copy()
        sym_state['field'] = O2_func[g](sym_state['field'])
        sym_state['explosion_map'] = O2_func[g](sym_state['explosion_map'])
        sym_state['self'][3] = (sym_state['self'][3] - origin) @ O2[g].T + origin
        for i, other in enumerate(sym_state['bombs']):
            sym_state['others'][i][3] = (other[0] - origin) @ O2[g].T + origin
        for i, bomb in enumerate(sym_state['bombs']):
            sym_state['bombs'][i][0] = (bomb[0] - origin) @ O2[g].T + origin
        
        


def response(self, new_game_state, reward):
    return reward + GAMMA * np.max(Qs(self, new_game_state))

    
def gradient_descent(self):
    for i, a in enumerate(ACTIONS):
        action_mask = self.self_actions==a
        action_length = int(np.sum(action_mask))
        if action_length==0:
            self.logger.debug(f"No gradient descent possible for {a}")
            continue
        idx = np.random.choice(np.arange(action_length), BATCH_SIZE)
        
        Psi = self.old_game_states[action_mask][idx]
        Y = self.responses[action_mask][idx]
        new_beta = self.betas[i] + ALPHA / BATCH_SIZE * np.sum(Psi * (Y - np.dot(Psi, self.betas[i]))[:,None], axis=0)
        self.betas[i] = new_beta / np.linalg.norm(new_beta)


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.rho = INITIAL_TEMP
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.counter = 0
    self.old_game_states = np.empty((TRANSITION_HISTORY_SIZE, psi(self, None, True)[0]))
    self.self_actions = np.empty(TRANSITION_HISTORY_SIZE).astype(str)
    self.responses = np.empty(TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    
    
    if not old_game_state is None:
        old_pos = old_game_state['self'][3]
        new_pos = new_game_state['self'][3]
        l, l_env, l_exp, l_b, l_c = psi(self, None, give_length=True)
        for i, bomb in enumerate(old_game_state['bombs']):
            if i>=int(l_b/2):
                break
            old_dist = np.linalg.norm(np.array(bomb[0])-old_pos, ord=1)
            if old_dist<=3:
                if np.linalg.norm(np.array(bomb[0])-new_pos, ord=1) > old_dist:
                    events.append(WALKED_FROM_BOMB)
                else:
                    events.append(WALKED_TO_BOMB)
        if not e.COIN_FOUND in events:
            for i, coin in old_game_state['coins']:
                if i>=int(l_c/2):
                    break
                if np.linalg.norm(np.array(coin)-old_pos, ord=1)>np.linalg.norm(np.array(coin)-new_pos, ord=1):
                    events.append(WALKED_TO_COIN)
                else:
                    events.append(WALKED_FROM_COIN)

    # state_to_features is defined in callbacks.py
    reward = reward_from_events(self, events)
    red_game_state = psi(self, old_game_state)
    i = np.random.randint(TRANSITION_HISTORY_SIZE)
#    self.transitions.append(Transition(red_game_state, self_action, psi(self, new_game_state), reward))
    if not old_game_state is None:
        self.old_game_states[i] = red_game_state
        self.self_actions[i] = self_action
        self.responses[i] = response(self, new_game_state, reward)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    reward = reward_from_events(self, events)
    red_game_state = psi(self, last_game_state)
    i = np.random.randint(TRANSITION_HISTORY_SIZE)
#    self.transitions.append(Transition(red_game_state, last_action, None, reward))
    self.old_game_states[i] = red_game_state
    self.self_actions[i] = last_action
    self.responses[i] = reward

    gradient_descent(self)
    self.rho *= (FINAL_TEMP/INITIAL_TEMP)**(1/N_ROUNDS)
    
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(dict(betas=self.betas, rho=self.rho), file)
    self.logger.debug(f"Betas and rho ({self.rho}) updated")


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
 #       e.KILLED_OPPONENT: 1,
  #      e.KILLED_SELF: 0,
        e.GOT_KILLED: -1,
        e.INVALID_ACTION: -1,
    #    e.CRATE_DESTROYED: .5,
     #   WALKED_TO_COIN: .1,
      #  WALKED_FROM_COIN: -.1,
        WALKED_TO_BOMB: -1,
        WALKED_FROM_BOMB: .1
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
