from collections import namedtuple, deque

import pickle
from typing import List
import numpy as np

import events as e
from .callbacks import psi, Qs, L, N_BOMBS, N_COINS, closest_coins

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 8 * 5000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
GAMMA = .5
ALPHA = 1
BATCH_SIZE = 50
INITIAL_TEMP = 50
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
VECTOR_TO_ACTIONS = {tuple(v): k for k, v in ACTIONS_TO_VECTOR.items()}  # ambiguity in 'BOMB'/'WAIT'

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
    'pi': lambda m: np.rot90(np.rot90(m)),
    '3pi/2': lambda m: np.rot90(np.rot90(np.rot90(m))),
    'sigma_x': np.flipud,
    'sigma_y': np.fliplr,
    'sigma_d1': lambda m: np.rot90(np.fliplr(m)),
    'sigma_d2': lambda m: np.rot90(np.flipud(m))
    }

origin = np.array([8,8])


def symmetry(game_state):
    if game_state is None:
        return [None]*8
    states = []
    for g in O2.keys():
        sym_state = game_state.copy()
        sym_state['field'] = O2_func[g](sym_state['field'])
        sym_state['explosion_map'] = O2_func[g](sym_state['explosion_map'])
        sym_state['self'] = (*sym_state['self'][:3], (sym_state['self'][3] - origin) @ O2[g].T + origin)
        for i, other in enumerate(sym_state['others']):
            sym_state['others'][i] = (*sym_state['others'][i][:3], (other[0] - origin) @ O2[g].T + origin)
        for i, bomb in enumerate(sym_state['bombs']):
            sym_state['bombs'][i] = ((bomb[0] - origin) @ O2[g].T + origin, bomb[1])
        sym_state['coins'] = (np.array(sym_state['coins']) - origin) @ O2[g].T + origin
        states.append(sym_state)
    return states
        
        
def symmetry_training(old_game_state, self_action, new_game_state):
    old_sym_states = symmetry(old_game_state)
    new_sym_states = symmetry(new_game_state)

    if self_action in ['WAIT', 'BOMB']:
        sym_actions = [self_action]*8
    else:
        sym_actions = [VECTOR_TO_ACTIONS[tuple(ACTIONS_TO_VECTOR[self_action] @ R.T)] for R in O2.values()]
    return old_sym_states, sym_actions, new_sym_states
    

def update_batch(self, old_game_state, self_action, new_game_state, reward):
    i = self.counter
    self.old_game_states[i:i+8] = [psi(self, s) for s in symmetry(old_game_state)]
    self.new_game_states[i:i+8] = [psi(self, s) for s in symmetry(new_game_state)]
    if self_action in ['WAIT', 'BOMB']:
        self.self_actions[i:i+8] = [self_action]*8
    else:
        self.self_actions[i:i+8] = [VECTOR_TO_ACTIONS[tuple(ACTIONS_TO_VECTOR[self_action] @ R.T)] for R in O2.values()]
    self.rewards[i:i+8] = [reward]*8
    self.counter = (self.counter + 8) % TRANSITION_HISTORY_SIZE
    pass


def response(self, new_game_state, reward):
#    self.logger.debug(f"new game state in response: {new_game_state}")
    return reward + GAMMA * np.max(Qs(self, new_game_state))

    
def gradient_descent(self):
    for i, a in enumerate(ACTIONS):
        action_mask = self.self_actions==a
        action_length = int(np.sum(action_mask))
        if action_length==0:
            self.logger.debug(f"No gradient descent possible for {a}")
            continue
        idx = np.random.choice(np.arange(action_length), BATCH_SIZE)
 
#        self.logger.debug(f"old game state: {self.old_game_states[action_mask][idx]}")
        Psi = self.old_game_states[action_mask][idx]
#        self.logger.debug(f"new game state in gradient descent method for action {a} {self.new_game_states[idx]}")
        Y = np.array([response(self, s, r) for s, r in zip(self.new_game_states[idx], self.rewards[idx])])
#        self.logger.debug(f"For action {a}: Y={Y}, Psi={Psi}")
        new_beta = self.betas[i] + ALPHA / BATCH_SIZE * np.sum(Psi * (Y - np.dot(Psi, self.betas[i]))[:,None], axis=0)
        self.betas[i] = new_beta / np.linalg.norm(new_beta)
#    self.logger.debug(f"betas after gradient descent {self.betas}")
    pass


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.rho = INITIAL_TEMP
#    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.counter = 0
    self.old_game_states = np.zeros((TRANSITION_HISTORY_SIZE, L))
    self.new_game_states = np.zeros_like(self.old_game_states)
    self.self_actions = np.empty(TRANSITION_HISTORY_SIZE).astype(str)
    self.rewards = np.empty(TRANSITION_HISTORY_SIZE)


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
        for i, bomb in enumerate(old_game_state['bombs']):
            if i>=N_BOMBS:
                break
            old_dist = np.linalg.norm(np.array(bomb[0])-old_pos, ord=1)
            if old_dist<=3:
                if np.linalg.norm(np.array(bomb[0])-new_pos, ord=1) > old_dist:
                    events.append(WALKED_FROM_BOMB)
                else:
                    events.append(WALKED_TO_BOMB)
        if not e.COIN_COLLECTED in events:
            # find closest coins
            _, idx, old_dist = closest_coins(old_pos, old_game_state['coins'], verbose=True)
            _, new_idx, new_dist = closest_coins(new_pos, np.array(new_game_state['coins'])[idx], verbose=True)
            potential = np.sum((new_dist - old_dist[new_idx])**2)            
            if potential<0:
                events.append(WALKED_TO_COIN)
            else:
                events.append(WALKED_FROM_COIN)

    update_batch(self, old_game_state, self_action, new_game_state, reward_from_events(self, events))


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
    
    update_batch(self, last_game_state, last_action, None, reward_from_events(self, events))

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
#        e.KILLED_OPPONENT: 1,
#        e.KILLED_SELF: 0,          
        e.GOT_KILLED: -5,
#        e.INVALID_ACTION: -1,
#        e.CRATE_DESTROYED: .5,
        WALKED_TO_COIN: .2,
        WALKED_FROM_COIN: -.1,
#        WALKED_TO_BOMB: -1,
#        WALKED_FROM_BOMB: .1,
#        e.MOVED_RIGHT: 1,
#        e.MOVED_UP: 1
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
