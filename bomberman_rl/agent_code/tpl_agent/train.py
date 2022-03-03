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
TRANSITION_HISTORY_SIZE = 1000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
GAMMA = .5
ALPHA = 1
BATCH_SIZE = 5

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def response(self, new_game_state, reward):
    return reward + GAMMA * np.max(Qs(self, new_game_state))

    
def gradient_descent(self, random_state=1337):
    rng = np.random.default_rng(random_state)
    for i, a in enumerate(ACTIONS):
        action_mask = self.self_actions==a
        action_length = int(np.sum(action_mask))
        if action_length==0:
            self.logger.debug(f"No gradient descent possible for {a}")
            continue
        idx = rng.choice(np.arange(action_length), BATCH_SIZE)
        
        Psi = self.old_game_states[action_mask][idx]
        Y = self.responses[action_mask][idx]
        self.betas[i] += ALPHA / BATCH_SIZE * np.sum(Psi * (Y - np.dot(Psi, self.betas[i]))[:,None], axis=0)


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.counter = 0
    self.old_game_states = np.empty((TRANSITION_HISTORY_SIZE, 12))
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

    # Idea: Add your own events to hand out rewards
#    if ...:
#        events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(psi(old_game_state), self_action, psi(new_game_state), reward_from_events(self, events)))
    if not old_game_state is None:
        self.old_game_states[self.counter] = psi(old_game_state)
        self.self_actions[self.counter] = self_action
        self.responses[self.counter] = response(self, new_game_state, reward_from_events(self, events))
        self.counter = (self.counter + 1) % TRANSITION_HISTORY_SIZE


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
    self.transitions.append(Transition(psi(last_game_state), last_action, None, reward_from_events(self, events)))
    self.old_game_states[self.counter] = psi(last_game_state)
    self.self_actions[self.counter] = last_action
    self.responses[self.counter] = reward_from_events(self, events)
    self.counter = (self.counter + 1) % TRANSITION_HISTORY_SIZE

    gradient_descent(self)
    self.rho = 1
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(dict(betas=self.betas, rho=self.rho), file)
    self.logger.debug(f"Betas updated to {self.betas}")


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 1,
        e.KILLED_SELF: -.5,
        e.GOT_KILLED: -1,
        e.INVALID_ACTION: -1,
        e.CRATE_DESTROYED: .5,
        e.WAITED: -.01
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
