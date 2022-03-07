from collections import namedtuple, deque
import random
import numpy as np
from hashlib import sha1
import pickle
from typing import List
from agent_code.my_agent.callbacks import tabular_Q_func
import events as e
from .callbacks import ACTIONS, EXPLORATION_DECAY, EXPLORATION_MIN, lr_Q_func, state_to_features

# This is only an example!
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1000  # keep only ... last transitions
BATCH_SIZE = 20
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
ALPHA = 0.001
GAMMA = 0.9

A_TO_NUM = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3, "WAIT": 4, "BOMB": 5}

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def tabular_Q_update(self):
    transition = self.transitions[-1]
    used_Q_value = self.Q_values[A_TO_NUM[transition.action]]
    next_Q_values = tabular_Q_func(self, transition.next_state)

    rhs = ALPHA * (transition.reward + GAMMA * np.max(next_Q_values) - used_Q_value)
    self.Q_dicts[A_TO_NUM[transition.action]][hash(str(transition.state))] = (
        used_Q_value + rhs
    )

def lr_immediate_Q_update(self):
    transition = self.transitions[-1]
    idx = A_TO_NUM[transition.action] 
    next_q_value = lr_Q_func(self, transition.next_state)

    Y = transition.reward + GAMMA * np.max(next_q_value)
    rhs = Y - np.dot(transition.state, self.weights[idx]) * transition.state

    self.weights[idx] -= ALPHA * rhs


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
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
    # self.logger.debug(
    #     f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}'
    # )

    # Idea: Add your own events to hand out rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)

    if old_game_state is not None:
        # state_to_features is defined in callbacks.py
        self.transitions.append(
            Transition(
                state_to_features(old_game_state),
                self_action,
                state_to_features(new_game_state),
                reward_from_events(self, events),
            )
        )
        lr_immediate_Q_update(self)
        self.logger.info(f"weights: {self.weights}")


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
    # self.logger.debug(
    #     f'Encountered event(s) {", ".join(map(repr, events))} in final step'
    # )
    self.transitions.append(
        Transition(
            state_to_features(last_game_state),
            last_action,
            None,
            reward_from_events(self, events),
        )
    )

    if len(self.transitions) < BATCH_SIZE:
        return
    batch = random.sample(self.transitions, BATCH_SIZE)
    X = []
    targets = []
    for t in batch:
        q_update = t.reward 
        if (type(t.next_state) != None and type(t.state) != None):
            print('next state: ', t.next_state)
            print('state: ', t.state)
            if self.isFit:
                q_update = (t.reward + GAMMA + np.max(self.model.predict(t.next_state.reshape)))
            else:
                q_update = t.reward
        if self.isFit:
            q_values = self.model.predict(t.state)
        else:
            q_values = np.zeros(len(ACTIONS)).reshape(1,-1)
            print('q_values: ', q_values[0])
        q_values[0][A_TO_NUM[t.action]] = q_update

        X.append(list(t.state))
        targets.append(q_values)
    print('targets: ', targets)
    self.model.fit(X, targets)
    self.isFit = True
    self.exploration_rate *= EXPLORATION_DECAY
    self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    # self.logger.info(f"Q_dicts: {self.Q_dicts}")
    # self.logger.info(f"transitions: {self.transitions}")
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.INVALID_ACTION: -1,
        e.KILLED_SELF: -5,
        # e.KILLED_OPPONENT: 5,
        # PLACEHOLDER_EVENT: -0.1,  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    # self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
