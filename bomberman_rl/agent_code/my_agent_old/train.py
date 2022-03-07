from agent_code.my_agent.callbacks import A_TO_NUM, Q_func
from collections import namedtuple, deque
import pickle
from typing import List
import events as e
from agent_code.my_agent.callbacks import state_to_features
import numpy as np

# This is only an example!
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 10  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
COIN_K = 1

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
ALPHA = 0.1
BATCH_SIZE = 10
L = 1

def TD_target_respones(self, gamma=0.1):
    transition = self.transitions.pop()

    # Q function of next value
    # next_feat = state_to_features(transition.next_state)
    next_q_value = Q_func(self, transition.next_state)

    return transition.reward + gamma * np.max(next_q_value)


def gradient_update(self, idx):

    if len(self.feat_history[idx]) >= BATCH_SIZE:
        selection_size = len(self.target_history[idx])
        selection_mask = np.random.permutation(np.arange(selection_size))[0:BATCH_SIZE]

        Y_tt = np.array(self.target_history[idx])[selection_mask]
        feats = np.array(self.feat_history[idx])[selection_mask]
        RHS = np.sum(np.sum(feats * (Y_tt[:, None] - feats * self.betas[idx]), axis=1))
        self.betas[idx] = self.betas[idx] + (ALPHA / BATCH_SIZE) * RHS

    else:
        Y_tt = np.array(self.target_history[idx])
        feats = np.array(self.feat_history[idx])
        RHS = np.sum(np.sum(feats * (Y_tt[:, None] - feats * self.betas[idx]), axis=1))
        self.betas[idx] = self.betas[idx] + (ALPHA / len(self.feat_history[idx])) * RHS

    self.betas=[np.array(beta)/np.linalg.norm(np.array(self.betas), ord=2) for beta in self.betas]


def setup_training(self):
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):

    # Idea: Add your own events to hand out rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    if old_game_state is not None:
        self.transitions.append(
            Transition(
                state_to_features(old_game_state),
                self_action,
                state_to_features(new_game_state),
                #reward_from_events(self, events),
                total_rewards(self, events, old_game_state, new_game_state)
            )
        )
        feat = state_to_features(old_game_state)

        Y_tt = TD_target_respones(self, 0.1)
        idx = A_TO_NUM[self_action]

        self.target_history[A_TO_NUM[self_action]].append(Y_tt)
        self.feat_history[idx].append(feat)
        #self.logger.info(f"self.target_history: {self.target_history}")
        #self.logger.info(f"self.feat_history: {self.feat_history}")

        gradient_update(self, idx)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    #self.logger.debug(
    #    f'Encountered event(s) {", ".join(map(repr, events))} in final step'
    #)
    self.transitions.append(
        Transition(
            state_to_features(last_game_state),
            last_action,
            None,
            reward_from_events(self, events),
        )
    )

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.betas, file)

def total_rewards(self, events, old_game_state, new_game_state):
    rewards = [reward_from_events(self, events)]
    #rewards.append(potential_reward(self, old_game_state, new_game_state))
    return sum(rewards)

def reward_from_events(self, events: List[str]):
    game_rewards = {
        e.COIN_COLLECTED: 100,
        e.INVALID_ACTION: -100,
        e.KILLED_SELF: -100,
        #e.SURVIVED_ROUND: 100,
        e.CRATE_DESTROYED: 2,
        e.BOMB_DROPPED: 10
        # e.KILLED_OPPONENT: 2,
        # PLACEHOLDER_EVENT: -0.1,  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    #self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def potential_reward(self, old_game_state, new_game_state):
    old_pot = coin_potential(self, old_game_state)+bomb_potential(self, old_game_state)
    new_pot = coin_potential(self, new_game_state)+bomb_potential(self, old_game_state)
    return old_pot-new_pot #reward when player gets closer to coin(s)

def coin_potential_func(pos):
    x,y = pos
    if (x,y) == (0,0):
        return 0
    #return (np.abs(x)**L+np.abs(y)**L)**(1/L)/20
    return np.linalg.norm(pos, ord=L)**2/10

def coin_potential(self, game_state):
    _, score, bombs_left, (x, y) = game_state["self"]
    coins = game_state["coins"]
    coin_xy_rel = np.array([[coin_x-x,coin_y-y] for (coin_x,coin_y) in coins])
    if len(coin_xy_rel) == 0:
        return 0
    if coin_xy_rel.ndim == 1:
        return coin_potential_func(coin_xy_rel)
    if len(coins)<COIN_K:
        sorted_index = np.argsort(np.linalg.norm(coin_xy_rel, axis=1))
        coins_feat = np.concatenate((coin_xy_rel[sorted_index], np.zeros((COIN_K-len(coins),2))))
    else:
        sorted_index = np.argsort(np.linalg.norm(coin_xy_rel, axis=1))[:COIN_K]
        coins_feat = coin_xy_rel[sorted_index]
    pot = [coin_potential_func(pos) for pos in coins_feat]
    non_zero = np.sum(pot!=0)
    if non_zero == 0:
        return 0
    else:
        return np.sum(pot)/non_zero

def bomb_range(t):
    #range of explosion as function of remaining time
    t_inv = 5-t #goes from 1 to 4
    b_range=[[0,0]]
    for i in range(t_inv)[0:]:
        b_range.append([0,i])
        b_range.append([0,-i])
        b_range.append([i,0])
        b_range.append([-i,0])
    return b_range

def bomb_potential_func(pos):
    bomb_x,bomb_y,bomb_t = pos
    for position in bomb_range(bomb_t):
        if bomb_x == position[0] and bomb_y == position[1]:
            return -8*(5-bomb_t) #penalty if player is in range for a bomb
    else:
        return 0
    

def bomb_potential(self, game_state):
    _, score, bombs_left, (x, y) = game_state["self"]
    bombs = game_state["bombs"]
    bomb_xy_rel = [[bomb_x-x, bomb_y-y, bomb_t] for ((bomb_x, bomb_y), bomb_t) in bombs]
    pot = [bomb_potential_func(pos) for pos in bomb_xy_rel]
    non_zero = np.sum(pot!=0)
    if non_zero == 0:
        return 0
    else:
        return np.sum(pot)/non_zero