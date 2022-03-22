import collections
import numpy as np

"""
  Input: game_state: {
    'round': int,
    'step': int.
    'field': np.array(width, height),
    'bombs': [(int, int), int],
    'explosion_map': np.array(wifht, height)
    'coins': [(x, y)],
    'self': (str, int, bool, (int, int)),
    'others': [(str, int, bool, (int, int))],
    'user_input': str|None
  }
  Output: features: np.ndarray()
"""


# Feature Parameter
BOMBS_FEAT_SIZE = 12
BR = 3
FEAT_DIM = 14

A_TO_VEC = {
    "UP": np.array([0,-1]),
    "RIGHT": np.array([1, 0]),
    "DOWN": np.array([0, 1]),
    "LEFT": np.array([-1, 0]),
        }

A_TO_NUM = {
    "UP": 0, 
    "RIGHT": 1, 
    "DOWN": 2, 
    "LEFT": 3, 
    "WAIT": 4, 
    "BOMB": 5
    }


VEC_TO_IDX = {
    ( 0,-1): 0,
    ( 1, 0): 1,
    ( 0, 1): 2,
    (-1, 0): 3,
    }


def explosion_range(bomb_xy, arena):
    b = bomb_xy
    u_b, r_b, d_b, l_b = False, False, False, False
    for i in range(BR + 1):
        if not u_b:
            if (arena[b[0] - i, b[1]] != -1):
                if (arena[b[0] - i, b[1]] != 1):
                    arena[b[0] - i, b[1]] = 4
            else:
                u_b = True
        if not d_b:
            if arena[b[0] + i, b[1]] != -1:
                if (arena[b[0] + i, b[1]] != 1):
                    arena[b[0] + i, b[1]] = 4
            else:
                d_b = True
        if not l_b:
            if arena[b[0], b[1] - i] != -1:
                if (arena[b[0], b[1] - i] != 1):
                    arena[b[0], b[1] - i] = 4
            else:
                l_b = True
        if not r_b:
            if arena[b[0], b[1] + i] != -1:
                if (arena[b[0], b[1] + i] != 1):
                    arena[b[0], b[1] + i] = 4
            else:
                r_b = True


def state_to_features_bfs_2(game_state):
    if game_state is None:
        return np.zeros((FEAT_DIM))

    arena = game_state["field"]
    others = np.asarray([xy for (n, s, b, xy) in game_state["others"]])
    coins = np.asarray(game_state["coins"])
    bombs = game_state["bombs"]
    bombs_xy = np.asarray([[bomb_x, bomb_y] for ((bomb_x, bomb_y), _) in bombs])
    explosion_map = game_state["explosion_map"]
    _, _, has_bomb, (x, y) = game_state["self"]
    pos_self = np.asarray((x, y))

    #save original arena
    orig_arena = arena.copy()

    # overlay arena
    if others.shape[0] != 0:
        arena[others[:, 0], others[:, 1]] = 2
    if coins.shape[0] != 0:
        arena[coins[:, 0], coins[:, 1]] = 3
    if bombs_xy.shape[0] != 0:
        for b in bombs_xy:
            explosion_range(b, arena)

    explosion_array = np.argwhere(explosion_map == 1)
    if explosion_array.shape[0] != 0:
        arena[explosion_array[:, 0], explosion_array[:, 1]] = 5

    coin_step = np.array([1, 1])
    coin_dist = -1
    if coins.shape[0] != 0:
        next_coord, coin_dist = bfs_cc(orig_arena, arena, pos_self, "coin")
        if next_coord is not None:
            coin_step = next_coord - pos_self

    crate_step = np.array([1, 1])
    crate_dist = -1
    if 1 in arena:
        next_coord, crate_dist = bfs_cc(orig_arena, arena, (x, y), "crate")
        if next_coord is not None:
            crate_step =  next_coord - pos_self

    free_step = np.array([1, 1])
    free_dist = -1
    next_coord, free_dist = bfs_cc(orig_arena, arena, pos_self, "free")
    if next_coord is not None:
        free_step = next_coord - pos_self
    
    other_step = np.array([1, 1])
    other_dist = -1
    if others.shape[0] != 0:
        next_coord, other_dist = bfs_cc(orig_arena, arena, pos_self, "other")
        if next_coord is not None:
            other_step = next_coord - pos_self
    
    #drop test bomb at location to see if can escape
    explosion_range((x,y), arena)
    _, escape_dist = bfs_cc(orig_arena, arena, pos_self, "free")
    if escape_dist == -1 or escape_dist > 4:
        escape = [False]
    else:
        escape = [True]

    coin_feat = np.append(coin_step, coin_dist)
    crate_feat = np.append(crate_step, crate_dist)
    free_feat = np.append(free_step, free_dist)
    other_feat = np.append(other_step, other_dist)

    #set distances to maximal values of 4  over that threshhold to reduce feature space size
    ##for testing only
    # if not self.train:
    #     coin_dist = min(5, coin_dist)
    #     crate_dist = min(5, crate_dist)
    #     #other_dist = min(5, other_dist)
    #     #free_dist = min(2, free_dist)

    feature_vec = np.concatenate([coin_feat, crate_feat, free_feat, other_feat, escape, [has_bomb]])
    return feature_vec


def bfs_cc(orig_arena, arena, start, target: str):
    target_dict = {
        "free": 0,
        "other": 2,
        "crate": 1,
        "coin": 3,
    }
    if target == "free":
        crate_block = 1
        danger_block = -1
    elif target == "coin":
        crate_block = 1
        danger_block = 4
    elif target == "other":
        crate_block = 1
        danger_block = 4
    else:
        crate_block = -1
        danger_block = 4
    goal = target_dict[target]
    queue = collections.deque([[start]])
    seen = set(start)
    dist = -1
    while queue:
        path = queue.popleft()
        xy = path[-1]
        x = xy[0]
        y = xy[1]
        if arena[x][y] == goal:
            if len(path) > 1:
                dist = len(path)-1
                return path[1], dist
            else:
                dist = len(path)-1
                return path[0], dist
        for x2, y2 in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if (
                1 <= x2 < 16
                and 0 <= y2 < 16
                and arena[x2][y2] not in [-1, crate_block, danger_block, 5]
                and orig_arena[x2][y2] not in [-1]
                and (x2, y2) not in seen
            ):
                queue.append(path + [(x2, y2)])
                seen.add((x2, y2))

    return None, dist


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

def orbit(s):
    feats = {}
    for g, R in O2.items():
        sprime = s.copy()
        if not (s[0:2] == [1,1]).all():
            sprime[0:2] = s[0:2] @ R.T
        if not (s[3:5] == [1,1]).all():
            sprime[3:5] = s[3:5] @ R.T
        if not (s[6:8] == [1,1]).all():
            sprime[6:8] = s[6:8] @ R.T
        if not (s[9:11] == [1,1]).all():
            sprime[9:11] = s[9:11] @ R.T
        feats[g] = sprime
    return feats


def update_batch(self, old_feat, new_feat, reward, self_action):
    old_orbit = orbit(old_feat)
    new_orbit = orbit(new_feat)
    if self_action in ["WAIT", "BOMB"]:
        idx = A_TO_NUM[self_action]
        self.feat_history[idx].extend(old_orbit.values())
        self.next_feat_history[idx].extend(new_orbit.values())
        self.reward_history[idx].extend([reward]*len(O2))
    else:
        v = A_TO_VEC[self_action]
        for g, R in O2.items():
            idx = VEC_TO_IDX[tuple(v @ R.T)]
            self.feat_history[idx].append(old_orbit[g])
            self.next_feat_history[idx].append(new_orbit[g])
            self.reward_history[idx].append(reward)
    # self.logger.debug(f"Old Orbit: {old_orbit}")
    pass