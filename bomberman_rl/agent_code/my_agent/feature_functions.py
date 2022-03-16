import collections
from operator import ne
from attr import field

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

import numpy as np
from agent_code.my_agent import callbacks

# Feature Parameter
COIN_K = 1
SIGHT = 1
BOMBS_FEAT_SIZE = 12
BR = 3

def state_to_features_bfs_2(game_state):
    if game_state == None:
        return None
    arena = game_state["field"]
    others = np.asarray([xy for (n, s, b, xy) in game_state["others"]])
    coins = np.asarray(game_state["coins"])
    bombs = game_state["bombs"]
    bombs_xy = np.asarray([[bomb_x, bomb_y] for ((bomb_x, bomb_y), _) in bombs])
    explosion_map = game_state["explosion_map"]
    _, _, _, (x, y) = game_state["self"]
    pos_self = np.asarray((x, y))

    rel_field = np.array([arena[x-1, y], arena[x+1, y], arena[x, y-1], arena[x, y+1]])

    #save original arena
    orig_arena = arena
    
    # overlay arena
    if others.shape[0] != 0:
        arena[others[:, 0], others[:, 1]] = 2

    if coins.shape[0] != 0:
        arena[coins[:, 0], coins[:, 1]] = 3

    # if bombs_xy.shape[0] != 0:
    #     arena[bombs_xy[:, 1], bombs_xy[:, 0]] = 4

    is_in_danger_zone = False
    if bombs_xy.shape[0] != 0:
        for b in bombs_xy:
            u_b, r_b, d_b, l_b = False, False, False, False
            for i in range(BR + 1):
                if not u_b:
                    if arena[b[0] - i, b[1]] != -1:
                        arena[b[0] - i, b[1]] = 4
                    else:
                        u_b = True
                if not d_b:
                    if arena[b[0] + i, b[1]] != -1:
                        arena[b[0] + i, b[1]] = 4
                    else:
                        d_b = True
                if not l_b:
                    if arena[b[0], b[1] - i] != -1:
                        arena[b[0], b[1] - i] = 4
                    else:
                        l_b = True
                if not r_b:
                    if arena[b[0], b[1] + i] != -1:
                        arena[b[0], b[1] + i] = 4
                    else:
                        r_b = True

    if arena[pos_self[0], pos_self[1]] == 4:
        is_in_danger_zone = True

    explosion_array = np.argwhere(explosion_map == 1)
    if explosion_array.shape[0] != 0:
        arena[explosion_array[:, 0], explosion_array[:, 1]] = 5

    coin_step = np.array([1, 1])
    coin_dist = -1
    if coins.shape[0] != 0:
        next_coord, coin_dist = bfs_cc(orig_arena, arena, pos_self, "coin")
        # print('coin dist: ', coin_dist)
        if next_coord is not None:
            coin_step = next_coord - pos_self

    crate_step = np.array([1, 1])
    crate_dist = -1
    if 1 in arena:
        next_coord, crate_dist = bfs_cc(orig_arena, arena, (x, y), "crate")
        if next_coord is not None:
            crate_step =  next_coord - pos_self
        # if 1 in rel_field:
        #     # print('1 in rel field')
        #     next_step = np.array([1, 1])

    free_step = np.array([1, 1])
    free_dist = -1
    # print("in danger zone")
    next_coord, free_dist = bfs_cc(orig_arena, arena, pos_self, "free")
    if next_coord is not None:
        free_step = next_coord - pos_self
    
    other_step = np.array([1, 1])
    other_dist = -1
    if others.shape[0] != 0:
        next_coord, other_dist = bfs_cc(orig_arena, arena, pos_self, "other")
        if next_coord is not None:
            other_step = next_coord - pos_self

    # print("final arena: ", arena.T)
    # print("--------------------------------------------------------")

    # is_in_danger_zone = np.asarray(is_in_danger_zone).astype(int)
    coin_feat = np.append(coin_step, coin_dist)
    crate_feat = np.append(crate_step, crate_dist)
    free_feat = np.append(free_step, free_dist)
    other_feat = np.append(other_step, other_dist)

    feature_vec = np.concatenate((coin_feat, crate_feat, free_feat, other_feat))
    # print('feature vec: ', feature_vec)
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
    # print('target: ', target_dict[target])
    while queue:
        path = queue.popleft()
        xy = path[-1]
        x = xy[0]
        y = xy[1]
        if arena[x][y] == goal:
            # print('arena[y][x]', arena[x][y])
            # print('path: ', path)
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
