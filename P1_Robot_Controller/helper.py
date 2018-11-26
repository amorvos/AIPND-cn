import random
import time

MAZE = [
    [3, 2, 2, 2, 2, 2, 2, 2, 1],
    [0, 0, 2, 2, 2, 2, 2, 0, 0],
    [2, 0, 0, 2, 2, 2, 0, 0, 2],
    [2, 2, 0, 0, 2, 0, 0, 2, 2],
    [2, 2, 2, 0, 0, 0, 2, 2, 2]]


def fetch_maze():
    print("maze-id {}-{}".format(1, round(time.time())))
    print('[' + str(MAZE[0]) + ',')
    for line in MAZE[1:-1]:
        print(' ' + str(line) + ',')
    print(' ' + str(MAZE[-1]) + ']')
    return MAZE


env_data = [
    [3, 2, 2, 2, 2, 2, 2, 2, 1],
    [0, 0, 2, 2, 2, 2, 2, 0, 0],
    [2, 0, 0, 2, 2, 2, 0, 0, 2],
    [2, 2, 0, 0, 2, 0, 0, 2, 2],
    [2, 2, 2, 0, 0, 0, 2, 2, 2]]


def count_row_barriers(data: [[]], row: int):
    result = 0
    for ele in data[row - 1]:
        if ele == 2:
            result += 1
    return result


def count_col_barriers(data: [[]], col: int):
    result = 0
    for ele in data:
        if ele[col - 1] == 2:
            result += 1
    return result


def is_move_valid(env_data: [[]], loc: (), act: str):
    # 向上走
    if 'u' == act and loc[0] - 1 >= 0 and env_data[loc[0] - 1][loc[1]] != 2:
        return True
    # 向下走
    if 'd' == act and loc[0] + 1 < env_data.__len__() and env_data[loc[0] + 1][loc[1]] != 2:
        return True
    # 向左走
    if 'l' == act and loc[1] - 1 >= 0 and env_data[loc[0]][loc[1] - 1] != 2:
        return True
    # 向右走
    if 'r' == act and loc[1] + 1 < env_data[0].__len__() and env_data[loc[0]][loc[1] + 1] != 2:
        return True
    return False


actions = ['u', 'd', 'l', 'r']


def valid_actions(env_data: [[]], loc: ()) -> []:
    result = []
    for action in actions:
        if is_move_valid(env_data, loc, action):
            result.append(action)
    return result


def move_robot(loc: (), act: str) -> ():
    # 向上走
    if 'u' == act and is_move_valid(env_data, loc, act):
        return loc[0] - 1, loc[1]
    # 向下走
    if 'd' == act and is_move_valid(env_data, loc, act):
        return loc[0] + 1, loc[1]
    # 向左走
    if 'l' == act and is_move_valid(env_data, loc, act):
        return loc[0], loc[1] - 1
    # 向右走
    if 'r' == act and is_move_valid(env_data, loc, act):
        return loc[0], loc[1] + 1
    return loc


def random_choose_actions(env_data: [[]], loc: ()):
    for time in range(0, 300):
        if env_data[loc[0]][loc[1]] == 3:
            print("在第{}个回合找到宝藏！".format(time))
            return
        action = random.choice(valid_actions(env_data, loc))
        loc = move_robot(loc, action)


def dfs(self, loc):
    queue, order = [], []
    queue.append(loc)
    while queue:
        v = queue.pop()
        order.append(v)
        for w in self.sequense[v]:
            if w not in order and w not in queue:
                queue.append(w)
    return order


def bfs(self, loc):
    queue, order = [], []
    queue.append(loc)
    order.append(loc)
    while queue:
        v = queue.pop(0)
        for w in self.sequense[v]:
            if w not in order:
                order.append(w)
                queue.append(w)
    return order


def walk(env_data, x, y):
    if x == 0 and y == 0:
        print("successful!")
        return True
    print(x, y)
    env_data[x][y] = 2
    actions = valid_actions(env_data, (x, y))
    if actions.__len__() == 0:
        return False
    for action in actions:
        loc = move_robot((x, y), action)
        if not walk(env_data, loc[0], loc[1]):
            env_data[x][y] = 1
        else:
            return False
    return True
