# Adapted from: https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl
# Adapted from: https://github.com/omerbsezer/Reinforcement_learning_tutorial_with_demo
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from rl.games import standard_grid, negative_grid

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
MAX_ITER = 1000


def gen_episode(grid, policy):
    # starting at (2,0)
    s = (2, 0)
    grid.set_state(s)
    a = policy[s]
    r = grid.move(a)
    states_actions_rewards = [(s,a,r)]
    while not grid.game_over():
        s = grid.current_state()
        a = policy[s]
        r = grid.move(a)
        states_actions_rewards.append((s, a, r))
    return states_actions_rewards


def play_game(grid, policy):
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])
    # starting at (2,0)
    s = (2,0)
    grid.set_state(s)
    a = policy[s]
    s = grid.current_state()
    states_and_rewards = [(s, 0)]  # list of tuples of (state, reward)
    while not grid.game_over():
        a = policy[s]
        r = grid.move(a)
        s = grid.current_state()
        states_and_rewards.append((s, r))
    # calculate the returns by working backwards from the terminal state
    G = 0
    states_and_returns = []
    first = True
    for s, r in reversed(states_and_rewards):
        # the value of the terminal state is 0 by definition
        # we should ignore the first state we encounter
        # and ignore the last G, which is meaningless since it doesn't correspond to any move
        if first:
            first = False
        else:
            states_and_returns.append((s, G))
        G = r + GAMMA * G
    states_and_returns.reverse()  # we want it to be in order of state visited
    return states_and_returns


# game setup
# OPTION 1: standard reward
grid = standard_grid()
# OPTION 2: negative reward on none reward states
# grid = negative_grid(step_cost=-0.1)


# input : target policy
policy = {
    (2, 0): 'U',
    (1, 0): 'U',
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'R',
    (1, 2): 'R',
    (2, 1): 'R',
    (2, 2): 'R',
    (2, 3): 'U',
}


# initialize state-value and return
V = {}
returns = {}  # dictionary of state -> list of returns we've received
states = grid.all_states()
for s in states:
    returns[s]=[]
    V[s]=np.random.random()


# print initial policy, initial state-value, and return
print("initial policy:")
grid.print_policy(policy)
print("initial value:")
grid.print_values(V)
print("initial return:")
print(returns)


# repeat
for t in range(MAX_ITER):
    states_actions_returns = gen_episode(grid, policy)
    states_actions_returns.reverse()
    states = [item[0] for item in states_actions_returns]
    G = 0
    for id, (s, a, r) in enumerate(states_actions_returns):
        G = GAMMA * G + r
        if s not in set(states[(id+1):]):
            returns[s].append(G)
            V[s] = np.mean(returns[s])

print("state-value (V(s)):")
grid.print_values(V)
print("policy:")
grid.print_policy(policy)
