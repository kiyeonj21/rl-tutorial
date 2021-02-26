# the temporal difference 0 method to find the optimal policy
# only policy evaluation, not optimization
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from rl.games import standard_grid, negative_grid

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALPHA = 0.1


# def play_game(grid, policy):
#     # returns a list of states and corresponding rewards (not returns as in MC)
#     # start at the designated start state
#     s = (2, 0)
#     grid.set_state(s)
#     states_and_rewards = [(s, 0)]  # list of tuples of (state, reward)
#     while not grid.game_over():
#         a = policy[s]
#         a = random_action(a)
#         r = grid.move(a)
#         s = grid.current_state()
#         states_and_rewards.append((s, r))
#     return states_and_rewards


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


# initial policy and state-value
print("initial policy:")
grid.print_policy(policy)
print("initial value:")
grid.print_values(V)

# repeat until convergence
for it in range(2000):
    if it % 1000 == 0:
        print("iteration:", it)
    # initialize state
    s = (2, 0)  # start state
    grid.set_state(s)
    biggest_change = 0
    while not grid.game_over():
        a = policy[s]
        r = grid.move(a)
        s_next = grid.current_state()
        old_v = V[s]
        V[s] = V[s] + ALPHA * (r+GAMMA * V[s_next] - V[s])
        s = s_next

print("policy:")
grid.print_policy(policy)
print("value:")
grid.print_values(V)
