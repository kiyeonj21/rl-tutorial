# Adapted from: https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl
# Adapted from: https://github.com/omerbsezer/Reinforcement_learning_tutorial_with_demo
# Value iteration
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from rl.games import standard_grid, negative_grid


theta = 1e-3
GAMMA = 0.9
# ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


# game setup
# OPTION 1: standard reward
# grid = standard_grid()
# OPTION 2: negative reward on none reward states
grid = negative_grid(step_cost=-0.1)


# initialize policy
policy = {}
for s in grid.actions.keys():
    policy[s] = np.random.choice(grid.actions[s])

# initialize state-value function
V = {}
states = grid.all_states()
for s in states:
    # Option 1: zero initialize
    # V[s] = 0
    # Option 2: random initialize
    if s in grid.actions:
        V[s] = np.random.random()
    else:
        # terminal state
        V[s] = 0

# print initial policy and state-value
print("initial policy:")
grid.print_policy(policy)
print("initial value:")
grid.print_values(V)


# repeat until convergence
it = 0
while True:
    it += 1
    print("policy at iter %d: " % it)
    grid.print_policy(policy)
    print("values at iter %d: " % it)
    grid.print_values(V)

    biggest_change = 0
    for s in states:
        old_v = V[s]
        if s in policy:
            new_v = float('-inf')
            for a in grid.actions[s]:
                grid.set_state(s)
                r = grid.move(a)
                s_next = grid.current_state()
                v = r + GAMMA * V[s_next]
                if v > new_v:
                    new_v = v
            V[s] = new_v
            biggest_change = max(biggest_change, np.abs(old_v - V[s]))

    if biggest_change < theta:
        break

# find an optimal policy that leads to the optimal value function
for s in policy.keys():
    best_a = None
    best_value = float('-inf')
    # loop through all possible actions to find the best current action
    for a in grid.actions[s]:
        grid.set_state(s)
        r = grid.move(a)
        s_next = grid.current_state()
        v = r + GAMMA * V[s_next]
        if v > best_value:
            best_value = v
            best_a = a
    policy[s] = best_a

# our goal here is to verify that we get the same answer as with policy iteration
print("optimal state-value:")
grid.print_values(V)
print("optimal policy:")
grid.print_policy(policy)
