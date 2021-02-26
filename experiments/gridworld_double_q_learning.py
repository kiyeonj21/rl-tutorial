# Adapted from: https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl
# Adapted from: https://github.com/omerbsezer/Reinforcement_learning_tutorial_with_demo
# Double Q-learning for estimating optimal policy (Q1~Q2~q*)
import pprint
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from rl.games import standard_grid, negative_grid
from rl.utils import max_dict

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALPHA = 0.1
EPS = 0.1
MAX_ITER = 5000

# game setup
# OPTION 1: standard reward
# grid = standard_grid()
# OPTION 2: negative reward on none reward states
grid = negative_grid(step_cost=-0.1)

# initialize Q1(s,a), Q2(s,a)
Q1 = grid.initialize_Q()
Q2 = grid.initialize_Q()

# print intial Q
print("Q1")
pprint.pprint(Q1, width=20)
print("Q2")
pprint.pprint(Q2, width=20)

# repeat max iteration times
deltas = []
for it in range(MAX_ITER):
    if it % 100 == 0:
        print("iteration:",it)
    # initialize state
    s = (2, 0)
    grid.set_state(s)
    biggest_change = 0
    while not grid.game_over():
        Q = {s: {}}
        for key, val in Q1[s].items():
            Q[s].update({key: val + Q2[s][key]})
        a = max_dict(Q[s])[0]
        a = grid.eps_greedy(a, EPS)
        r = grid.move(a)
        s_next = grid.current_state()
        p = np.random.random()
        if p <0.5:
            old_qsa = Q1[s][a]
            _a = max_dict(Q1[s_next])[0]
            Q1[s][a] = Q1[s][a] + ALPHA * (r + GAMMA * Q2[s_next][_a] - Q1[s][a])
            biggest_change = max(biggest_change, np.abs(old_qsa - Q1[s][a]))
        else:
            old_qsa = Q2[s][a]
            _a = max_dict(Q2[s_next])[0]
            Q2[s][a] = Q2[s][a] + ALPHA * (r + GAMMA * Q1[s_next][_a] - Q2[s][a])
            biggest_change = max(biggest_change, np.abs(old_qsa - Q2[s][a]))
        s = s_next
    deltas.append(biggest_change)
plt.plot(deltas)
plt.show()

# optimal policy and state-value function
policy = {}
V = {}
for s in grid.actions.keys():
    a, max_q = max_dict(Q1[s])
    policy[s] = a
    V[s] = max_q

print("optimal state-value (V(s)):")
grid.print_values(V)
print("optimal policy:")
grid.print_policy(policy)
