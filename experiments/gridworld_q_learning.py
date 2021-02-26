# Adapted from: https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl
# Adapted from: https://github.com/omerbsezer/Reinforcement_learning_tutorial_with_demo
# Q-learning (off-policy TD control) for estimating optimal policy
import pprint
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from rl.games import standard_grid, negative_grid
from rl.utils import max_dict

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALPHA = 0.1
EPS = 0.1
MAX_ITER = 10000

# game setup
# OPTION 1: standard reward
grid = standard_grid()
# OPTION 2: negative reward on none reward states
# grid = negative_grid(step_cost=-0.1)

# initialize Q(s,a)
Q = grid.initialize_Q()

# print intial Q
pprint.pprint(Q, width=20)

# repeat
deltas = []
for it in range(MAX_ITER):
    if it % 100 == 0:
        print("iteration:",it)
    # initialize state
    s = (2, 0)  # start state
    grid.set_state(s)
    biggest_change = 0
    while not grid.game_over():
        a = max_dict(Q[s])[0]
        a = grid.eps_greedy(a, EPS)
        r = grid.move(a)
        s_next = grid.current_state()
        old_qsa = Q[s][a]
        Q[s][a] = Q[s][a] + ALPHA * (r + GAMMA * max_dict(Q[s_next])[1] - Q[s][a])
        biggest_change = max(biggest_change, np.abs(old_qsa - Q[s][a]))
        s = s_next
    deltas.append(biggest_change)
plt.plot(deltas)
plt.show()

# optimal policy and state-value function
policy = {}
V = {}
for s in grid.actions.keys():
    a, max_q = max_dict(Q[s])
    policy[s] = a
    V[s] = max_q

print("optimal state-value (V(s)):")
grid.print_values(V)
print("optimal policy:")
grid.print_policy(policy)
