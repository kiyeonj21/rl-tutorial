# Adapted from: https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl
# Adapted from: https://github.com/omerbsezer/Reinforcement_learning_tutorial_with_demo
# the Monte Carlo Epsilon-Greedy method to find the optimal policy and value function
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from rl.games import standard_grid, negative_grid

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
EPS = 0.
MAX_ITER = 5000
MAX_STEP = 1000


def gen_episode(grid, policy):
    # starting at (2,0)
    s = (2, 0)  # start state
    grid.set_state(s)
    a = policy[s]
    a = grid.eps_greedy(action=a, eps=EPS)
    r = grid.move(a)
    step = 1
    states_actions_rewards = [(s,a,r)]
    while not grid.game_over():
        s = grid.current_state()
        a = policy[s]
        a = grid.eps_greedy(action=a,eps=EPS)
        r = grid.move(a)
        step += 1
        states_actions_rewards.append((s, a, r))
        if step>MAX_STEP:
            break
    return states_actions_rewards


# game setup
# OPTION 1: standard reward
# grid = standard_grid()
# OPTION 2: negative reward on none reward states
grid = negative_grid(step_cost=-0.1)

# input : arbitrary target policy
policy = {}
for s in grid.actions.keys():
    policy[s] = np.random.choice(grid.actions[s])

# initialize Q(s,a) and return
Q = {}
C = {}
returns = {}
for s in grid.actions:
    Q[s]={}
    C[s]={}
    for a in grid.actions[s]:
        Q[s][a]=np.random.random()
        C[s][a]=0.

# initial Q values for all states in grid
print("initial policy:")
grid.print_policy(policy)
print("initial Q")
print(Q)
print("initial C")
print(C)

# repeat
deltas = []
for it in range(MAX_ITER):
    if it % 1000 == 0:
        print("iteration:", it)
    states_actions_returns = gen_episode(grid, policy)
    states_actions_returns.reverse()
    G = 0
    w = 1
    biggest_change = 0
    for id, (s, a, r) in enumerate(states_actions_returns):
        G = GAMMA * G + r
        C[s][a] = C[s][a] + w
        old_q = Q[s][a]
        Q[s][a] = Q[s][a] + (w / C[s][a]) * (G - Q[s][a])
        biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
    deltas.append(biggest_change)
plt.plot(deltas)
plt.show()


print("optiaml Q")
print(Q)

print("optimal policy:")
grid.print_policy(policy)