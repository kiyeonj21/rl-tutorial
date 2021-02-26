# Adapted from: https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl
# Adapted from: https://github.com/omerbsezer/Reinforcement_learning_tutorial_with_demo
# Monte Carlo ES (Exploring Starts), for estimating optimal policy
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from rl.games import standard_grid, negative_grid
from rl.utils import max_dict

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
EPS = 0.
MAX_ITER = 5000
MAX_STEP = 1000


def gen_episode(grid, policy):
    # starting at random state and random action
    start_states = list(grid.actions.keys())
    id = np.random.randint(len(start_states))
    s = start_states[id]
    a = np.random.choice(grid.actions[s])
    grid.set_state(s)
    r = grid.move(a)
    step = 1
    states_actions_rewards = [(s,a,r)]
    while not grid.game_over():
        s = grid.current_state()
        a = policy[s]
        r = grid.move(a)
        step += 1
        states_actions_rewards.append((s, a, r))
        if step>MAX_STEP:
            break
    return states_actions_rewards


# game setup
# OPTION 1: standard reward
grid = standard_grid()
# OPTION 2: negative reward on none reward states
# grid = negative_grid(step_cost=-0.1)


# initialize policy
policy = {}
for s in grid.actions.keys():
    policy[s] = np.random.choice(grid.actions[s])

# initialize Q(s,a) and return
Q = {}
returns = {}
for s in grid.actions:
    Q[s]={}
    for a in grid.actions[s]:
        Q[s][a]=np.random.random()
        returns[(s,a)] = []


# initial Q values for all states in grid
print("initial policy:")
grid.print_policy(policy)
print("initial Q")
print(Q)
print("initial returns")
print(returns)

# repeat
deltas = []
for it in range(MAX_ITER):
    if it % 1000 ==0:
        print("iteration:",it)
    states_actions_returns = gen_episode(grid, policy)
    states_actions_returns.reverse()
    states_actions = [(item[0],item[1]) for item in states_actions_returns]
    G = 0
    biggest_change = 0
    for id, (s, a, r) in enumerate(states_actions_returns):
        G = GAMMA * G + r
        if (s,a) not in set(states_actions[(id + 1):]):
            returns[(s,a)].append(G)
            old_q = Q[s][a]
            Q[s][a] = np.mean(returns[(s,a)])
            biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
            policy[s] = max_dict(Q[s])[0]
    deltas.append(biggest_change)
plt.plot(deltas)
plt.show()


print("optiaml Q")
print(Q)

print("optimal policy:")
grid.print_policy(policy)

