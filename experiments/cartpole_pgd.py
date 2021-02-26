import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.autograd import Variable
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
import gym

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from rl.games import CartPole
from rl.utils import plot_durations
from rl.models import PolicyNet

# game setup
game = CartPole()
game.reset()
plt.figure()
plt.imshow(game.get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
plt.title('Example extracted screen')
plt.show()


# hyperparams and utils
NUM_EPISODE = 50
BATCH_SIZE = 5
STEP_SIZE = 0.01
GAMMA = 0.99

# policy network setup
policy_net = PolicyNet()
optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=STEP_SIZE)

episode_durations = []

# Batch History
state_pool = []
action_pool = []
reward_pool = []
steps = 0

for i_episode in range(NUM_EPISODE):
    state = game.env.reset()
    state = torch.from_numpy(state).float()
    state = Variable(state)
    game.env.render(mode='rgb_array')

    for t in count():
        probs = policy_net(state)
        m = Bernoulli(probs)
        action = m.sample()

        action = action.data.numpy().astype(int)[0]
        next_state, reward, done, _ = game.env.step(action)
        game.env.render(mode='rgb_array')

        if done:
            reward = 0
        state_pool.append(state)
        action_pool.append(float(action))
        reward_pool.append(reward)

        state = next_state
        state = torch.from_numpy(state).float()
        state = Variable(state)

        steps += 1

        if done:
            episode_durations.append(t + 1)
            plot_durations(episode_durations)
            break

    # Update policy
    if i_episode > 0 and i_episode % BATCH_SIZE == 0:

        # Discount reward
        running_add = 0
        for i in reversed(range(steps)):
            if reward_pool[i] == 0:
                running_add = 0
            else:
                running_add = running_add * GAMMA + reward_pool[i]
                reward_pool[i] = running_add

        # Normalize reward
        reward_mean = np.mean(reward_pool)
        reward_std = np.std(reward_pool)
        for i in range(steps):
            reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

        # Gradient Desent
        optimizer.zero_grad()

        for i in range(steps):
            state = state_pool[i]
            action = Variable(torch.FloatTensor([action_pool[i]]))
            reward = reward_pool[i]

            probs = policy_net(state)
            m = Bernoulli(probs)
            loss = -m.log_prob(action) * reward  # Negtive score function x reward
            loss.backward()

        optimizer.step()

        state_pool = []
        action_pool = []
        reward_pool = []
        steps = 0
