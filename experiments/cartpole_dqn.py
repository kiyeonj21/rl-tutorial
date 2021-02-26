import matplotlib.pyplot as plt
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rl.utils import ReplayMemory, plot_durations
from rl.games import CartPole
from rl.models import DQN

# game setup
game = CartPole()
game.reset()
plt.figure()
plt.imshow(game.get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
plt.title('Example extracted screen')
plt.show()

# hyperparams and utils
BATCH_SIZE = 128
GAMMA = 0.999
TARGET_UPDATE = 10
NUM_EPISODE = 10
NUM_MEM = 10000


# policy and target network setup
init_screen = game.get_screen()
_, _, screen_height, screen_width = init_screen.shape
n_actions = game.get_n_actions()
policy_net = DQN(screen_height, screen_width, n_actions).to(game.device)
target_net = DQN(screen_height, screen_width, n_actions).to(game.device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(NUM_MEM)


# training loop
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = memory.transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=game.device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=game.device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


episode_durations = []
for i_episode in range(NUM_EPISODE):
    print(i_episode, "episode")
    # Initialize the environment and state
    game.env.reset()
    last_screen = game.get_screen()
    current_screen = game.get_screen()
    state = current_screen - last_screen
    for t in count():
        action = game.select_action(state, policy_net)
        _, reward, done, _ = game.env.step(action.item())
        reward = torch.tensor([reward], device=game.device)

        last_screen = current_screen
        current_screen = game.get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None
        memory.push(state, action, next_state, reward)
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations(episode_durations)
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
game.env.render()
game.env.close()
plt.ioff()
plt.show()