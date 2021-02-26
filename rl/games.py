import gym
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as plt


class CartPole:
    def __init__(self):
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.env = gym.make('CartPole-v0')
        # set up matplotlib
        is_ipython = 'inline' in matplotlib.get_backend()
        if is_ipython:
            from IPython import display

        plt.ion()
        self.resize = T.Compose([T.ToPILImage(),
                                 T.Resize(40, interpolation=Image.CUBIC),
                                 T.ToTensor()])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.steps_done = 0

    def reset(self):
        self.env.reset()

    def get_cart_location(self, screen_width):
        env = self.env
        world_width = env.x_threshold * 2
        scale = screen_width / world_width
        return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

    def get_screen(self):
        env = self.env
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = env.render(mode='rgb_array').transpose((2, 0, 1))
        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = self.get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return self.resize(screen).unsqueeze(0).to(self.device)

    def get_n_actions(self):
        return self.env.action_space.n

    def select_action(self, state, policy_net):
        steps_done = self.steps_done
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * steps_done / self.EPS_DECAY)
        steps_done += 1
        self.steps_done = steps_done
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.get_n_actions())]], device=self.device, dtype=torch.long)


class GridWorld:  # Environment
    def __init__(self, width, height, start):
        self.width = width
        self.height = height
        self.i = start[0]
        self.j = start[1]
        self.all_possible_actions = ('U', 'D', 'L', 'R')

    def set(self, rewards, actions):
        self.rewards = rewards
        self.actions = actions

    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        return (self.i, self.j)

    def is_terminal(self, s):
        return s not in self.actions

    def move(self, action):
        # check if legal move first
        if action in self.actions[(self.i, self.j)]:
            if action == 'U':
                self.i -= 1
            elif action == 'D':
                self.i += 1
            elif action == 'R':
                self.j += 1
            elif action == 'L':
                self.j -= 1
        return self.rewards.get((self.i, self.j), 0)

    def game_over(self):
        return (self.i, self.j) not in self.actions

    def all_states(self):
        return set(self.actions.keys()) | set(self.rewards.keys())

    def eps_greedy(self, action, eps=0.):
        if self.is_terminal(self.current_state()):
            return 'dot'
        p = np.random.random()
        num_actions = len(self.actions[self.current_state()])
        if p < (1 - eps + eps / num_actions):
            return action
        else:
            return np.random.choice(self.actions[self.current_state()])

    def prob_policy(self,action,state,soft_policy, eps):
        num_actions = len(self.actions[state])
        if soft_policy[state]==action:
            p = (1 - eps + eps / num_actions)
        else:
            p = eps / num_actions
        return p


    def print_values(self, value):
        for i in range(self.height):
            for j in range(self.width):
                v = value.get((i, j), 0)
                if v >= 0:
                    print(" %.2f|" % v, end="")
                else:
                    print("%.2f|" % v, end="")
            print("")
        print("-" * 24)

    def print_policy(self, policy):
        for i in range(self.height):
            for j in range(self.width):
                a = policy.get((i, j), ' ')
                print("  %s  |" % a, end="")
            print("")
        print("-" * 24)

    def initialize_Q(self):
        Q ={}
        states = self.all_states()
        for s in states:
            Q[s] = {}
            if self.is_terminal(s):
                Q[s]['dot'] = 0.
            else:
                for a in self.actions[s]:
                    Q[s][a] = np.random.random()
        return Q


def standard_grid():
    g = GridWorld(4, 3, (2, 0))
    rewards = {(0, 3): 1, (1, 3): -1}
    actions = {
        (0, 0): ('D', 'R'),
        (0, 1): ('L', 'R'),
        (0, 2): ('L', 'D', 'R'),
        (1, 0): ('U', 'D'),
        (1, 2): ('U', 'D', 'R'),
        (2, 0): ('U', 'R'),
        (2, 1): ('L', 'R'),
        (2, 2): ('L', 'R', 'U'),
        (2, 3): ('L', 'U'),
    }
    g.set(rewards, actions)
    return g


def negative_grid(step_cost=-0.1):
    # in this game we want to try to minimize the number of moves
    # so we will penalize every move
    g = standard_grid()
    g.rewards.update({
        (0, 0): step_cost,
        (0, 1): step_cost,
        (0, 2): step_cost,
        (1, 0): step_cost,
        (1, 2): step_cost,
        (2, 0): step_cost,
        (2, 1): step_cost,
        (2, 2): step_cost,
        (2, 3): step_cost,
    })
    return g