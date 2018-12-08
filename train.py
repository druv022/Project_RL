import numpy as np
from environment import get_env
from model import QNetwork
from replay import *
import argparse
import torch
from torch import optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm as _tqdm
import random
from torch.autograd import Variable

# -------setup seed-----------
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
# ----------------------------


# ----device-----------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# -------------------


def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)


def get_epsilon(it):
    # using exponential decay rather than linear decay
    # YOUR CODE HERE
    return np.exp(-it / 450)


def get_beta(it, total_it, beta0):
    return beta0 + (it/total_it) * (1 - beta0)


def select_action(model, state, epsilon):
    # YOUR CODE HERE
    state = torch.from_numpy(state).float()
    with torch.no_grad():
        actions = model(state.to(device))

        rand_num = np.random.uniform(0, 1, 1)
        if epsilon > rand_num:
            index = torch.randint(0, len(actions), (1, 1))
        else:
            value, index = actions.max(0)

        return int(index.item())


def compute_q_val(model, state, action):
    # YOUR CODE HERE
    actions = model(state)
    return actions.gather(1, action.unsqueeze(1))


def compute_target(model, reward, next_state, done, discount_factor):
    # done is a boolean (vector) that indicates if next_state is terminal (episode is done)
    # YOUR CODE HERE
    non_terminal_states_mask = torch.tensor([1 if not s else 0 for s in done])
    non_terminal_states = next_state[non_terminal_states_mask.nonzero().squeeze(1)]

    next_state_values = torch.zeros(done.size()[0]).to(device)
    next_state_values[non_terminal_states_mask.nonzero().squeeze(1)], _ = model(non_terminal_states).max(1)

    target = reward + discount_factor * next_state_values

    return target.unsqueeze(1)


def train(model, memory, optimizer, batch_size, discount_factor, beta):
    # DO NOT MODIFY THIS FUNCTION

    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    # transitions = memory.sample(batch_size)
    # ---------------------------- per--------------------------
    transitions, batch_idx, priorities = memory.sample(batch_size)

    # transition is a list of 5-tuples, instead we want 5 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float).to(device)
    action = torch.tensor(action, dtype=torch.int64).to(device)  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float).to(device)
    reward = torch.tensor(reward, dtype=torch.float).to(device)
    done = torch.tensor(done, dtype=torch.uint8).to(device)  # Boolean

    # compute the q value
    q_val = compute_q_val(model, state, action)

    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_target(model, reward, next_state, done, discount_factor)

    # loss is measured from error between current and newly expected Q values
    w = (1/(batch_size * np.array(priorities)) ** beta)
    w = torch.tensor(w, dtype=torch.float).to(device).detach()

    loss = torch.mean(w * abs(q_val - target))
    # loss = F.mse_loss(q_val, target)
    # ------------------------------------------------------ per -------------------------------------
    td_error = target - q_val
    for i in range(batch_size):
        val = abs(td_error[i].data[0])
        memory.update(batch_idx[i], val)

    # backpropagation of loss to Neural Network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def main():
    # update this dictionary as per the implementation of methods
    memory = {'NaiveReplayMemory': NaiveReplayMemory,
              'CombinedReplayMemory': CombinedReplayMemory,
              'PrioritizedReplayMemory': PrioritizedReplayMemory}

    # -----------initialization---------------
    env, (input_size, output_size) = get_env(ARGS.env)
    replay = memory[ARGS.replay](ARGS.buffer, ARGS.pmethod)

    model = QNetwork(input_size, output_size, ARGS.num_hidden).to(device)
    optimizer = optim.Adam(model.parameters(), ARGS.lr)

    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    # -------------------------------------------------------

    for i in tqdm(range(ARGS.num_episodes)):
        # YOUR CODE HERE
        # Sample a transition
        s = env.reset()
        done = False
        epi_duration = 0
        while not done:
            eps = get_epsilon(global_steps)
            a = select_action(model, s, eps)
            s_next, r, done, _ = env.step(a)

            q_val = model(torch.tensor(s, dtype=torch.float).to(device))[a]
            target = torch.tensor(r, dtype=torch.float).to(device) + \
                     ARGS.discount_factor * model(torch.tensor(s_next, dtype=torch.float).to(device)).max()
            # print(q_val)
            # print(target)
            #td_err = target - q_val
            td_err = F.smooth_l1_loss(q_val, target.detach())
            # print(td_err)
            # print(torch.isnan(td_err))

            replay.push(td_err.detach(), [s, a, r, s_next, done])

            # replay.push((s, a, r, s_next, done))
            beta = get_beta(i, ARGS.num_episodes, ARGS.beta0)
            train(model, replay, optimizer, ARGS.batch_size, ARGS.discount_factor, beta)

            s = s_next
            epi_duration += 1
            global_steps += 1
            #if epi_duration >= 100:
            #    done = True

        episode_durations.append(epi_duration)

    #
    cum_sum = np.cumsum(np.insert(episode_durations, 0, 0))
    cum_sum = (cum_sum[10:] - cum_sum[:-10]) / float(10)

    plt.plot(cum_sum)
    plt.title('Episode durations per episode')
    plt.show()
    return episode_durations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', default=100, type=int,
                        help='max number of episodes')
    parser.add_argument('--batch_size', default=64, type=int)

    parser.add_argument('--num_hidden', default=128, type=int,
                        help='dimensionality of hidden space')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--discount_factor', default=0.8, type=float)
    # parser.add_argument('--replay', default='NaiveReplayMemory',type=str,
    #                    help='type of experience replay')

    parser.add_argument('--replay', default='PrioritizedReplayMemory', type=str,
                        help='type of experience replay')

    # parser.add_argument('--replay', default='CombinedReplayMemory', type=str,
    #                   help='type of experience replay')
    parser.add_argument('--env', default='CartPole-v1', type=str,
                        help='environments you want to evaluate')
    parser.add_argument('--buffer', default='100000', type=int,
                        help='buffer size for experience replay')
    parser.add_argument('--beta0', default=0.4, type=float)
    parser.add_argument('--pmethod', default='prop', type=str, help='proritized reply method: {prop or rank}')
    ARGS = parser.parse_args()
    main()
