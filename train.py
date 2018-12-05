import numpy as np
from environment import get_env
from model import QNetwork
from replay import* 
import argparse
import torch
from torch import optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm as _tqdm


#-------setup seed-----------
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
#----------------------------


#----device-----------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#-------------------



def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)    

#using exponential decay rather than linear decay
def get_epsilon(it):
    
    # YOUR CODE HERE
    return np.exp(-it/450)

def select_action(model, state, epsilon):
    # YOUR CODE HERE
    state = torch.from_numpy(state).float()
    with torch.no_grad():
        actions = model(state.to(device))
        
        rand_num = np.random.uniform(0,1,1)
        if epsilon > rand_num:
            index = torch.randint(0,len(actions),(1,1))
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
    next_state_values[non_terminal_states_mask.nonzero().squeeze(1)],_ = model(non_terminal_states).max(1)
    
    target = reward + discount_factor*next_state_values
    
    return target.unsqueeze(1)

def train(model, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
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
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


def main():

    #update this disctionary as per the implementation of methods
    memory= {'NaiveReplayMemory':NaiveReplayMemory,'CombinedReplayMemory' :CombinedReplayMemory}

    #-----------initialization---------------
    env, (input_size, output_size) = get_env(ARGS.env)
    replay = memory[ARGS.replay](ARGS.buffer)

    model =  QNetwork(input_size, output_size, ARGS.num_hidden).to(device)
    optimizer = optim.Adam(model.parameters(), ARGS.lr)

    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    #-------------------------------------------------------

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

            replay.push((s, a, r, s_next, done))
            loss = train(model, replay, optimizer, ARGS.batch_size, ARGS.discount_factor)

            s = s_next
            epi_duration += 1
            global_steps +=1

        episode_durations.append(epi_duration)

    #
    cumsum = np.cumsum(np.insert(episode_durations, 0, 0)) 
    cumsum = (cumsum[10:] - cumsum[:-10]) / float(10)

    plt.plot(cumsum)
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
    parser.add_argument('--replay', default='NaiveReplayMemory',type=str,
                        help='type of experience replay')
    parser.add_argument('--env', default='CartPole-v1', type=str,
                        help='environments you want to evaluate')
    parser.add_argument('--buffer', default='100000', type=int,
                        help='buffer size for experience replay')

    ARGS = parser.parse_args()

    main()


