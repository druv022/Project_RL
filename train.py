import numpy as np
from environment import get_env
from model import* 
from replay import* 
import argparse
import torch
from torch import optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm as _tqdm


#-------setup seed-----------
random.seed(5)
torch.manual_seed(5)
np.random.seed(5)
#----------------------------


#----device-----------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#-------------------



def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)    

def get_epsilon(it):
    # YOUR CODE HERE
    
    if it < 1000:
        return -9.5*1e-4*it+1
    else:
        return 0.05
        

def select_action(model, state, epsilon):
    
    state = torch.from_numpy(state).float()
    with torch.no_grad():
        actions = model(state.to(device))
        
    rand_num = np.random.uniform(0,1,1)
    if epsilon > rand_num:
        index = torch.randint(0,len(actions),(1,1))
    else:
        value, index = actions.max(0)
        
    return int(index.item())
def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

def compute_q_val(model, state, action):
    
    # YOUR CODE HERE
    actions = model(state)
    return actions.gather(1, action.unsqueeze(1))

def compute_target(model_target, reward, next_state, done, discount_factor):
    # done is a boolean (vector) that indicates if next_state is terminal (episode is done)
    # YOUR CODE HERE
    non_terminal_states_mask = torch.tensor([1 if not s else 0 for s in done])
    non_terminal_states = next_state[non_terminal_states_mask.nonzero().squeeze(1)]
    
    next_state_values = torch.zeros(done.size()[0]).to(device)
    next_state_values[non_terminal_states_mask.nonzero().squeeze(1)],_ = model_target(non_terminal_states).max(1)
    
    target = reward + discount_factor*next_state_values
    
    return target.detach().unsqueeze(1)

def train(model, model_target, memory, optimizer, batch_size, discount_factor, TAU):
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
    
    # with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
    target = compute_target(model_target, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.mse_loss(q_val, target)

    # backpropagation of loss to Neural Network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    soft_update(model, model_target, TAU)
    
    return loss.item()


def main():

    #update this disctionary as per the implementation of methods
    memory= {'NaiveReplayMemory':NaiveReplayMemory}
    env, (input_size, output_size) = get_env(ARGS.env)
    env.seed(5)

    network = { 
                'CartPole-v1':CartNetwork(input_size, output_size, ARGS.num_hidden).to(device),
                'MountainCar-v0':MountainNetwork(input_size, output_size, ARGS.num_hidden).to(device),
                'LunarLander-v2':LanderNetwork(input_size, output_size, ARGS.num_hidden).to(device)
              }

    #-----------initialization---------------
    
    replay = memory[ARGS.replay](ARGS.buffer)

    #-----------------------------------------
    model =  network[ARGS.env]#local_network
    model_target = network[ARGS.env]#target_network
    #-----------------------------------------
    
    optimizer = optim.Adam(model.parameters(), ARGS.lr)

    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    scores = []# list containing scores from each episode
    scores_window = deque(maxlen=100) 
    eps = ARGS.EPS
    STEP = 200
    #-------------------------------------------------------

    for i_episode in range(ARGS.num_episodes):
        # YOUR CODE HERE
        # Sample a transition
        s = env.reset()
        epi_duration = 0
        score=0
        for t in range(1000):
            env.render()
            # eps = get_epsilon(global_steps)
            #-------------------------------
            model.eval()
            a = select_action(model, s, eps)
            model.train()
            #------------------------------
            s_next, r, done, _ = env.step(a)
            # print(r, done)
            replay.push((s, a, r, s_next, done))
            loss = train(model, model_target, replay, optimizer, ARGS.batch_size, ARGS.discount_factor, ARGS.TAU)

            s = s_next
            epi_duration += 1
            global_steps +=1
            score += r
            if done:
                break
        eps = max(0.01, ARGS.eps_decay*eps)
        episode_durations.append(epi_duration)
        scores_window.append(score)# save most recent score
        scores.append(score) 
        
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            # torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
        if epi_duration >= 199:
            print("Failed to complete in trial {}".format(i_episode))
            
        else:
            print("Completed in {} trials".format(i_episode))
            # break
            
    #
    cumsum = np.cumsum(np.insert(episode_durations, 0, 0)) 
    cumsum = (cumsum[10:] - cumsum[:-10]) / float(10)
    env.close()
    plt.plot(cumsum)
    plt.title('Episode durations per episode')
    plt.show()    
    return episode_durations








if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', default=1000, type=int,
                        help='max number of episodes')
    parser.add_argument('--batch_size', default=64, type=int)
                      
    parser.add_argument('--num_hidden', default=128, type=int,
                        help='dimensionality of hidden space')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--discount_factor', default=0.85, type=float)
    parser.add_argument('--replay', default='NaiveReplayMemory',type=str,
                        help='type of experience replay')
    parser.add_argument('--env', default='LunarLander-v2', type=str,
                        help='environments you want to evaluate')
    parser.add_argument('--buffer', default='10000', type=int,
                        help='buffer size for experience replay')
    parser.add_argument('--TAU', default='1e-3', type=float,
                        help='parameter for soft update of weight')
    parser.add_argument('--EPS', default='1.0', type=float,
                        help='epsilon')
    parser.add_argument('--eps_decay', default='.995', type=float,
                        help='decay constant')
    

    ARGS = parser.parse_args()

    main()
    # python train.py --env MountainCar-v0 --lr 0.001 --num_episodes 5000 --num_hidden 64
    # python train.py --buffer 100000 --discount_factor 0.99 --lr .0005

