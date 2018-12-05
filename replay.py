import random
from collections import deque
from environment import get_env


class NaiveReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        
        # YOUR CODE HERE
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

#Add different experience replay methods

class CombinedReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        
        # YOUR CODE HERE
        self.memory.append(transition)
        self.transition = transition

    def sample(self, batch_size):
        samples = random.sample(self.memory, batch_size-1)
        samples.append(self.transition)
        return samples

    def __len__(self):
        return len(self.memory)






#sanity check
if __name__=="__main__":

    capacity = 10
    memory = NaiveReplayMemory(capacity)

    env, _ = get_env("Acrobot-v1")

    # Sample a transition
    s = env.reset()
    a = env.action_space.sample()
    s_next, r, done, _ = env.step(a)

    # Push a transition
    memory.push((s, a, r, s_next, done))

    # Sample a batch size of 1
    print(memory.sample(1))
