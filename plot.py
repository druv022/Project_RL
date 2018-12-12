import numpy as np
import argparse
import matplotlib.pyplot as plt

def smooth(x, N=10):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def main():
    f=open("durations%d.txt" % ARGS.file, "r")
    
    episode_durations_tmp = f.readlines()[:-1]
    episode_durations     = []
    for i in range(len(episode_durations_tmp)):
        episode_durations.append(int(episode_durations_tmp[i]))
    
    f=open("rewards%d.txt" % ARGS.file, "r")
    
    rewards_per_episode_tmp = f.readlines()[:-1]
    rewards_per_episode     = []

    for i in range(len(rewards_per_episode_tmp)):
        rewards_per_episode.append(int(rewards_per_episode_tmp[i]))
        
    plt.plot(smooth(episode_durations,10))
    plt.title('Episode durations per episode')
    plt.show()
    plt.plot(smooth(rewards_per_episode,10))
    plt.title("Rewards per episode")
    plt.show()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default=None, type=int,
                        help='number of file to read')
    ARGS = parser.parse_args()
    main()