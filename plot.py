import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

def smooth(x, N=10):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def plot_data(title, data):
    data_mean = np.mean(data,axis=0)
    data_std = np.std(data,axis=0)

    plt.plot(smooth(data_mean),color='red', label='mean')
    plt.title(title)
    plt.show()

def read_files(file_name):
    j = 0
    fd_name = ARGS.file+"0.txt"
    exists = os.path.isfile(fd_name)
    data_all = []
    while exists:
        f = open(fd_name,"r")
        data_tmp = f.readlines()[:-1]
        data     = []
        for i in range(len(data_tmp)):
            data.append(int(data_tmp[i]))
        j += 1
        fd_name = ARGS.file+"%d.txt" % j
        exists = os.path.isfile(fd_name)
        data_all.append(np.asarray(data))
    
    return np.asarray(data_all)

def main():
    file_name1 = ARGS.file + "_duration"
    episodes_data = read_files(file_name1)
    plot_data('Episode durations per episode', episodes_data)

    file_name2 = ARGS.file + "_rewards"
    rewards_data = read_files(file_name2)
    plot_data('Rewards per episode', rewards_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='NaiveReplayMemory_prop_durations', type=str,
                        help='name of file to read until numeric value')
    ARGS = parser.parse_args()
    main()