import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

def get_color(counter):
    if counter > 9:
        counter = 0
    else:
        counter += 1
    return 'C'+str(counter)

def smooth(x, N=50):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def save_file(data_mean, file_name):
    with open(file_name,'w') as f:
        for i in range(len(data_mean)):
            f.write("%d\n" % data_mean[i])

def plot_data(data_mean, data_std,title, color,label=''):
    data_x = range(len(smooth(data_mean)))
    data_y_plus = smooth(data_mean+data_std)
    data_y_minus = smooth(data_mean-data_std)

    plt.plot(smooth(data_mean),color=color, label='mean '+label)
    plt.fill_between(data_x,data_y_plus,data_y_minus,color=color,alpha=0.2)
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
    counter = 0
    color = get_color(counter)
    file_name1 = ARGS.file + "_duration"
    episodes_data = read_files(file_name1)
    data_mean = np.mean(episodes_data,axis=0)
    data_std = np.std(episodes_data,axis=0)
    plot_data(data_mean, data_std, 'Episode durations per episode', color)
    save_file(data_mean, ARGS.newFile+'_durations.txt')

    color = get_color(counter)
    file_name2 = ARGS.file + "_rewards"
    rewards_data = read_files(file_name2)
    data_mean = np.mean(rewards_data,axis=0)
    data_std = np.std(rewards_data,axis=0)
    plot_data(data_mean, data_std, 'Rewards per episode', color)
    save_file(data_mean, ARGS.file_name+'_rewards.txt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='NaiveReplayMemory_prop_durations', type=str,
                        help='name of file to read until numeric value')
    parser.add_argument('--newFile', default='NaiveReplayAll',type=str, help='Name of the file')
    ARGS = parser.parse_args()
    main()