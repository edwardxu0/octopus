import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

ROOT = 'raw_train'


def parse_txt(path, list_line=False, dtype=np.float32):
    loss = []
    lines = open(path, 'r').readlines()
    for l in lines:
        if not list_line:
            loss += [float(l.strip())]
        else:
            loss += [float(l.split(',')[0][7:])]
            # for x in l.strip()[1:-1].split(','):
            #    loss += [float(x)]

    return np.array(loss, dtype=dtype)


def plot_loss_distribution(loss):
    plt.scatter(loss, norm.pdf(loss, np.mean(loss), np.std(loss)))
    plt.show()
    print(f'mean: {np.mean(loss)}, std:{np.std(loss)}')


def main():
    # no train
    #l1loss0 = parse_txt(os.path.join(ROOT, 'l1loss0.txt'))
    #rsloss0 = parse_txt(os.path.join(ROOT, 'rsloss0.txt'), list_line=True)
    #wdloss0 = parse_txt(os.path.join(ROOT, 'wdloss0.txt'))

    # train with l1+rsloss
    l1loss1 = parse_txt(os.path.join(ROOT, 'l1loss1.txt'))
    rsloss1 = parse_txt(os.path.join(ROOT, 'rsloss1.txt'))
    # wdloss1 = parse_txt(os.path.join(ROOT, 'wdloss1.txt'))

    # train with rsloss
    #l1loss2 = parse_txt(os.path.join(ROOT, 'l1loss2.txt'))
    #rsloss2 = parse_txt(os.path.join(ROOT, 'rsloss2.txt'), list_line=True)
    #wdloss2 = parse_txt(os.path.join(ROOT, 'wdloss2.txt'))

    # plot_loss_distribution(l1loss0)
    # plot_loss_distribution(rsloss0)
    # plot_loss_distribution(wdloss0)

    plot_loss_distribution(l1loss1)
    plot_loss_distribution(rsloss1)
    # plot_loss_distribution(wdloss1)

    # plot_loss_distribution(l1loss2)
    # plot_loss_distribution(rsloss2)
    # plot_loss_distribution(wdloss2)

if __name__ == '__main__':
    main()