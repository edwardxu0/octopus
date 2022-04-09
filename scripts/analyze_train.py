import enum
from pprint import pp
from turtle import Turtle, up
import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path
from run import TRAIN_LOG_DIR
from src.plot.train_progress import ProgressPlot

def analyze_train(problems, nets, itst, ocrc, upbd, seeds, max_safe_relus, show=False, debug=False):
    for p in problems:
        for n in nets:
            for s in seeds:
                plot_progress(p, n, itst, ocrc, upbd, s, max_safe_relus, debug=debug, show=show)

            # analyze_net_space(n, itst, ocrc, s, msr, debug=debug, show=show)


def analyze_net_space(problem, net, itst, ocrc, upbd, seed, msr, log_path='meta/rurh.csv', debug=False, show=False):
    res = np.zeros((len(itst),len(ocrc),4))

    res_lines = ['itst, ocrc, safe_ge_zero, safe_le_zero, total_safe, % safe, test acc.\n']
    for i, i_ in enumerate(itst):
        for j, j_ in enumerate(ocrc):
            for k, k_ in enumerate(upbd):
                model_name = f'p={problem}_m={net}_itst={i_}_ocrc={j_}_upbd={k_}_seed={seed}'
                log = f"{TRAIN_LOG_DIR}/{model_name}.txt"
                lines = open(log, 'r').readlines()[-8:]

                safe_relu = []
                safe_relu_le_zero = []
                safe_relu_ge_zero = []

                for l in lines:
                    if 'Train' in l:
                        tks = l.split()
                        print(tks)
                        SR = int(tks[9][:-1])
                        SR_le_zero = int(tks[11][:-1])
                        SR_ge_zero = int(tks[13][:-1])
                        assert SR == SR_ge_zero + SR_le_zero

                        safe_relu += [SR]
                        safe_relu_le_zero += [SR_le_zero]
                        safe_relu_ge_zero += [SR_ge_zero]

                    elif 'Test' in l:
                        ta = float(l.split()[6].split('/')[0])/100
                    else:
                        pass
                
            sr = np.mean(np.array(safe_relu))
            srlz = np.mean(np.array(safe_relu_le_zero))
            srgz = np.mean(np.array(safe_relu_ge_zero))

            res_lines += [f'{i_}%, {j_}%, {srgz:5.2f}, {srlz:5.2f}, {sr:5.2f}, {sr/1.28:5.2f}%, {ta}%\n']
            res[i][j][0] = i_
            res[i][j][1] = j_
            res[i][j][2] = sr/msr*100
            res[i][j][3] = ta
    open(log_path,'w').writelines(res_lines)
    plot_3d(res, net, seed, debug=debug, show=show)

    
def plot_3d(res, net, seed, debug=False, show=False):
    if debug:
        print('3D Plot ... ')

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')

    res = res.transpose(2,0,1).reshape(4,-1)
    xs = res[0]
    ys = res[1]
    zs = res[2]
    zs_ = res[3]
    
    ax.scatter(xs, ys, zs, marker='o', color = 'blue', label='Safe ReLU')
    ax.scatter(xs, ys, zs_, marker='x', color = 'red', label='Accuracy')
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Occurance')

    # ax.set_zlabel('Safe ReLU(o)')
    # ax.set_zlabel('Accuracy(x)')
    ax.set_zlabel('Safe ReLU %(o), Accuracy %(x)')
    
    X = sorted([int(x) for x in set(res[0])])
    Y = sorted([int(x) for x in set(res[1])])

    Z = zs.reshape((len(X), len(Y)))
    Z_ = zs_.reshape((len(X), len(Y)))
    X, Y = np.meshgrid(X, Y)

    ax.plot_surface(X, Y, Z.T, cmap="Blues")
    ax.plot_surface(X, Y, Z_.T, cmap="Reds")

    # ax.view_init(30, -135)
    # ax.view_init(30, 45)
    ax.view_init(30, -60)
    plt.legend()
    plt.title(f'Safe ReLU %(o) vs. Accuracy %(x) of {net} with seed {seed}')
    
    # plt.savefig("figs/rurh_safe_ReLU.pdf", format="pdf", bbox_inches="tight")
    # plt.savefig("figs/rurh_accuracy.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"figs/BS_3D_{net}_{seed}.pdf", format="pdf", bbox_inches="tight")

    if show:
        plt.show()

    if debug:
        print('3D Plot done.')


def plot_progress(problem, net, itst, ocrc, upbd, seed, msr, fig_root='./figs_progress', debug=False, show=False):
    if debug:
        print('Progress Plot ... ')

    # fig_dir = fig_root+f'_{net}_{seed}'
    fig_dir = 'fig_bs'
    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    # res = np.zeros((len(itst),len(ocrc),4))
    # res_lines = ['itst, ocrc, safe_ge_zero, safe_le_zero, total_safe, % safe, test acc.\n']

    pplot = ProgressPlot()

    for i, i_ in enumerate(itst):
        for j, j_ in enumerate(ocrc):
            for k, k_ in enumerate(upbd):
                if debug:
                    print('.', end='', flush=True)

                model_name = f'p={problem}_m={net}_itst={i_}_ocrc={j_}_upbd={k_}_seed={seed}'
                log = f"{TRAIN_LOG_DIR}/{model_name}.txt"
                lines = [x for x in open(log, 'r').readlines()] # or 'Test' in x]
                
                safe_relu = []
                safe_relu_per_property = []
                hammer_points = []
                losses = []

                test_batch_id = []
                test_accuracy = []
                
                batch_counter = 0
                for l in lines:
                    if 'Train' in l:
                        batch_counter += 1
                        tks = l.split()
                        sr = int(tks[12][:-1])
                        safe_relu += [sr]
                        loss = float(tks[7])
                        losses += [loss]
                        hammer_points += [batch_counter] if tks[9] == 'T' else []

                    elif 'Test' in l:
                        test_accuracy += [int(l.split()[-2].split('/')[0])/10000]
                        test_batch_id += [batch_counter]
                    elif 'property' in l:
                        safe_relu_per_property += [[int(x) for x in l.strip().split(':')[1][2:-1].split(',')]]
                
                safe_relu_per_property = np.mean(np.array(safe_relu_per_property), axis=1)
                
                X1 = range(len(safe_relu))
                Y1 = safe_relu

                X2 = hammer_points
                Y2 = np.array(Y1)[hammer_points]

                X3 = test_batch_id
                Y3 = np.array(test_accuracy)

                X4 = range(len(losses))
                Y4 = losses

                pplot.draw_train(X1, Y1, X2, Y2, (0, msr[net]))
                
                pplot.draw_safe_relu_pp(test_batch_id, safe_relu_per_property)

                pplot.draw_accuracy(X3, Y3, X4, Y4, (0,1))

                title = f'# Safe ReLU with {i_}% Intensity and {j_}% Occurance Bias Shaping'
                path = f"{fig_dir}/{model_name}_acc={Y3[-1]*100:5.2f}.png"
                pplot.save(title, path)
                pplot.clear()

    print()
    if debug:
        print('Progress Plot done.')