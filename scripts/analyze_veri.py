import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path
from run import VERI_LOG_DIR, META_PATH

from gdvb.plot.pie_scatter import PieScatter2D

CODE_V = {'unsat':1, 'sat':2, 'unknown':3, 'timeout':4, 'memout':4, 'error':5}

def analyze_veri(nets, itst, ocrc, props, epses, seeds, verifiers, debug=False, show=False):
    for n in nets:
        for s in seeds:
            for v in verifiers:
                for e in epses:
                    analyze_veri1(n, itst, ocrc, props, e, s, v, debug=debug, show=show)


def analyze_veri1(net, itst, ocrc, props, eps, seed, verifier, debug=False, show=False):
    prefix = f'm={net}_eps={eps}_v={verifier}'
    res_v_path = f'{META_PATH}/{prefix}_res_v.npy'
    res_t_path = f'{META_PATH}/{prefix}_res_t.npy'

    # parse files
    if not Path(res_v_path).exists() or not Path(res_t_path).exists():
        res_v = []
        res_t = []
        for i, i_ in enumerate(itst):
            for j, j_ in enumerate(ocrc):

                r_v = []
                r_t = []
                for k, k_ in enumerate(props):
                    v = None
                    t = None

                    log = f'{VERI_LOG_DIR}/m={net}_itst={i_}_ocrc={j_}_prop={k_}_eps={eps}_seeds={seed}_{verifier}.txt'
                    lines = open(log, 'r').readlines()[-200:]

                    for l in lines:
                        if '  result: ' in l:
                            if 'Error' in l:
                                v = CODE_V['error']
                            else:
                                v = CODE_V[l.split()[-1]]
                        if '  time: ' in l:
                            t = float(l.split()[-1])
                        if 'Timeout' in l:
                            v = CODE_V['timeout']
                            t = float(l.split()[-1])

                    assert v is not None and t is not None, log
                    r_v += [v]
                    r_t += [t]

                if debug:
                    print(f'itst={i_}, ocrc={j_}, {r_v}, {r_t}')
                res_v += [r_v]
                res_t += [r_t]

        res_v = np.array(res_v)
        res_t = np.array(res_t)

        np.save(res_v_path, res_v)
        np.save(res_t_path, res_t)

    else:
        res_v = np.load(res_v_path)
        res_t = np.load(res_t_path)
    
    # plot
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot()

    res = res_t.reshape(21,21,5)
    res = np.mean(res, axis= -1)
    res = res/res[0][0]
    res = np.flipud(res)

    im = ax.imshow(res)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(itst)), labels=itst)
    ax.set_yticks(np.arange(len(ocrc)), labels=reversed(ocrc))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(ocrc)):
        for j in range(len(itst)):
            text = ax.text(j, i, f'{int(res[i, j]*100)}%',
                        ha="center", va="center", color="w")

    ax.set_xlabel('Intensity')
    ax.set_ylabel('Occurance')
    ax.set_title(f'Avg. Veri. Time of {net} with eps of {eps} on seed {seed} via {verifier}')

    if show:
        plt.show()
    
    plt.savefig(f"figs/V_Time_heatmap_{net}_{seed}_{eps}_{verifier}.pdf", format="pdf", bbox_inches="tight")

    # plot pie chart
    data = res_v.reshape(len(itst), len(ocrc), len(props))+1
    ps2d = PieScatter2D(data)
    ps2d.draw(itst, ocrc, 'Intensity', 'Ocrance', title=f'Verification Answers of {net} with eps of {eps} on seed {seed} via {verifier}')
    ps2d.save(f'figs/V_answer_PieChart_{net}_{seed}_{eps}_{verifier}.pdf')


