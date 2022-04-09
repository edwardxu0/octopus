import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from run import TRAIN_LOG_DIR, VERI_LOG_DIR

from gdvb.plot.pie_scatter import PieScatter2D


CODE_V = {'unsat':1, 'sat':2, 'unknown':3, 'timeout':4, 'memout':4, 'error':5}


def analyze_tv(problems, nets, epses, verifiers, msr):
    df = parse()
    for v in verifiers:
        df_tmp = df[df['v'] == v]
        for p in problems:
            pie_data = []
            for n in nets:
                df_o = get_base(df_tmp, p, n)
                #df_bs = get_bs_best(df_tmp, n)
                df_bs = get_bs(df_tmp, p, n)
                print(len(df_o), len(df_bs))
                assert len(df_o) == len(df_bs)
                

                LMD_average = lambda X : np.mean(np.array(X, dtype=np.float32), axis=1)

                vr1, vt1, sr1, srpp1 = get_vr_vt(df_o, epses)
                vt1 = LMD_average(vt1)
                sr1 = LMD_average(sr1)
                srpp1 = LMD_average(srpp1)

                vr2, vt2, sr2, srpp2 = get_vr_vt(df_bs, epses)
                vt2 = LMD_average(vt2)
                sr2 = LMD_average(sr2)
                srpp2 = LMD_average(srpp2)

                # df_o.append(df_bs).to_csv(f'meta_bs_v1.0.1/VR_{v}.csv')

                title = f'Verification time of {p} on {n} via {v} verifier'
                save_path = f'meta_bs_v1.0.2/VT_{p}_{n}_{v}.png'
                print(save_path)

                # plot_bs_only(vt2, sr2, epses, title, save_path, msr[n])
                plot(vt1, vt2, sr1, sr2, srpp1, srpp2, epses, title, save_path, msr[n])

                pie_data += [vr1]
                pie_data += [vr2]

            pie_data = np.array(pie_data)
            ps2d = PieScatter2D(pie_data)
            xticks = []
            for n in nets:
                xticks += [n+' baseline']
                xticks += [n+' bias_shaping']
            ps2d.draw(xticks, epses, '', 'Epsilon', title=f'Verification Answer of {p}/{v}', rotation=(45,0))
            ps2d.save(f'meta_bs_v1.0.2/VA_{p}_{v}.png')
        

def parse():
    res = []
    files = os.listdir(VERI_LOG_DIR)
    for f in files:
        tk = f[:-4].split('_')

        if 'Net_' in f:
            tk = [tk.pop(0)+'_'+tk.pop(0)] + tk
        else:
            pass
        
        if 'v=' not in f:
            tk = tk + ['v='+tk.pop()]

        params = {}
        for x in tk:
            tmp = x.split('=')
            tmp[1] = int(tmp[1]) if tmp[1].isdigit() else tmp[1]
            if type(tmp[1]) == str:
                try:
                    float(tmp[1])
                    tmp[1] = float(tmp[1])
                except:
                    pass            
                
            params[tmp[0]] = tmp[1]

        # bs = params['itst'] == '0' or params['ocrc'] == '0'

        # accuracy
        t_log_path = f"{TRAIN_LOG_DIR}/p={params['p']}_m={params['m']}_itst={params['itst']}_ocrc={params['ocrc']}_upbd={params['upbd']}_seed={params['seeds']}.txt"
        lines = open(t_log_path, 'r').readlines()

        srs = []
        srs_pp = []
        for l in lines[-10:]:
            if 'Safe ReLU' in l:
                sr = int(l.split()[-5][:-1])
                srs += [sr]
            elif 'Test' in l:
                acc = float(int(l.split()[-2].split('/')[0])/100)
            elif 'property' in l:
                srs_pp += [[int(x) for x in l.strip().split(':')[1][2:-1].split(',')]]
        srs = np.mean(srs)
        params['acc'] = acc
        params['sr'] = srs
        params['sr_pp'] = np.mean(np.array(srs_pp))

        # v results
        lines = open(f'{VERI_LOG_DIR}/{f}', 'r').readlines()

        vr = None
        vt = None
        for l in lines:
            if '  result: ' in l:
                if 'Error' in l:
                    vr =  'error'
                    vt = 600
                    # print('error: vt = 600', params)
                    break
                else:
                    vr = l.strip().split()[-1]
            elif '  time: ' in l:
                vt = float(l.strip().split()[-1])
            elif 'Timeout' in l:
                vr = 'timeout'
                vt = (l.strip().split()[-3])
        assert vr is not None and vt is not None, (vr,vt)
        params['vr'] = vr
        params['vt'] = vt

        
        res += [params]

    label = res[0].keys()
    data = {}
    for l in label:
        tmp = []
        for r in res:
            tmp += [r[l]]
        data[l] = tmp
    
    df = pd.DataFrame(data)

    return df


def get_base(df, problem, net):
    df_tmp = df[df['p'] == problem]
    df_tmp = df_tmp[df_tmp['m'] == net]
    df_tmp = df_tmp[df_tmp['itst'] == 0]
    df_tmp = df_tmp.sort_values(by='eps')
    
    return df_tmp

def get_bs_best(df, net,  order_by=['acc','sr']):
    df_tmp = df[df['m'] == net]
    df_tmp = df_tmp[df_tmp['itst'] != 0]

    for order in order_by:
        best_item = sorted(df_tmp[order].tolist())[0]
        df_tmp = df_tmp[df_tmp[order] == best_item]
    df_tmp = df_tmp.sort_values(by='eps')
    return df_tmp

def get_bs(df, problem, net):
    df_tmp = df[df['p'] == problem]
    df_tmp = df_tmp[df_tmp['m'] == net]
    df_tmp = df_tmp[df_tmp['itst'] != 0]
    df_tmp = df_tmp.sort_values(by='eps')

    return df_tmp

def get_vr_vt(df, epses):
    vr_all = []
    vt_all = []
    sr_all = []
    sr_pp_all = []
    for eps in epses:
        tmp = df[df['eps'] == eps]
        vr = tmp['vr'].tolist()
        vt = tmp['vt'].tolist()
        sr = tmp['sr'].tolist()
        sr_pp = tmp['sr_pp'].tolist()
        vr_all += [[CODE_V[x] for x in vr]]
        vt_all += [vt]
        sr_all += [sr]
        sr_pp_all += [sr_pp]
    return vr_all, vt_all, sr_all, sr_pp_all


def plot(Y1, Y2, Y3, Y4, Y5, Y6, xticks, title, path, msr):
    fig = plt.figure(figsize=(10,6))
    X = [*range(len(Y1))]
    ax1 = fig.add_subplot()
    ax1.plot(Y1, color = 'lightblue', alpha=1, marker='o', label = 'Baseline')
    ax1.plot(Y2, color = 'darkblue', alpha=1, marker='o', label = 'Bias shaping')
    ax1.legend(loc='center left')

    ax2 = ax1.twinx()
    ax2.plot(Y3, color = 'pink', alpha=1, marker='x', label = 'Baseline')
    ax2.plot(Y4, color = 'red', alpha=1, marker='x', label = 'Bias shaping')

    ax2.plot(Y5, color = 'bisque', alpha=1, marker='v', label = 'Baseline(P)')
    ax2.plot(Y6, color = 'darkorange', alpha=1, marker='v', label = 'Bias shaping(P)')
    
    ax2.set_ylim(0, msr)
    ax2.legend(loc='lower right')

    ax1.set_xticks(X, xticks)
    ax1.set_xlabel('Epsilon')
    ax1.set_ylabel('Verification Time(s)')
    ax1.set_title(title)

    ax2.set_ylabel('Training # Safe ReLU')

    plt.savefig(path, format="png", bbox_inches="tight")


def plot_bs_only(Y2,  Y4, xticks, title, path, msr):
    fig = plt.figure(figsize=(10,6))
    X = [*range(len(Y2))]
    ax1 = fig.add_subplot()
    # ax1.plot(Y1, color = 'lightblue', alpha=1, marker='o', label = 'Baseline')
    ax1.plot(Y2, color = 'darkblue', alpha=1, marker='o', label = 'Bias shaping')
    ax1.legend(loc='upper left')
    ax1.set_ylim(0,610)

    ax2 = ax1.twinx()
    # ax2.plot(Y3, color = 'pink', alpha=1, marker='x', label = 'Baseline')
    #ax2.plot(Y4, color = 'red', alpha=1, marker='x', label = 'Bias shaping')
    #ax2.set_ylim(0, msr)
    #ax2.legend(loc='upper right')

    ax1.set_xticks(X, xticks)
    ax1.set_xlabel('Epsilon')
    ax1.set_ylabel('Verification Time(s)')
    ax1.set_title(title)

    ax2.set_ylabel('Training # Safe ReLU')

    plt.savefig(path, format="png", bbox_inches="tight")