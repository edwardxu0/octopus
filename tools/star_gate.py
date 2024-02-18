#!/usr/bin/env python

import os
import re
import pathlib
import copy
import argparse
import pandas as pd
import numpy as np

import seaborn as sns


import matplotlib.pyplot as plt

from nexus import Settings


def _parse_args():
    parser = argparse.ArgumentParser(description="Plotter")
    parser.add_argument("task", type=str)
    parser.add_argument("feathers", nargs="+", type=str)
    parser.add_argument("--ta_threshold", type=int, default=100)
    parser.add_argument("--root", type=str, default="results")
    parser.add_argument("--s", type=str)
    parser.add_argument("--a", type=str)
    args = parser.parse_args()
    return args


def main(args):
    root = args.root

    dfs = []
    for x in args.feathers:
        df = pd.read_feather(os.path.join(root, f"{x}.feather"))
        dfs += [df]

    df = pd.concat(dfs, ignore_index=True)

    #print(df.columns.to_list())

    print('Networks: ',set(df["network"]))

    #print("Before renaming:", set(df["stabilizer"]))
    df["stabilizer"] = df["stabilizer"].map(Settings.stabilizer_convert_names)
    #print("After renaming:", set(df["stabilizer"]))

    df["network"] = df["network"].map(Settings.network_conver_names)
    df["verifier"] = df["verifier"].map(Settings.verifier_convert_names)

    stabilizers = list(sorted(set(df["stabilizer"])))

    stabilizers.sort(key=Settings.stabilizers_order.index)

    # stabilizers = set(df["stabilizer"])

    print('Stabilizers: ', stabilizers)

    if args.task == "pb":
        plot_baseline(df)
    elif args.task == "st":
        stable_relu_table(df, stabilizers, args.ta_threshold)
    elif args.task == "vt":
        verification_table(df, stabilizers, args.ta_threshold)
    elif args.task == "svt":
        stable_relu_table(df, stabilizers, args.ta_threshold)
        verification_table(df, stabilizers, args.ta_threshold)
    elif args.task == "vp":
        verification_plot(df, stabilizers)
    elif args.task == "scatter":
        scatter_plot(df, stabilizers)
    elif args.task == "eps":
        plot_eps(df, stabilizers)

    else:
        assert False

    print("Verification Time: ", np.sum(df["veri time"]) / 3600)
    print("Training Time: ", np.sum(df["training time"]) / 3600 / 50)


def stable_relu_table(df, stabilizers, ta_threshold):
    # artifacts = [*reversed(sorted(set(df["artifact"])))]
    artifacts = list(set(df["network"]))
    artifacts.sort(key=Settings.network_order.index)
    res_ = []
    res_e = []

    df = df[df["verifier"] == "NNEnum"]

    for i, h in enumerate(stabilizers):
        res = []
        rese = []
        for a in artifacts:
            # dft = df[df["artifact"] == a]
            dft = df[df["network"] == a]
            dft = dft[dft["stabilizer"] == h]
            ta = dft["test accuracy"]
            ta = np.array(ta) * 100

            # dfb = df[df["artifact"] == a]
            dfb = df[df["network"] == a]
            dfb = dfb[dfb["stabilizer"] == "Baseline"]
            ta_baseline = np.mean(dfb["test accuracy"].values) * 100

            # relative test accuracy
            # dft = dft[dft["test accuracy"] >= (ta_baseline - ta_threshold / 100)]

            # srn = dft["relu accuracy veri"]
            srn = dft["nb stable"] / (dft["nb stable"] + dft["nb unstable"])
            srn = list(srn)
            srn = np.array(srn) * 100

            # if h == "B_NIP" and a == "FC6":
            #    print(np.mean(ta), np.mean(srn))
            #    exit()

            index_meat_ta_threshold = np.where(ta + ta_threshold >= ta_baseline)
            # index_meat_ta_threshold = np.where(ta >= ta_baseline * 0.98)
            # print(index_meat_ta_threshold)
            if len(index_meat_ta_threshold) != 0:
                ta = ta[index_meat_ta_threshold]
                srn = srn[index_meat_ta_threshold]
            else:
                ta = -1
                srn = -1

            assert len(ta) == len(srn)
            if len(ta) == 0:
                res += [-1, -1]
            else:
                res += [np.mean(ta), np.mean(srn)]

            rese += [np.array(ta), np.array(srn)]

        # print(np.array(rese).shape)

        res_ += [res]
        res_e += [rese]

    res_ = np.array(res_)

    # gres_e = np.array(res_e)
    assert stabilizers[0] == "Baseline"

    best = "max"
    target_ids = eval(f"np.arg{best}")(res_.T, axis=1)

    res_2 = copy.copy(res_)
    bt_ = []
    for i, h in enumerate(stabilizers):
        bt = []

        # below TA flag
        # 0: None below TA
        # 1: Some
        # 2: All

        for j, x in enumerate(res_e[i]):
            # test accuracy
            if j % 2 == 0:
                if all(x + ta_threshold > res_[0][j]):
                    # if all(x > res_[0][j] * 0.98):
                    below_ta = 0
                elif any(x + ta_threshold > res_[0][j]):
                    # elif any(x > res_[0][j] * 0.98):
                    below_ta = 1
                    print(res_[0][j])
                else:
                    below_ta = 2
                    res_2[i][j] = 0
                bt += [below_ta, below_ta]
            # relu accuracy
            else:
                if below_ta == 2:
                    res_2[i][j] = 0

        bt_ += [bt]

    target_ids = []
    for x in res_2.T:
        i = eval(f"np.arg{best}")(x)
        target_ids += [np.where(x == x[i])[0].tolist()]

    for i, h in enumerate(stabilizers):
        line = h.replace("_", "\_")
        """
        for j, x in enumerate(res_[i]):
            if bt_[i][j] == 0:
                line += " &"
            elif bt_[i][j] == 1:
                line += " &\\sout{"
            elif bt_[i][j] == 2:
                line += " &\\xcancel{"

            if x == 0:
                line += " -"
            elif i in target_ids[j]:
                line += f" \\tb{x:.2f}"
            else:
                line += f" {x:.2f}"
                # if h == "P_SDD" and j == 4:
                #    print(res_e[i][j])

            if bt_[i][j] == 0:
                line += " "
            elif bt_[i][j] == 1:
                line += "} "
            elif bt_[i][j] == 2:
                line += "} "
        """
        for j, x in enumerate(res_[i]):
            line += " &"
            if x in [-1, 0]:
                line += " - "
            elif i in target_ids[j]:
                line += f" \\tb{x:.2f}"
            else:
                line += f" {x:.2f}"

        print(f"{line}\\\\")
        if h == "Baseline" or i % 6 == 0:
            print("\\hline")

    line = "Best "
    for i, x in enumerate(np.argmax(res_2.T, axis=1)):
        y = res_.T[i][x]
        # print(i, x, y, res_.T[i][0], y / res_.T[i][0])
        if i % 2 == 0:
            if y - res_.T[i][0] >= 0:
                line += f"& +{y-res_.T[i][0]:.2f} "
            else:
                line += f"& {y-res_.T[i][0]:.2f} "
        else:
            line += f"& {y/res_.T[i][0]:.2f} "
    line += " \\\\"
    print(line)
    print("\\hline")
    print()


def verification_table(df, stabilizers, ta_threshold):  # , excludes):
    # artifacts = [*reversed(sorted(set(df["artifact"])))]
    artifacts = list(set(df["network"]))
    artifacts.sort(key=Settings.network_order.index)
    #print(artifacts)

    def verification_table(
        metric,
        normalize,
        best,
        worst_case,
        ta_threshold,
        target_ids=None,
        zero_ids=None,
        zero_ids_flag=False,
    ):
        if best == "min":
            exclude_case = worst_case + 1
        elif best == "max":
            exclude_case = worst_case - 1
        print()
        print(f"{metric}:")
        res_ = []
        seeds = len(set(df["seed"]))

        verifiers = list(set(df["verifier"]))
        verifiers.sort(key=Settings.verifier_order.index)
        print("Verifiers: ",verifiers)
        dft = df
        dft = dft[dft["verifier"] == verifiers[0]]
        # dft = dft[dft["artifact"] == artifacts[0]]
        dft = dft[dft["artifact"] == artifacts[0]]
        dft = dft[dft["stabilizer"] == "Baseline"]
        nb_problems = len(dft)

        left_problems = []

        for h in stabilizers:
            res = []
            left_p = []
            tt = 0
            for v in verifiers:
                total = 0
                for a in artifacts:
                    # exclude bad examples

                    # if h in excludes and a in excludes[h]:
                    #    X = exclude_case

                    # calculate metric for good samples

                    dft = df
                    dft = dft[dft["verifier"] == v]
                    # dft = dft[dft["artifact"] == a]
                    dft = dft[dft["network"] == a]
                    df_baseline = dft[dft["stabilizer"] == "Baseline"]
                    ta_baseline = np.mean(df_baseline["test accuracy"].values)
                    # print(f"{ta_baseline:.4f}")
                    dft = dft[dft["stabilizer"] == h]

                    # relative test accuracy
                    nb_problems_total = len(dft)

                    dft = dft[dft["test accuracy"] >= ta_baseline - ta_threshold / 100]
                    # dft = dft[dft["test accuracy"] >= ta_baseline * 0.98]

                    nb_problems_meet_ta = len(dft)
                    nb_problems_excluded = nb_problems_total - nb_problems_meet_ta

                    # best_ta = sorted(set(dft["test accuracy"].values))[-1]
                    # dft = dft[dft["test accuracy"] == best_ta]

                    veri_ans = dft["veri ans"].values

                    veri_time = dft["veri time"].values
                    if metric == "scr":
                        X = sum([1 for x in veri_ans if x in [1, 2]])
                        # X = sum([1 for x in veri_ans if x in [1]])
                    # print(len(dft), X)
                    elif metric == "time":
                        if len(veri_time) == 0:
                            X = float("inf")
                        else:
                            X = np.mean(veri_time)

                    elif metric == "par2":
                        assert len(veri_ans) == len(veri_time)
                        par2_a = sum(
                            [
                                veri_time[i]
                                for i in range(len(veri_ans))
                                if veri_ans[i] in [1, 2]
                            ]
                        )
                        par2_b = (
                            sum(
                                [
                                    Settings.timeout * 2
                                    for i in range(len(veri_ans))
                                    if veri_ans[i] not in [1, 2]
                                ]
                            )
                            + nb_problems_excluded * Settings.timeout * 2
                        )
                        X = (par2_a + par2_b) / 1000

                    else:
                        assert False
                    total += X

                    res += [X]
                    left_p += [nb_problems_meet_ta]

                # res += [total]
                tt += total
            if metric != "time":
                res = [x / len(set(df[df["stabilizer"] == h]["seed"])) for x in res]
            res_ += [res]
            left_problems += [left_p]
            # res_ += [res + [tt]]

        # res_ = np.array(res_) / seeds
        res_ = np.array(res_)

        if metric == "scr":
            baseline = res_[0]
            # target_ids = eval(f"np.arg{best}")(res_.T, axis=1)
            target_ids = []
            zero_ids = []

            for i, x in enumerate(res_.T):
                temp = np.argsort(x)[-3:]
                ids = []
                for j in temp:
                    if x[j] >= baseline[i] and j != 0:
                        ids += [j]
                zeros = []
                for j, x_ in enumerate(x):
                    if x_ == 0:
                        zeros += [j]

                target_ids += [ids]
                zero_ids += [zeros]

            #print(target_ids)

        assert stabilizers[0] == "Baseline"
        for i, row in enumerate(res_):
            h_name = stabilizers[i].replace("_", "\_")
            line = f"{h_name} & "

            for j, x in enumerate(row):
                if left_problems[i][j] == 0:
                    line += "- & "
                elif x == 0:
                    line += f"{x:,.0f}& "
                elif zero_ids_flag and i in zero_ids[j]:
                    line += "- &"
                elif i in target_ids[j]:
                    line += f"\\tb{x:,.2f}& "
                else:
                    line += f"{x:,.2f}& "

            print(f"{line[:-2]} \\\\")
            if h == "Baseline" or i % 6 == 0:
                print("\\hline")
        print()

        line += " \\\\"


        return target_ids, zero_ids

    target_ids, zero_ids = verification_table("scr", False, "max", 0, ta_threshold)
    verification_table(
        "time",
        False,
        "min",
        0,
        ta_threshold,
        target_ids=target_ids,
        zero_ids=zero_ids,
        zero_ids_flag=True,
    )
    # verification_table("par2", False, "min", 0, ta_threshold)
    # verification_table('time', False, 'min')
    # verification_table('par2', False, 'min')

    # verification_table('src', True, 'max')
    # verification_table("time", True, "min", excludes, 45000)
    # verification_table("par2", True, "min", excludes, 90000)


def verification_plot(df, stabilizers, excludes):
    verifiers = sorted(list(set(df["verifier"])))
    colors = [(0, 0, 0)] + sns.color_palette("hls", len(stabilizers) - 1)

    for v in verifiers:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
        print(v)
        X_ = {}
        for i, h in enumerate(stabilizers):
            dft = df[df["verifier"] == v]
            dft = dft[dft["stabilizer"] == h]

            if h in excludes:
                for x in excludes[h]:
                    dft = dft[dft["artifact"] != x]

            X = []
            for y in np.arange(1, 601, 1):
                temp = dft[dft["veri time"] < y]
                temp1 = temp[temp["veri ans"] == 1]
                temp2 = temp[temp["veri ans"] == 2]
                X += [len(temp1) + len(temp2)]
            print(f"{h}\t{X[-1]}")
            X_[h] = X

        temp = [(x, X_[x][-1]) for x in X_]
        temp = sorted(temp, key=lambda x: x[1])
        best_heu = temp[-1][0]

        for i, h in enumerate(X_):
            if h == "Baseline":
                linewidth = 2
                linestyle = ":"
            elif h == best_heu:
                linewidth = 2
                linestyle = "--"
            else:
                linewidth = 1
                linestyle = "-"

            X = X_[h]
            ax1.plot(
                X,
                np.arange(len(X)) + 1,
                linewidth=linewidth,
                linestyle=linestyle,
                label=h,
                color=(*colors[i], 2 / 3),
            )
        ax1.legend(
            loc="upper left",
            bbox_to_anchor=(1, 1),
            fancybox=True,
        )
        ax1.set_xlabel("Number of Instances Verified")
        ax1.set_ylabel("Time (s)")
        ax1.set_ylim(0, Settings.timeout)
        # ax1.set_xlim(0,len(dft))
        # ax1.set_xticks(range(len(dft)))
        # ax1.set_xticklabels(self.epsilons)

        # plt.title(title_pr + " All Instances")
        plt.savefig(
            os.path.join(Settings.fig_dir, f"V_{v}.pdf"),
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
        )
        fig.clear()
        plt.close(fig)


def plot_baseline(df):
    df = df[df["stabilizer"] == "Baseline"]
    verifiers = set(df["verifier"])
    eps = sorted(set(df["epsilon"]))

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
    for v in verifiers:
        dft = df[df["verifier"] == v]
        nb_unsat = []
        nb_sat = []
        for e in eps:
            dft_ = dft[dft["epsilon"] == e]
            unsat = len([x for x in dft_["veri ans"] if x == 1])
            sat = len([x for x in dft_["veri ans"] if x == 2])
            nb_unsat += [unsat]
            nb_sat += [sat]

        print(v, nb_unsat, nb_sat)

        ax1.plot(
            # np.arange(len(nb_unsat)) + 1,
            eps,
            nb_unsat,
            linewidth=1,
            linestyle="--",
            label="UNSAT",
            color="blue",
        )

        ax1.plot(
            # np.arange(len(nb_sat)) + 1,
            eps,
            nb_sat,
            linewidth=1,
            linestyle=":",
            label="SAT",
            color="orange",
        )
        ax1.legend(
            # loc="upper right",
            bbox_to_anchor=(1, 1),
            fancybox=True,
        )
        ax1.set_xlabel("Epsilon")
        ax1.set_ylabel("Instances Proved/Falsified")
        ax1.set_ylim(-1, len(dft_) + 1)
        pathlib.Path(Settings.fig_dir).mkdir(parents=True, exist_ok=True)

        artifact = f'{list(set(dft_["artifact"]))[0]}:{list(set(dft_["network"]))[0]}'
        plt.savefig(
            os.path.join(Settings.fig_dir, f"A={artifact}_V={v}.pdf"),
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
        )
        ax1.clear()
        # plt.close(fig)


def refine_df(df, stabilizers):
    networks = list(set(df["network"]))
    networks.sort(key=Settings.network_order.index)
    verifiers = list(set(df["verifier"]))
    verifiers.sort(key=Settings.verifier_order.index)
    nb_seeds = len(list(set(df["seed"])))

    print(networks)
    print(verifiers)

    labels = [
        "Verifier",
        "Network",
        "Stabilizer",
        "Approach",
        "Estimator",
        "Test Accuracy(%)",
        "Problem Solved",
        "Speedup",
        "Stable Neurons(%)",
        "Normalized Training Time",
    ]

    df2 = pd.DataFrame({x: [] for x in labels})

    for v in verifiers:
        for n in networks:
            for s in stabilizers:
                dft = df[df["verifier"] == v]
                dft = dft[dft["network"] == n]
                dft_base = dft[dft["stabilizer"] == "Baseline"]
                base_time = np.mean(dft_base["veri time"])
                base_training_time = np.mean(dft_base["training time"])
                dft = dft[dft["stabilizer"] == s]

                if s == "Baseline":
                    method = "Baseline"
                    estimator = "Baseline"
                else:
                    method = s.split("_")[0]
                    estimator = s.split("_")[1]

                test_acc = np.mean(dft["test accuracy"]) * 100
                solved = len(dft[dft["veri ans"] <= 2]) / nb_seeds
                srn = dft["nb stable"] / (dft["nb stable"] + dft["nb unstable"])
                srn = np.mean(srn) * 100
                veri_time = np.mean(dft["veri time"])

                train_time = np.mean(dft["training time"]) / np.mean(base_training_time)

                speedup = base_time / veri_time

                df2.loc[len(df2.index)] = [
                    v,
                    n,
                    s,
                    method,
                    estimator,
                    test_acc,
                    solved,
                    speedup,
                    srn,
                    train_time,
                ]
    return df2


def scatter_plot(df, stabilizers, mode="seperated"):
    sns.set(rc={"figure.figsize": (8, 8)})
    networks = list(set(df["network"]))
    networks.sort(key=Settings.network_order.index)
    verifiers = list(set(df["verifier"]))
    verifiers.sort(key=Settings.verifier_order.index)

    df = refine_df(df, stabilizers)

    cs = sns.color_palette("bright", 6) + [(0, 0, 0)]

    plot_all(df, verifiers, networks, cs, hl=False)
    plot_all(df, verifiers, networks, cs, hl=True)


def plot_all(df, verifiers, networks, cs, ms=100, hl=False):
    print("Plotting per verifier and network cominations ...")
    for v in verifiers:
        for i, n in enumerate(networks):
            dft = df[df["Verifier"] == v]
            dft = dft[dft["Network"] == n]
            pr = "[VN]"
            po = f"_{v}_{n}"
            hy = i != 0
            hy = True

            # plot_TA_SPVS(dft, po, ms=ms, hl=hl)
            plot_TA_SP(dft, cs, pr, po, ms=ms, hl=hl, hy=hy)
            plot_TA_VS(dft, cs, pr, po, ms=ms, hl=hl, type="bar", hy=hy)
            plot_TA_VS(dft, cs, pr, po, ms=ms, hl=hl, type="scatter", hy=hy)
            plot_TA_VS(
                dft, cs, pr, po, ms=ms, hl=hl, type="scatter", scale="log", hy=hy
            )
            plot_TA_SN(dft, cs, pr, po, ms=ms, hl=hl, hy=hy)

    print("Plotting per verifier ...")
    for v in verifiers:
        dft = df[df["Verifier"] == v]
        pr = "[V]"
        po = f"_{v}"
        # plot_TA_SPVS(dft, po, ms=ms, hl=hl)
        plot_TA_SP(dft, cs, pr, po, ms=ms, hl=hl)
        plot_TA_VS(dft, cs, pr, po, ms=ms, hl=hl, type="bar")
        plot_TA_VS(dft, cs, pr, po, ms=ms, hl=hl, type="scatter")
        plot_TA_VS(dft, cs, pr, po, ms=ms, hl=hl, type="scatter", scale="log")
        plot_TA_SN(dft, cs, pr, po, ms=ms, hl=hl)

    print("Plotting per network ...")
    for i, n in enumerate(networks):
        dft = df[df["Network"] == n]
        pr = "[N]"
        po = f"_{n}"
        hy = i != 0

        # plot_TA_SPVS(dft, po, ms=ms, hl=hl)
        plot_TA_SP(dft, cs, pr, po, ms=ms, hl=hl, hy=hy)
        # plot_TA_VS(dft, cs, pr, po, ms=ms, hl=hl)
        plot_TA_VS(dft, cs, pr, po, ms=ms, hl=hl, type="bar", hy=hy)
        plot_TA_VS(dft, cs, pr, po, ms=ms, hl=hl, type="scatter", hy=hy)
        plot_TA_VS(dft, cs, pr, po, ms=ms, hl=hl, type="scatter", scale="log", hy=hy)
        plot_TA_SN(dft, cs, pr, po, ms=ms, hl=hl, hy=hy)
        plot_TT(dft, cs, pr, po, ms=ms, hl=hl, scale="log", hy=hy)

    print("Plotting all ...")
    dft = df
    pr = "[A]"
    po = ""

    plot_TA_SP(dft, cs, pr, po, ms=ms, hl=hl)
    plot_TA_VS(dft, cs, pr, po, ms=ms, hl=hl, type="bar")
    plot_TA_VS(dft, cs, pr, po, ms=ms, hl=hl, type="scatter")
    plot_TA_VS(dft, cs, pr, po, ms=ms, hl=hl, type="scatter", scale="log")
    plot_TA_SN(dft, cs, pr, po, ms=ms, hl=hl)
    plot_TT(dft, cs, pr, po, ms=ms, hl=hl, scale="log")


# TA vs Solved Problems and Verification Speed
def plot_TA_SPVS(dft, colors, pr, po, ms=100, hl=False):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    sns.scatterplot(
        ax=ax[0],
        data=dft,
        x="Test Accuracy(%)",
        y="Problem Solved",
        style="Approach",
        hue="Estimator",
        palette=colors,
        alpha=0.8,
    )

    sns.barplot(
        ax=ax[1],
        data=dft,
        x="Approach",
        y="Speedup",
        hue="Estimator",
        palette=colors,
        alpha=0.8,
    )
    ax[0].collections[0].set_sizes([ms])

    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=8)
    if hl:
        ax[0].get_legend().remove()
        ax[1].get_legend().remove()
        po += "_"
    fig.tight_layout()

    # plt.suptitle(f"Verification Performance of {v} on {n}")
    plt.savefig(
        os.path.join(
            Settings.fig_dir,
            f"{pr}SPVS{po}.pdf".replace("α", "a").replace("β", "b"),
        ),
        format="pdf",
        bbox_inches="tight",
        pad_inches=0,
    )
    fig.clear()
    plt.close(fig)


# TA vs Solved Problems
def plot_TA_SP(dft, colors, pr, po, ms=100, hl=False, hy=False):
    # TA vs Solved Problems
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    sns.scatterplot(
        ax=ax,
        data=dft,
        x="Test Accuracy(%)",
        y="Problem Solved",
        style="Approach",
        hue="Estimator",
        palette=colors,
        alpha=0.8,
    )

    ax.collections[0].set_sizes([ms])
    ax.legend(fontsize=9, markerscale=1.5)
    if hy:
        ax.set_ylabel(None)
    if hl:
        ax.get_legend().remove()
        po += "_"

    fig.tight_layout()

    # plt.suptitle(f"Verification Performance of {v} on {n}")
    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    plt.savefig(
        os.path.join(
            Settings.fig_dir,
            f"{pr}SP{po}.pdf".replace("α", "a").replace("β", "b"),
        ),
        format="pdf",
        bbox_inches="tight",
        pad_inches=0,
    )
    fig.clear()
    plt.close(fig)


# TA vs Verification Speed
def plot_TA_VS(
    dft, colors, pr, po, ms=100, hl=False, type=None, scale="linear", hy=False
):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    if type == "bar":
        dft = copy.deepcopy(dft)
        dft.drop(index=dft[dft["Stabilizer"] == "Baseline"].index, inplace=True)
        colors = colors[:-1]
        sns.barplot(
            ax=ax,
            data=dft,
            x="Approach",
            y="Speedup",
            hue="Estimator",
            palette=colors,
            alpha=0.8,
        )
        plt.axhline(y=1, color="black", linestyle=":", linewidth=0.5, label="Baseline")

    elif type == "scatter":
        sns.scatterplot(
            ax=ax,
            data=dft,
            x="Test Accuracy(%)",
            y="Speedup",
            style="Approach",
            hue="Estimator",
            palette=colors,
            alpha=0.8,
        )
        ax.collections[0].set_sizes([ms])
        ax.legend(fontsize=9, markerscale=1.5)
        po += "_s"

    ax.set_yscale(scale)
    ax.legend(fontsize=9)
    if scale == "log":
        po += "_l"
    if hy:
        ax.set_ylabel(None)
    if hl:
        ax.get_legend().remove()
        po += "_"

    fig.tight_layout()
    # plt.suptitle(f"Verification Performance of {v} on {n}")
    plt.savefig(
        os.path.join(
            Settings.fig_dir,
            f"{pr}VS{po}.pdf".replace("α", "a").replace("β", "b"),
        ),
        format="pdf",
        bbox_inches="tight",
        pad_inches=0,
    )
    fig.clear()
    plt.close(fig)


# TA vs Stable Neurons
def plot_TA_SN(dft, colors, pr, po, ms=100, hl=False, hy=False):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    sns.scatterplot(
        ax=ax,
        data=dft,
        x="Test Accuracy(%)",
        y="Stable Neurons(%)",
        style="Approach",
        hue="Estimator",
        palette=colors,
        alpha=0.8,
    )

    ax.collections[0].set_sizes([ms])
    ax.legend(fontsize=9, markerscale=1.5)
    if hy:
        ax.set_ylabel(None)
    if hl:
        ax.get_legend().remove()
        po += "_"
    fig.tight_layout()
    # plt.suptitle(f"Verification Performance of {v} on {n}")
    plt.savefig(
        os.path.join(
            Settings.fig_dir,
            f"{pr}SN{po}.pdf".replace("α", "a").replace("β", "b"),
        ),
        format="pdf",
        bbox_inches="tight",
        pad_inches=0,
    )

    fig.clear()
    plt.close(fig)


# Training time
def plot_TT(dft, colors, pr, po, ms=100, hl=False, scale="linear", hy=False):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    dft = copy.deepcopy(dft)
    dft.drop(index=dft[dft["Stabilizer"] == "Baseline"].index, inplace=True)
    colors = colors[:-1]

    sns.barplot(
        ax=ax,
        data=dft,
        x="Approach",
        y="Normalized Training Time",
        hue="Estimator",
        palette=colors,
        alpha=0.8,
    )

    plt.axhline(y=1, color="black", linestyle=":", linewidth=0.5, label="Baseline")

    ax.set_yscale(scale)
    ax.legend(fontsize=9)
    if scale == "log":
        po += "_l"
    ax.legend(fontsize=9)
    if hy:
        ax.set_ylabel(None)
    if hl:
        ax.get_legend().remove()
        po += "_"
    fig.tight_layout()
    # plt.suptitle(f"Verification Performance of {v} on {n}")
    plt.savefig(
        os.path.join(
            Settings.fig_dir,
            f"{pr}TT{po}.pdf".replace("α", "a").replace("β", "b"),
        ),
        format="pdf",
        bbox_inches="tight",
        pad_inches=0,
    )
    fig.clear()
    plt.close(fig)


def plot_eps(df, stabilizers):
    networks = list(set(df["network"]))
    networks.sort(key=Settings.network_order.index)
    verifiers = list(set(df["verifier"]))
    verifiers.sort(key=Settings.verifier_order.index)
    verifiers = verifiers[1:]
    nb_seeds = len(list(set(df["seed"])))
    epsilons = [float(x) for x in list(set(df["epsilon"]))]

    # print(networks)
    print(verifiers)
    print(epsilons)
    labels = ["Verifier", "ε", "UNSAT", "SAT", "Verification Time(s)"]

    df2 = pd.DataFrame({x: [] for x in labels})

    for v in verifiers:
        for s in stabilizers:
            for e in epsilons:
                dft = df[df["verifier"] == v]
                dft = dft[dft["epsilon"] == e]
                unsat = len(dft[dft["veri ans"] == 1]) / nb_seeds
                sat = len(dft[dft["veri ans"] == 2]) / nb_seeds
                veri_time = np.mean(dft["veri time"])
                df2.loc[len(df2.index)] = [v, e, unsat, sat, veri_time]

    colors = sns.color_palette("husl", len(verifiers))

    fig, ax1 = plt.subplots(1, 1, figsize=(5, 3.5))
    ax2 = plt.twinx()

    ms = 100
    sns.lineplot(
        ax=ax1,
        data=df2,
        x="ε",
        y="UNSAT",
        hue="Verifier",
        palette=colors,
        linewidth=2,
        alpha=1,
        legend="auto",
    )
    for x in ax1.lines[:3]:
        x.set_linestyle("--")

    sns.lineplot(
        ax=ax1,
        data=df2,
        x="ε",
        y="SAT",
        hue="Verifier",
        palette=colors,
        linewidth=2,
        alpha=1,
        legend=None,
    )
    for x in ax1.lines[6:]:
        x.set_linestyle(":")

    sns.lineplot(
        ax=ax2,
        data=df2,
        x="ε",
        y="Verification Time(s)",
        hue="Verifier",
        palette=colors,
        linewidth=2,
        alpha=0.2,
        legend=None,
    )

    ax1.collections[0].set_sizes([ms])

    ax1.legend(
        bbox_to_anchor=(0.97, 0.05), loc="lower right", fontsize=9, markerscale=1.5
    )
    ax1.grid(color="grey", linestyle=(0, (1, 5)), linewidth=0.2)

    ax1.set_xlabel("Radii(ε)")
    ax1.set_ylabel("Verified(--)/Falsified(··)")
    ax2.set_ylabel("Verification Time(s, —)")

    fig.tight_layout()

    # plt.suptitle(f"Verification Performance of {v} on {n}")
    plt.savefig(
        os.path.join(
            Settings.fig_dir,
            f"eps.pdf".replace("α", "a").replace("β", "b"),
        ),
        format="pdf",
        bbox_inches="tight",
        pad_inches=0,
    )

    fig.clear()
    plt.close(fig)


if __name__ == "__main__":
    main(_parse_args())
