#!/usr/bin/env python

import os
import argparse
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from nexus import Settings


def _parse_args():
    parser = argparse.ArgumentParser(description="Plotter")
    parser.add_argument("study", type=str)
    parser.add_argument("task", type=str)
    parser.add_argument("--root", type=str, default="results")
    args = parser.parse_args()
    return args


def main(args):
    heuristics = Settings.heuristics[args.study.replace("_", "")]

    combine(args)
    df = pd.read_feather(os.path.join(args.root, f"{args.study}.feather"))

    if args.task == "st":
        stable_relu_table(df, heuristics, Settings.ta_threshold)
    elif args.task == "vt":
        verification_table(df, heuristics)
    elif args.task == "vp":
        verification_plot(df, heuristics)
    else:
        assert False


def combine(args):
    study = args.study
    if study == "e1":
        studies = [f"e1p{x}" for x in [1, 2, 3]]
        # studies = [f"e1p{x}" for x in [1, 2]]
        studies = [f"e1p1v3", "e1p2v3"]
    elif study == "e1_":
        studies = [f"e1p{x}" for x in [1, 2, "3_"]]
    elif study in ["e2", "e2_"]:
        p1 = [f"e2p{x}a" for x in [1, 2, 3]]
        p2 = [f"e2p{x}b" for x in [1, 2, 3]]
        studies = p1 + p2
    elif study in "e3":
        p1 = [f"e3p{x}a" for x in [1, 2, 3]]
        p2 = [f"e3p{x}b" for x in [1, 2, 3]]
        studies = p1 + p2

    dfs = []
    for s in studies:
        df = pd.read_feather(os.path.join(args.root, f"{s}.feather"))
        dfs += [df]

    if study in ["e2", "e2_", "e3", "e3_"]:
        df = pd.read_feather(os.path.join(args.root, "e1/e1.feather"))
        df = df[df["heuristic"] == "Baseline"]
        dfs += [df]

    df_ = pd.concat(dfs, ignore_index=True)
    df_["heuristic"] = df_["heuristic"].map(Settings.convert_names)

    save_path = os.path.join(args.root, f"{study}.feather")
    df_.to_feather(save_path)


def stable_relu_table(df, heuristics, ta_threshold):
    artifacts = [*reversed(sorted(set(df["artifact"])))]
    res_ = {}
    for i, h in enumerate(heuristics):
        res = []
        for a in artifacts:
            dft = df[df["verifier"] == "DNNVWB:neurify"]
            dft = dft[dft["artifact"] == a]
            dft = dft[dft["heuristic"] == h]
            ta = dft["test accuracy"]
            ta = np.mean(ta) * 100

            srn = dft["relu accuracy veri"]
            srn = np.mean(srn) * 100
            res += [ta, srn]
        res_[h] = res

    for i, h in enumerate(heuristics):
        line = h.replace("_", "\_")
        for j, x in enumerate(res_[h]):
            if j % 2 == 0:
                # print(x)
                # exit()
                if res_["Baseline"][j] - x <= ta_threshold:
                    line += f" & {x:.2f}"
                else:
                    line += f" & \\tb{x:.2f}"
            else:
                line += f" & {x:.2f}"
        print(f"{line}\\\\")
        if h == "Baseline" or i % 4 == 0:
            print("\\hline")
    print()


def stable_relu_table_detailed(df, heuristics):
    artifacts = [*reversed(sorted(set(df["artifact"])))]

    for a in artifacts:
        print(a)
        for i, h in enumerate(heuristics):
            dft = df[df["verifier"] == "DNNVWB:neurify"]
            dft = dft[dft["artifact"] == a]
            dft = dft[dft["heuristic"] == h]
            ta = dft["test accuracy"]
            ta = np.mean(ta) * 100

            sr1 = np.mean(dft["ra(sdd)"]) * 100
            sr2 = np.mean(dft["ra(sad)"]) * 100
            sr3 = np.mean(dft["ra(nip)"]) * 100
            sr4 = np.mean(dft["ra(sip)"]) * 100
            srn = np.mean(dft["relu accuracy veri"]) * 100
            h_name = h.replace("_", "\_")
            print(
                f"{h_name} & {ta:.2f} & {sr1:.2f} & {sr2:.2f} & {sr3:.2f} & {sr4:.2f} & {srn:.2f} \\\\"
            )

            if h == "Baseline" or i % 4 == 0:
                print("\\hline")
        print()


def verification_table(df, heuristics):  # , excludes):
    artifacts = [*reversed(sorted(set(df["artifact"])))]

    def verification_table(metric, normalize, best, worst_case):
        if best == "min":
            exclude_case = worst_case + 1
        elif best == "max":
            exclude_case = worst_case - 1
        print(f"{metric}:")
        res_ = []
        for h in heuristics:
            res = []
            for v in Settings.verifiers:

                total = 0
                for a in artifacts:
                    # exclude bad examples

                    # if h in excludes and a in excludes[h]:
                    #    X = exclude_case

                    # calculate metric for good samples

                    dft = df
                    dft = dft[dft["verifier"] == v]
                    dft = dft[dft["artifact"] == a]
                    df_baseline = dft[dft["heuristic"] == "Baseline"]
                    ta_baseline = np.mean(df_baseline["test accuracy"].values)
                    print(f"{ta_baseline:.4f}")
                    dft = dft[dft["heuristic"] == h]

                    # relative test accuracy
                    dft = dft[
                        dft["test accuracy"]
                        >= (ta_baseline - Settings.ta_threshold / 100)
                    ]

                    # best_ta = sorted(set(dft["test accuracy"].values))[-1]
                    # dft = dft[dft["test accuracy"] == best_ta]

                    print(len(dft))

                    veri_ans = dft["veri ans"].values
                    veri_time = dft["veri time"].values
                    if metric == "scr":
                        X = sum([1 for x in veri_ans if x in [1, 2]])

                    elif metric == "time":
                        X = sum(veri_time)

                    elif metric == "par2":
                        assert len(veri_ans) == len(veri_time)
                        par2_a = sum(
                            [
                                veri_time[i]
                                for i in range(len(veri_ans))
                                if veri_ans[i] in [1, 2]
                            ]
                        )
                        par2_b = sum(
                            [
                                Settings.timeout * 2
                                for i in range(len(veri_ans))
                                if veri_ans[i] not in [1, 2]
                            ]
                        )
                        X = par2_a + par2_b

                    else:
                        assert False
                    total += X

                    res += [X]

                res += [total]

            res_ += [res]

        res_ = np.array(res_)

        target_ids = eval(f"np.arg{best}")(res_.T, axis=1)

        target_ids = []
        for x in res_.T:
            i = eval(f"np.arg{best}")(x)
            if x[i] == worst_case:
                target_ids += [[]]
            else:
                target_ids += [np.where(x == x[i])[0].tolist()]

        assert heuristics[0] == "Baseline"
        for i, row in enumerate(res_):

            h_name = heuristics[i].replace("_", "\_")
            line = f"{h_name} &"

            for j, x in enumerate(row):

                if x == exclude_case:
                    line += " $\\times$ &"
                else:
                    if normalize:
                        if res_[0][j] == 0:
                            line += f" N/A &"
                        else:
                            x /= res_[0][j]
                            if i in target_ids[j]:
                                line += f" \\tb{x:,.2f} &"
                            else:
                                line += f" {x:,.2f} &"

                    else:
                        if i in target_ids[j]:
                            line += f" \\tb{x:,.0f} &"
                        else:
                            line += f" {x:,.0f} &"

            print(f"{line[:-1]} \\\\")
            if h == "Baseline" or i % 4 == 0:
                print("\\hline")
        print()

    verification_table("scr", False, "max", 0)
    # verification_table('time', False, 'min')
    # verification_table('par2', False, 'min')

    # verification_table('src', True, 'max')
    # verification_table("time", True, "min", excludes, 45000)
    # verification_table("par2", True, "min", excludes, 90000)


def verification_plot(df, heuristics, excludes):
    verifiers = sorted(list(set(df["verifier"])))
    colors = [(0, 0, 0)] + sns.color_palette("hls", len(heuristics) - 1)

    for v in verifiers:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
        print(v)
        X_ = {}
        for i, h in enumerate(heuristics):
            dft = df[df["verifier"] == v]
            dft = dft[dft["heuristic"] == h]

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

        # plt.title(title_prefix + " All Instances")
        plt.savefig(
            os.path.join(Settings.fig_dir, f"V_{v}.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
        fig.clear()
        plt.close(fig)


if __name__ == "__main__":
    main(_parse_args())
