import os
import copy
import time
import toml
import uuid
import importlib
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm

from octopus.core.problem import Problem
from octopus.plot.train_progress import ProgressPlot
from octopus.plot.box_plot import colored_box_plot


from swarm_host.core.problem import VerificationProblem


from matplotlib.ticker import AutoMinorLocator, MultipleLocator


class Benchmark:
    def __init__(self, base_settings, benchmark_settings, **kwargs):
        self.go = kwargs["go"]
        self.slurm = kwargs["slurm"]
        self.override = kwargs["override"]
        self.logger = kwargs["logger"]
        with open(base_settings, "r") as fp:
            self.base_settings = toml.load(fp)
        self.result_dir = os.path.join(kwargs["result_dir"], self.base_settings["name"])
        self.cached_res_path = os.path.join(
            kwargs["result_dir"],
            benchmark_settings.split(".")[0].split("/")[-1] + ".feather",
        )

        self._setup_(benchmark_settings)
        self._define_problems()

    def _setup_(self, benchmark_settings):
        self.code_veri_answer = {
            "unsat": 1,
            "sat": 2,
            "unknown": 3,
            "timeout": 4,
            "memout": 4,
            "error": 5,
        }

        self.labels = [
            "artifact",
            "network",
            "stabilizer",
            "seed",
            "property",
            "epsilon",
            "verifier",
            "test accuracy",
            "veri ans",
            "veri time",
            "training time",
            # "relu accuracy veri",
        ]

        self.sub_dirs = {}
        sub_dirs = [
            "train_config",
            "train_log",
            "model",
            "veri_config",
            "veri_log",
            "property",
            "figure",
        ]

        if self.slurm:
            sub_dirs += ["train_slurm", "veri_slurm"]
            if "save_log" in self.base_settings["train"]:
                self.logger.info(
                    "Disabled [save_log] and redirected training logs to slurm logging."
                )
                self.base_settings["train"]["save_log"] = False
            if "save_log" in self.base_settings["verify"]:
                self.logger.info(
                    "Disabled [save_log] and redirected verification logs to slurm logging."
                )
                self.base_settings["verify"]["save_log"] = False

        for sd in sub_dirs:
            sdp = os.path.join(self.result_dir, sd)
            Path(sdp).mkdir(exist_ok=True, parents=True)
            self.sub_dirs[sd + "_dir"] = sdp

        self.logger.info("Reading benchmark settings ...")
        module = importlib.import_module(
            benchmark_settings.split(".")[0].replace("/", ".")
        )
        for x in module.__dict__:
            if not x.startswith("__"):
                self.__setattr__(x, module.__dict__[x])

    def _define_problems(self):
        self.logger.info("Configuring problems ...")
        self.problems_T = []
        self.problems_T_hash = []
        self.problems_V = []
        self.problems_V_hash = []

        for a in self.artifacts:
            for n in self.networks:
                for h in self.stabilizers:
                    for s in self.seeds:
                        for e in self.epsilons:
                            self.problems_T += [(a, n, h, s, e)]
                            for p in self.props:
                                for v in self.verifiers:
                                    self.problems_V += [(a, n, h, s, e, p, v)]

    def _exec(self, cmd, slurm_cmd, sleep_time):
        if not self.go:
            self.logger.info(f"Dry: {cmd}")
            if self.slurm:
                self.logger.info(f"Dry: {slurm_cmd}")
            exit(0)
        else:
            self.logger.info(f"Fly: {cmd}")
            if not self.slurm:
                os.system(cmd)
            else:
                self.logger.info(f"Fly: {slurm_cmd}")
                os.system(slurm_cmd)
                time.sleep(sleep_time)

    def _get_problem_paths(self, task, **kwargs):
        sts = copy.deepcopy(self.base_settings)

        a, n, h, s, e = list(kwargs.values())[:5]

        sts["train"]["artifact"] = a
        sts["train"]["net_name"] = n
        sts["train"]["net_layers"] = self.networks[n]
        sts["stabilizers"] = copy.deepcopy(self.stabilizers[h])

        if e == 0.1:  #
            te = 0.1  #
        else:  #
            te = int(e * 100 + 1) / 100  #
        # te = 0.02  ##

        if "stable_estimators" in sts["train"]:
            for est in sts["train"]["stable_estimators"]:
                sts["train"]["stable_estimators"][est]["epsilon"] = float(te)  #
        if sts["stabilizers"]:
            for x in sts["stabilizers"]:
                for est in sts["stabilizers"][x]["stable_estimators"]:
                    sts["stabilizers"][x]["stable_estimators"][est]["epsilon"] = float(
                        te  #
                    )

        name = Problem.Utility.get_hashed_name([sts["train"], sts["stabilizers"], s])

        log_path = os.path.join(self.sub_dirs["train_log_dir"], f"{name}.txt")

        if task == "T":
            config_path = os.path.join(
                self.sub_dirs["train_config_dir"], f"{name}.toml"
            )
            slurm_script_path = (
                None
                if not self.slurm
                else os.path.join(self.sub_dirs["train_slurm_dir"], f"{name}.slurm")
            )

        elif task == "V":
            a, n, h, s, e, p, v = kwargs.values()
            sts["verify"]["property"] = p
            sts["verify"]["epsilon"] = str(e)
            sts["verify"]["verifier"] = v

            # configure log path
            target_model = (
                sts["verify"]["target_model"]
                if "target_model" in sts["verify"]
                else None
            )

            target_epoch = Problem.Utility.get_target_epoch(target_model, log_path)
            target_epoch = sts["train"]["epochs"] if not target_epoch else target_epoch
            # veri_name_postfix = Problem.Utility.get_verification_postfix(sts["verify"])
            # veri_name = f"{name}_e={target_epoch}_{veri_name_postfix}"
            veri_name = Problem.Utility.get_hashed_name(
                [sts["train"], sts["stabilizers"], sts["verify"], target_epoch, s]
            )
            log_path = os.path.join(self.sub_dirs["veri_log_dir"], f"{veri_name}.txt")
            config_path = os.path.join(
                self.sub_dirs["veri_config_dir"], f"{veri_name}.toml"
            )
            slurm_script_path = (
                None
                if not self.slurm
                else os.path.join(self.sub_dirs["veri_slurm_dir"], f"{veri_name}.slurm")
            )

        # dump octopus configurations
        return sts, config_path, slurm_script_path, log_path, name

    def train(self):
        self.logger.info("Training ...")
        nb_done = 0
        nb_todo = 0
        for i, (a, n, h, s, e) in enumerate(tqdm(self.problems_T)):
            (
                sts,
                config_path,
                slurm_script_path,
                log_path,
                name,
            ) = self._get_problem_paths("T", a=a, n=n, h=h, s=s, e=e)

            # print(h, name, e, self.stabilizers[h])

            if name not in self.problems_T_hash:
                self.problems_T_hash += [name]
            else:
                self.logger.info("Skipping training job with existing configs ...")
                continue

            # check if done
            self.logger.debug(f"Train log: {log_path}")
            if os.path.exists(log_path) and not self.override:
                with open(log_path, "r") as log_fp:
                    log_lines = log_fp.readlines()
                trained = (
                    len([x for x in log_lines if "[Test] epoch: " in x])
                    == self.base_settings["train"]["epochs"]
                )
                if not trained:
                    print("rm", log_path)
                nb_done += 1
                continue
            else:
                nb_todo += 1
            with open(config_path, "w") as fp:
                toml.dump(sts, fp)
            cmd = f"python -m octopus {config_path} T --seed {s} --debug"
            if self.override:
                cmd += " --override"

            slurm_cmd = None
            if self.slurm:
                lines = [
                    "#!/bin/sh",
                    f"#SBATCH --job-name=O.T",
                    f"#SBATCH --output={log_path}",
                    f"#SBATCH --error={log_path}",
                ]
                if self.base_settings["train"]["gpu"]:
                    lines += [f"#SBATCH --partition=gnolim", "#SBATCH --gres=gpu:1"]
                if (
                    "train_nodes_ex" in self.__dict__
                    and "train_nodes" not in self.__dict__
                ):
                    lines += [f"#SBATCH --exclude={self.train_nodes_ex}"]

                lines += [
                    "cat /proc/sys/kernel/hostname",
                    "source .env.d/openenv.sh",
                    # "source .env.d/openenv_cheetah01.sh",
                    "which python3",
                    "echo $CUDA_VISIBLE_DEVICES",
                    cmd,
                ]

                lines = [x + "\n" for x in lines]
                open(slurm_script_path, "w").writelines(lines)
                param_node_nb = (
                    f"-c {self.nb_train_nodes}"
                    if "nb_train_nodes" in self.__dict__
                    else ""
                )

                param_node = (
                    f"-w {self.train_nodes[i%len(self.train_nodes)]}"
                    if "train_nodes" in self.__dict__ and self.train_nodes
                    else ""
                )
                slurm_cmd = f"sbatch {param_node_nb} {param_node} {slurm_script_path}"

            self._exec(cmd, slurm_cmd, self.sleep_train)

        self.logger.info(
            f"Tasks: done: {nb_done}, todo: {nb_todo}, total: {len(self.problems_T)}."
        )

    def test(self):
        self.logger.info("Testing ...")
        nb_done = 0
        nb_todo = 0
        for i, (a, n, h, s, e) in enumerate(tqdm(self.problems_T)):
            (
                sts,
                config_path,
                slurm_script_path,
                log_path,
                name,
            ) = self._get_problem_paths("T", a=a, n=n, h=h, s=s, e=e)
            toml.dump(sts, open(config_path, "w"))
            if name not in self.problems_T_hash:
                self.problems_T_hash += [name]
            else:
                self.logger.info("Skipping training job with existing configs ...")
                continue

            # check if done
            # print(log_path)
            assert os.path.exists(log_path) and not self.slurm

            cmd = f"python -m octopus {config_path} Test --seed {s} --debug"

            self._exec(cmd, None)

        self.logger.info(
            f"Tasks: done: {nb_done}, todo: {nb_todo}, total: {len(self.problems_T)}."
        )

    def verify(self):
        self.logger.info("Verifying ...")
        nb_done = 0
        nb_todo = 0
        for i, (a, n, h, s, e, p, v) in enumerate(tqdm(self.problems_V)):
            ###########################
            # SKIP untrained networks for faster verification on partial benchmarks
            (
                sts,
                config_path,
                _,
                _,
                name,
            ) = self._get_problem_paths("T", a=a, n=n, h=h, s=s, e=e)

            model_path = f'{config_path.replace("train_config", "model")[:-5]}.{sts["train"]["epochs"]}.onnx'
            if not os.path.exists(model_path):
                print(model_path)
                print("No model")
                continue
            ###########################

            sts, config_path, slurm_script_path, log_path, _ = self._get_problem_paths(
                "V", a=a, n=n, h=h, s=s, e=e, p=p, v=v
            )
            # check if done
            if os.path.exists(log_path) and not self.override:
                nb_done += 1
                continue
            else:
                nb_todo += 1
            with open(config_path, "w") as fp:
                toml.dump(sts, fp)

            cmd = f"python -m octopus {config_path} V --seed {s} --debug"
            if self.override:
                cmd += " --override"

            tmpdir = f"/tmp/{uuid.uuid1()}"
            slurm_cmd = None
            if self.slurm:
                lines = [
                    "#!/bin/sh",
                    f"#SBATCH --job-name=O.V",
                    f"#SBATCH --output={log_path}",
                    f"#SBATCH --error={log_path}",
                    f"export TMPDIR={tmpdir}",
                    "export DNNV_OPTIONAL_SIMPLIFIERS=ReluifyMaxPool",
                    f"mkdir {tmpdir}",
                    "cat /proc/sys/kernel/hostname",
                    "source .env.d/openenv.sh",
                    cmd,
                    f"rm -rf {tmpdir}",
                ]
                lines = [x + "\n" for x in lines]

                open(slurm_script_path, "w").writelines(lines)
                param_node_nb = (
                    f"-c {self.nb_verify_nodes}"
                    if "nb_verify_nodes" in self.__dict__
                    else ""
                )

                param_node = (
                    f" -p nolim -w {self.veri_nodes[nb_todo%len(self.veri_nodes)]}"
                    if self.veri_nodes
                    else ""
                )
                slurm_cmd = f"sbatch {param_node_nb} {param_node} {slurm_script_path}"

            self._exec(cmd, slurm_cmd, self.sleep_verify)

        self.logger.info(
            f"Tasks: done: {nb_done}, todo: {nb_todo}, total: {len(self.problems_V)}."
        )

    # Analyze logs
    def analyze(self):
        self.logger.info("Analyzing ...")
        # self._train_progress_plot()
        # df_cache_path = os.path.join(self.result_dir, "result.feather")
        if os.path.exists(self.cached_res_path) and not self.override:
            self.logger.info("Using cached results.")
            df = pd.read_feather(self.cached_res_path)
        else:
            self.logger.info("Parsing log files ...")
            df = self._parse_logs()
            if self.go:
                df.to_feather(self.cached_res_path)
                self.logger.info("Result cached.")

        # self.logger.debug(f"Data frame: \n{df}")

        if self.go:
            self._analyze_time(df)
            self._analyze_training(df)
            self._analyze_verification(df)
            #######################
            #######################
            #######################
            #######################
            #######################
            #######################
            #######################
            #######################
            #######################
            #######################
            #######################
            #######################

    def _parse_logs(self):
        df = pd.DataFrame({x: [] for x in self.labels})
        self.logger.info("Failed tasks:")
        self.logger.info("--------------------")
        for i, (a, n, h, s, e, p, v) in enumerate(self.problems_V):
            sts, _, _, train_log_path, name = self._get_problem_paths(
                "T", a=a, n=n, h=h, s=s, e=e
            )
            self.logger.debug(f"Problem configs: {sts}")
            self.logger.debug(f"Name: {name}")
            self.logger.debug(f"Train log path: {train_log_path}")
            with open(train_log_path, "r") as fp:
                lines = fp.readlines()

            # TODO:
            # temp solution for new ReLU calculations
            # remove this for new log files

            if a == "CIFAR10" and n == "OVAL21_o":
                total_nb_relus = 3172
            elif a == "CIFAR10" and n == "OVAL21_w":
                total_nb_relus = 6244
            elif a == "CIFAR10" and n == "OVAL21_d":
                total_nb_relus = 6756
            elif a == "CIFAR10" and n in [
                "LeNet_o",
                "LeNet_w",
                "LeNet_oe",
                "LeNet_we",
            ]:
                assert False
            elif a == "MNIST" and n == "Net256x2":
                total_nb_relus = 512
            else:
                nb_relu_line = [x for x in lines if "# ReLUs" in x]
                # print(nb_relu_line)
                assert len(nb_relu_line) == 1
                total_nb_relus = int(nb_relu_line[0].split()[-1])

            test_lines = [x for x in lines if "[Test] epoch:" in x]
            relu_lines = [x for x in lines if "[Test] Stable ReLUs:" in x]

            # select the best accuracy instead of last epoch's accuracy
            target_model = sts["verify"]["target_model"]
            target_epoch = Problem.Utility.get_target_epoch(
                target_model, train_log_path
            )
            # print(test_lines, len(test_lines), train_log_path)
            test_accuracy = [
                float(x.strip().split()[-1][:-1]) / 100 for x in test_lines
            ][target_epoch - 1]

            # at least 1 stability estimator is enabled during training
            if "Disabled" not in relu_lines[0]:
                stable_ReLUs = {}
                for l in relu_lines:
                    tokens = l[l.index("[Test] Stable ReLUs:") + 20 :].split()
                    assert len(tokens) % 2 == 0
                    for i in range(int(len(tokens) / 2)):
                        se = tokens[i * 2][:-1]
                        acc = float(tokens[i * 2 + 1][:-1]) / 100
                        if se not in stable_ReLUs:
                            stable_ReLUs[se] = [acc]
                        else:
                            stable_ReLUs[se] += [acc]

                stable_estimators = sts["train"]["stable_estimators"]
                relu_accuracy = [
                    stable_ReLUs[x][target_epoch - 1] for x in stable_estimators
                ]

            # No stability estimator is enabled during training
            else:
                relu_accuracy = None
            # if h == "Baseline":
            #    stable_estimators = sts["train"]["stable_estimators"]
            #    relu_accuracy = [
            #        stable_ReLUs[x][target_epoch - 1] for x in stable_estimators
            #    ]

            # else:
            #    assert (
            #        len(list(self.stabilizers[h].values())[0]["stable_estimators"]) == 1
            #    )
            #    stable_estimators = list(
            #        list(self.stabilizers[h].values())[0]["stable_estimators"]
            #    )[0]
            #    relu_accuracy = [stable_ReLUs[stable_estimators][target_epoch - 1]]

            # this is a hack to work around broken training logs
            # info_lines = [x for x in lines if "(INFO)" in x]
            # start = datetime.strptime(info_lines[0][16:38], "%m/%d/%Y %H:%M:%S %p")
            # end = datetime.strptime(info_lines[-1][16:38], "%m/%d/%Y %H:%M:%S %p")
            # training_time = (end - start).seconds
            train_time_line = [x for x in lines if "Spent" in x]
            assert len(train_time_line) == 1
            training_time = float(train_time_line[0].split()[1])

            # TODO: retract to this after fixing pruning
            # assert 'Spent' in lines[-1], train_log_path
            # training_time = float(lines[-1].strip().split()[-2])
            _, _, _, veri_log_path, _ = self._get_problem_paths(
                "V", a=a, n=n, h=h, s=s, e=e, p=p, v=v
            )

            vp = VerificationProblem(self.logger, veri_log_path, _, v)

            self.logger.debug(f"Veri log path: {veri_log_path}")
            if os.path.exists(veri_log_path):
                answer, verification_time = vp.analyze()
            else:
                self.logger.warn(f"Missing veri log: {veri_log_path} ... ")
                self.logger.warn(f"of training model: {train_log_path} ... ")
                answer = None
                verification_time = None

            if answer is None or verification_time is None:
                print("rm", veri_log_path)  # , verification_time, answer)

            # compute unstable ReLUs for DNNVWB
            # supported verifiers:
            # DNNVWB:neurify
            """
            if os.path.exists(veri_log_path) and v in ["DNNVWB:neurify"]:
                lines = open(veri_log_path, "r").readlines()
                wb_lines = [x for x in lines if "WB" in x]
                unstable_relu_lines = [x for x in wb_lines if "unstable" in x]

                unstable_relus = []
                for x in unstable_relu_lines:
                    unstable_relus += [int(x.strip().split()[-3])]
                # print(unstable_relus, veri_log_path)
                avg_unstable_relus = np.mean(np.array(unstable_relus))

                relu_accuracy_veri = 1 - avg_unstable_relus / total_nb_relus

            else:
                relu_accuracy_veri = 0
            """
            # relu_accuracy_veri = 0

            if self.go:
                df.loc[len(df.index)] = [
                    a,
                    n,
                    h,
                    s,
                    p,
                    e,
                    v,
                    test_accuracy,
                    self.code_veri_answer[answer],
                    verification_time,
                    training_time,
                ]
        self.logger.info("--------------------")
        return df

    def _analyze_time(self, df):
        verification_times = df["veri time"].values
        dft = df[df["property"] == self.props[0]]
        dft = dft[dft["verifier"] == self.verifiers[0]]
        dft = dft[dft["epsilon"] == self.epsilons[0]]
        training_times = dft["training time"]
        self.logger.info(f"Total training time: {np.sum(training_times)/3600} hours.")
        self.logger.info(
            f"Total verification time: {np.sum(verification_times)/3600} hours."
        )

    def _analyze_training(self, df):
        self.logger.info("Analyzing training ...")

        # train progress plotter
        # self._train_progress_plot()
        # train test accuracy plot
        self._train_boxplot(df)
        # train catplot
        # self._train_catplot(df, "test accuracy")

    def _train_progress_plot(self):
        self.logger.info("Generating training progress plot ...")
        for i, (a, n, h, s, e) in enumerate(tqdm(self.problems_T)):
            (
                sts,
                _,
                _,
                log_path,
                name,
            ) = self._get_problem_paths("T", a=a, n=n, h=h, s=s, e=e)

            with open(log_path, "r") as fp:
                lines = fp.readlines()
            assert any([x for x in lines if "Mission Complete" in x])
            lines_train = [x for x in lines if "[Train]" in x]
            lines_test = [x for x in lines if "[Test]" in x]

            train_stable_ReLUs = {}
            for l in [x for x in lines_train if "Stable ReLUs" in x]:
                tokens = l[l.index("[Train] Stable ReLUs:") + 21 :].split()
                assert len(tokens) % 2 == 0
                for i in range(int(len(tokens) / 2)):
                    se = tokens[i * 2][:-1]
                    acc = float(tokens[i * 2 + 1][:-1]) / 100
                    if se not in train_stable_ReLUs:
                        train_stable_ReLUs[se] = [0, acc]
                    else:
                        train_stable_ReLUs[se] += [acc]

            test_accuracy = [
                float(x.strip().split()[-1][:-1]) / 100
                for x in lines_test
                if "epoch" in x
            ]

            train_loss = [
                float(x.strip().split()[-1]) for x in lines_train if "epoch" in x
            ]

            nb_steps = len(train_loss)
            nb_epochs = sts["train"]["epochs"]

            # train stable ReLUs
            Y1 = train_stable_ReLUs
            X1 = range(0, len(list(Y1.values())[0]))

            # bias shaping points
            X2 = []
            Y2 = []

            # test accuracy
            X3 = np.linspace(
                nb_steps / nb_epochs,
                nb_steps,
                nb_epochs,
            )
            Y3 = test_accuracy

            # train loss
            X4 = range(len(train_loss))
            Y4 = train_loss

            # max_safe_relu = sum([self.model.activation[layer].view(
            #    self.model.activation[layer].size(0), -1).shape[-1] for layer in self.model.activation])

            p_plot = ProgressPlot()
            # max_safe_relu))
            p_plot.draw_train(
                X1,
                Y1,
                X2,
                Y2,
                (0, nb_steps),
                (0, 1),
            )

            p_plot.draw_accuracy(X3, Y3, X4, Y4, (0, 1))
            p_plot.draw_grid(
                x_stride=nb_steps / 5,
                y_stride=0.2,
            )

            fig_dir = os.path.join(self.result_dir, "train_figures")
            Path(fig_dir).mkdir(parents=True, exist_ok=True)
            path = os.path.join(fig_dir, name + ".pdf")
            p_plot.save(name, path)
            p_plot.clear()

    def _train_boxplot(self, df):
        self.logger.info("Plotting training ...")
        # plot accuracy/stable relu among network/stabilizer pairs.
        x_labels = []
        c_test_acc = []
        c_relu_acc = []
        # c_stable_relu = []
        for a in self.artifacts:
            for n in self.networks:
                for h in self.stabilizers.keys():
                    # for e in self.epsilons:
                    dft = df
                    dft = dft[dft["artifact"] == a]
                    dft = dft[dft["network"] == n]
                    dft = dft[dft["stabilizer"] == h]
                    dft = dft[dft["property"] == self.props[0]]
                    dft = dft[dft["verifier"] == self.verifiers[0]]

                    x_labels += [f"{a[0]}:{n[-1]}:{h}"]

                    test_acc = dft["test accuracy"].values
                    relu_acc = 0
                    # relu_acc = dft["relu accuracy"].values

                    # stable_relu = dft['stable relu'].values
                    c_test_acc += [test_acc]
                    c_relu_acc += [relu_acc]
                    # c_stable_relu += [stable_relu]
        # print(c_test_acc)
        c_test_acc = np.array(c_test_acc, dtype=object) * 100
        c_relu_acc = np.array(c_relu_acc, dtype=object) * 100
        # c_ = np.array(c_stable_relu, dtype=object)

        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        # ax2 = ax1.twinx()
        bp1 = colored_box_plot(ax1, c_test_acc.T, "red", "tan")
        assert len(set(c_test_acc[0])) == 1
        ax1.axhline(y=c_test_acc[0][0], color="pink")
        ax1.axhline(y=c_test_acc[0][0] - 2, color="pink")
        bp2 = colored_box_plot(ax1, c_relu_acc.T, "blue", "cyan")
        # bp2 = colored_box_plot(ax2, collection_stable_relu.T, 'blue', 'cyan')

        ax1.legend(
            [bp1["boxes"][0], bp2["boxes"][0]],
            ["Test Accuracy", "# Stable ReLUs"],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.34),
            fancybox=True,
            shadow=True,
            ncol=2,
        )

        # make grid & set correct xlabels
        ax1.xaxis.set_major_locator(MultipleLocator(len(self.stabilizers)))
        ax1.xaxis.set_minor_locator(MultipleLocator(1))
        ax1.xaxis.set_major_formatter(
            lambda x, pos: x_labels[int(x - 1) % len(x_labels)]
        )
        ax1.xaxis.set_minor_formatter(
            lambda x, pos: x_labels[int(x - 1) % len(x_labels)]
        )
        ax1.grid(which="major", axis="x", color="grey", linestyle=":", linewidth=0.25)
        ax1.grid(which="minor", axis="x", color="grey", linestyle=":", linewidth=0.25)

        vt_lines = list(np.arange(0, len(x_labels), len(self.stabilizers)) + 0.5)[1:]
        for x in vt_lines:
            if int(x - 0.5) % (len(self.artifacts) * len(self.stabilizers)) == 0:
                ax1.axvline(x=x, color="grey", linestyle="--", linewidth=1)
            else:
                ax1.axvline(x=x, color="grey", linestyle="-.", linewidth=0.5)

        rotation = 90
        for tick in ax1.get_xmajorticklabels():
            tick.set_rotation(rotation)
        for x in ax1.get_xminorticklabels():
            x.set_rotation(rotation)

        ax1.plot()

        # ax1.set_ylim(0, 100)
        ax1.set_ylim(90, 100)
        ax1.grid()
        ax1.set_ylabel("Test Accuracy(%,Red)/Stable ReLUs(%,Blue)")
        # ax2.set_ylabel("# Stable ReLUs(%)")
        ax1.set_xlabel("Artifact:Network:stabilizers")
        plt.title(
            "Test Accuracy and # Stable ReLUs vs. Artifact, Network, and stabilizers"
        )
        plt.savefig(
            os.path.join(self.result_dir, "T.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
        plt.xlabel(x_labels)
        fig.clear()
        plt.close(fig)

    def _train_catplot(self, df, kind):
        dft = df
        dft = dft[dft["artifact"] == "MNIST"]
        dft = dft[dft["property"] == self.props[0]]
        dft = dft[dft["epsilon"] == self.epsilons[0]]
        dft = dft[dft["verifier"] == self.verifiers[0]]

        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        # bp1 = colored_box_plot(ax1, collection_accuracy.T, 'red', 'tan')
        # bp2 = colored_box_plot(ax2, collection_stable_relu.T, 'blue', 'cyan')
        # bp2 = colored_box_plot(ax2, collection_accuracy_relu.T, 'blue', 'cyan')

        # plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Test Acc.', 'ReLU Acc.'], loc='center left')

        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        sns.catplot(
            x="network",
            y=kind,
            hue="stabilizer",
            col="artifact",
            kind="box",
            data=df,
            palette="Set3",
        )
        plt.savefig(
            os.path.join(self.result_dir, f"Tc_{kind}.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
        fig.clear()
        plt.close(fig)

    def _analyze_verification(self, df):
        stabilizers = list(self.stabilizers.keys())

        stabilizers_ba = ["Baseline"]
        self._verification_plot_eps(df, stabilizers_ba)
        stabilizers_ba = ["Baseline"]
        self._verification_plot_eps(df, stabilizers_ba, distinguish_sat_vs_unsat=True)

        # stabilizers_bs = ["Baseline"] + [x for x in stabilizers if "BS" in x]
        # self._verification_plot_eps(df, stabilizers_bs)
        # stabilizers_rs = ["Baseline"] + [x for x in stabilizers if "RS" in x]
        # self._verification_plot_eps(df, stabilizers_rs)
        # stabilizers_sp = ["Baseline"] + [x for x in stabilizers if "SP" in x]
        # self._verification_plot_eps(df, stabilizers_sp)

        self._analyze_table(df)

    def _verification_plot_eps(self, df, stabilizers, distinguish_sat_vs_unsat=False):
        self.logger.info("Plotting verification ...")

        colors = [(0, 0, 0)] + sns.color_palette("hls", len(self.stabilizers) - 1)
        for a in self.artifacts:
            for n in self.networks:
                for v in self.verifiers:
                    # plot verification time(Y) vs. Epsilon Values(X) for each stabilizer
                    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
                    fig.autofmt_xdate(rotation=90)
                    ax2 = ax1.twinx()
                    title_prefix = f"[{a}:{n}:{v}]"
                    collection_verification_time = {}
                    collection_problem_solved = {}
                    # for i, h in enumerate(self.stabilizers.keys()):
                    for i, h in enumerate(stabilizers):
                        avg_v_time = []
                        nb_solved = []
                        nb_unsat = []
                        nb_sat = []
                        for e in self.epsilons:
                            dft = df
                            dft = dft[dft["artifact"] == a]
                            dft = dft[dft["network"] == n]
                            dft = dft[dft["verifier"] == v]
                            dft = dft[dft["stabilizer"] == h]
                            dft = dft[dft["epsilon"] == e]
                            nb_problems = len(dft["veri ans"])
                            avg_v_time += [np.mean(dft["veri time"].to_numpy())]
                            nb_solved += [
                                np.where(dft["veri ans"].to_numpy() < 3)[0].shape[0]
                            ]
                            nb_unsat += [
                                np.where(dft["veri ans"].to_numpy() == 1)[0].shape[0]
                            ]
                            nb_sat += [
                                np.where(dft["veri ans"].to_numpy() == 2)[0].shape[0]
                            ]

                        collection_verification_time[h] = avg_v_time
                        collection_problem_solved[h] = nb_solved

                        plot1 = ax1.plot(
                            avg_v_time,
                            linestyle="-",
                            label=h,
                            color=(*colors[i], 2 / 3),
                        )
                        if not distinguish_sat_vs_unsat:
                            plot2 = ax2.plot(
                                nb_solved,
                                linestyle=":",
                                label=h,
                                color=(*colors[i], 2 / 3),
                            )
                        else:
                            plot2 = ax2.plot(
                                nb_unsat,
                                linestyle="--",
                                label=h,
                                color=(*colors[i], 2 / 3),
                            )
                            plot3 = ax2.plot(
                                nb_sat,
                                linestyle=":",
                                label=h,
                                color=(*colors[i], 2 / 3),
                            )

                    ax1.legend(
                        loc="upper left",
                        bbox_to_anchor=(-0.22, 0.8),
                        fancybox=True,
                    )
                    ax2.legend(
                        loc="upper right",
                        bbox_to_anchor=(1.22, 0.8),
                        fancybox=True,
                    )

                    ax1.set_xlabel("Epsilon")
                    ax1.set_ylabel("Verification Time(s)")
                    ax2.set_ylabel("Solved Problems")
                    ax1.set_ylim(
                        -self.base_settings["verify"]["time"] * 0.05,
                        self.base_settings["verify"]["time"] * 1.05,
                    )
                    ax2.set_ylim(-1, nb_problems + 1)
                    ax1.set_xticks(range(len(self.epsilons)))
                    ax1.set_xticklabels(self.epsilons)

                    # ax1.xticks(rotation=90)
                    plt.title(
                        title_prefix
                        + " Avg. Verification Time/Problems Solved vs. Epsilons"
                    )
                    postfix = stabilizers[-1][:2]
                    if distinguish_sat_vs_unsat:
                        postfix += "_sus"
                    plt.savefig(
                        os.path.join(
                            self.result_dir, f"V_{title_prefix}_{postfix}.pdf"
                        ),
                        format="pdf",
                        bbox_inches="tight",
                    )
                    fig.clear()
                    plt.close(fig)

        self.logger.info("Plotting verification sat/unsat...")
        colors = [(0, 0, 0)] + sns.color_palette("hls", len(self.stabilizers) - 1)
        for a in self.artifacts:
            for n in self.networks:
                for v in self.verifiers:
                    # plot verification time(Y) vs. Epsilon Values(X) for each stabilizer
                    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
                    ax2 = ax1.twinx()
                    title_prefix = f"[{a}:{n}:{v}]"
                    for i, h in enumerate(self.stabilizers.keys()):
                        nb_unsat = []
                        nb_sat = []
                        for e in self.epsilons:
                            dft = df
                            dft = dft[dft["artifact"] == a]
                            dft = dft[dft["network"] == n]
                            dft = dft[dft["verifier"] == v]
                            dft = dft[dft["stabilizer"] == h]
                            dft = dft[dft["epsilon"] == e]
                            nb_problems = len(dft["veri ans"])
                            nb_unsat += [
                                np.where(dft["veri ans"].to_numpy() == 1)[0].shape[0]
                            ]
                            nb_sat += [
                                np.where(dft["veri ans"].to_numpy() == 2)[0].shape[0]
                            ]

                        plot1 = ax1.plot(
                            nb_unsat,
                            linestyle="-",
                            label=h,
                            color=(*colors[i], 2 / 3),
                        )
                        plot2 = ax2.plot(
                            nb_sat,
                            linestyle=":",
                            label=h,
                            color=(*colors[i], 2 / 3),
                        )

                    ax1.legend(
                        loc="upper left",
                        bbox_to_anchor=(-0.22, 0.8),
                        fancybox=True,
                    )
                    ax2.legend(
                        loc="upper right",
                        bbox_to_anchor=(1.22, 0.8),
                        fancybox=True,
                    )

                    ax1.set_xlabel("Epsilon")
                    ax1.set_ylabel("Solved Problems(Unsat)")
                    ax2.set_ylabel("Solved Problems(Sat)")
                    ax1.set_ylim(-1, nb_problems + 1)
                    ax2.set_ylim(-1, nb_problems + 1)
                    ax1.set_xticks(range(len(self.epsilons)))
                    ax1.set_xticklabels(self.epsilons)

                    plt.title(
                        title_prefix + " Avg. Verification Problems Solved vs. Epsilons"
                    )
                    plt.savefig(
                        os.path.join(self.result_dir, f"Vsus_{title_prefix}.pdf"),
                        format="pdf",
                        bbox_inches="tight",
                    )
                    fig.clear()
                    plt.close(fig)

    def _analyze_stable_ReLUs(self, df):
        self.logger.info("Plotting stable ReLUs ...")
        colors = [(0, 0, 0)] + sns.color_palette("hls", len(self.stabilizers) - 1)
        for a in self.artifacts:
            for n in self.networks:
                for v in self.verifiers:
                    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
                    ax2 = ax1.twinx()
                    title_prefix = f"[{a}:{n}:{v}]"
                    sr_train = {}
                    sr_veri = {}
                    for i, h in enumerate(self.stabilizers.keys()):
                        sr_train_ = []
                        sr_veri_ = []
                        for e in self.epsilons:
                            dft = df
                            dft = dft[dft["artifact"] == a]
                            dft = dft[dft["network"] == n]
                            dft = dft[dft["verifier"] == v]
                            dft = dft[dft["stabilizer"] == h]
                            dft = dft[dft["epsilon"] == e]

                            sr_train_ += [np.mean(dft["relu accuracy"].to_numpy())]
                            sr_veri_ += [np.mean(dft["relu accuracy veri"].to_numpy())]

                        sr_train[h] = sr_train_
                        sr_veri[h] = sr_veri_

                        plot1 = ax1.plot(
                            sr_train_,
                            linewidth=1,
                            linestyle="-",
                            label=h,
                            color=(*colors[i], 2 / 3),
                        )
                        plot2 = ax2.plot(
                            sr_veri_,
                            linewidth=1,
                            linestyle=":",
                            label=h,
                            color=(*colors[i], 2 / 3),
                        )

                    # ax1.axvline(x=4, color="grey", linestyle="..", linewidth=1)
                    ax1.legend(
                        loc="upper left",
                        bbox_to_anchor=(-0.22, 0.8),
                        fancybox=True,
                    )
                    ax2.legend(
                        loc="upper right",
                        bbox_to_anchor=(1.22, 0.8),
                        fancybox=True,
                    )

                    ax1.set_xlabel("Epsilon")
                    ax1.set_ylabel("Stable ReLUs in Training")
                    ax2.set_ylabel("Stable ReLUs in Verification")
                    ax1.set_ylim(-0.05, 1.05)
                    ax2.set_ylim(-0.05, 1.05)
                    ax1.set_xticks(range(len(self.epsilons)))
                    ax1.set_xticklabels(self.epsilons)

                    plt.title(title_prefix + " Stable ReLUs in Training/Verification")
                    plt.savefig(
                        os.path.join(self.result_dir, f"SR_{title_prefix}.pdf"),
                        format="pdf",
                        bbox_inches="tight",
                    )
                    fig.clear()
                    plt.close(fig)

    def _analyze_stable_ReLUs_two(self, df):
        colors = [(0, 0, 0)] + sns.color_palette("hls", len(self.stabilizers) - 1)
        for a in self.artifacts:
            for n in self.networks:
                for v in self.verifiers:
                    for s in self.seeds:
                        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
                        ax2 = ax1.twinx()
                        title_prefix = f"[{a}:{n}:{v}:{s}]"
                        sr_train = {}
                        sr_veri = {}
                        for i, h in enumerate(self.stabilizers.keys()):
                            sr_train_ = []
                            sr_veri_ = []
                            x1_ = []
                            x2_ = []
                            for e in self.epsilons:
                                dft = df
                                dft = dft[dft["seed"] == s]
                                dft = dft[dft["artifact"] == a]
                                dft = dft[dft["network"] == n]
                                dft = dft[dft["verifier"] == v]
                                dft = dft[dft["stabilizer"] == h]
                                dft = dft[dft["epsilon"] == e]

                                tmp = dft["relu accuracy"].to_numpy()
                                sr_train_ += [tmp]
                                x1_ += [e * 100 - 1] * len(tmp)

                                tmp = dft["relu accuracy veri"].to_numpy()
                                sr_veri_ += [tmp]
                                x2_ += [e * 100 - 1] * len(tmp)

                            sr_train[h] = sr_train_
                            sr_veri[h] = sr_veri_

                            plot1 = ax1.scatter(
                                x=x1_,
                                y=sr_train_,
                                label=h,
                                marker="x",
                                color=(*colors[i], 2 / 3),
                            )
                            plot2 = ax2.scatter(
                                x=x2_,
                                y=sr_veri_,
                                label=h,
                                facecolors="none",
                                edgecolors=(*colors[i], 2 / 3),
                            )

                        # ax1.axvline(x=4, color="grey", linestyle="--", linewidth=1)

                        ax1.legend(
                            loc="upper left",
                            bbox_to_anchor=(-0.22, 0.8),
                            fancybox=True,
                        )
                        ax2.legend(
                            loc="upper right",
                            bbox_to_anchor=(1.22, 0.8),
                            fancybox=True,
                        )

                        ax1.set_xlabel("Epsilon")
                        ax1.set_ylabel("Stable ReLUs in Training")
                        ax2.set_ylabel("Stable ReLUs in Verification")
                        ax1.set_ylim(-0.05, 1.05)
                        ax2.set_ylim(-0.05, 1.05)
                        ax1.set_xticks(range(len(self.epsilons)))
                        ax1.set_xticklabels(self.epsilons)

                        plt.title(
                            title_prefix + " Stable ReLUs in Training/Verification"
                        )
                        plt.savefig(
                            os.path.join(self.result_dir, f"SRe_{title_prefix}.pdf"),
                            format="pdf",
                            bbox_inches="tight",
                        )
                        fig.clear()
                        plt.close(fig)

    def _analyze_table(self, df):
        solved_ = {}
        time_ = {}
        par2_ = {}
        for a in self.artifacts:
            for v in self.verifiers:
                for n in self.networks:
                    for h in self.stabilizers.keys():
                        name = f"{a}:{v}:{n}:{h}"
                        solved = 0
                        time = 0
                        par2 = 0
                        for s in self.seeds:
                            for e in self.epsilons:
                                for p in self.props:
                                    dft = df
                                    dft = dft[dft["seed"] == s]
                                    dft = dft[dft["artifact"] == a]
                                    dft = dft[dft["network"] == n]
                                    dft = dft[dft["verifier"] == v]
                                    dft = dft[dft["stabilizer"] == h]
                                    dft = dft[dft["epsilon"] == e]
                                    dft = dft[dft["property"] == p]

                                    veri_ans = dft["veri ans"].to_numpy()
                                    veri_time = dft["veri time"].to_numpy()

                                    assert len(veri_ans) == 1 and len(veri_time) == 1
                                    if veri_ans[0] in [1, 2]:
                                        par2 += veri_time[0]
                                        solved += 1
                                    else:
                                        par2 += 600 * 2
                                    time += veri_time[0]
                        solved_[name] = solved
                        time_[name] = time
                        par2_[name] = par2

        print(self.verifiers)
        for a in self.artifacts:
            for i, x in enumerate([solved_, time_, par2_]):
                print("----------")
                for h in self.stabilizers:
                    print(h, end=",")
                    for v in self.verifiers:
                        for n in self.networks:
                            base = x[f"{a}:{v}:{n}:{'Baseline'}"]
                            if base == 0:
                                print(
                                    f'{x[f"{a}:{v}:{n}:{h}"]:.2f}, +{x[f"{a}:{v}:{n}:{h}"]:.2f}',
                                    end=",",
                                )
                            else:
                                print(
                                    f'{x[f"{a}:{v}:{n}:{h}"]:.2f}, {x[f"{a}:{v}:{n}:{h}"]/base:.2f}',
                                    end=",",
                                )
                    print()
