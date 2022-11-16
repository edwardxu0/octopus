import torch
import os
import numpy as np
import subprocess
import sys
import logging
import gc
import re


import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR

from ..artifact.artifacts import *

from ..stability_estimator import get_stability_estimators

from ..plot.train_progress import ProgressPlot
from ..architecture.ReLUNet import ReLUNet
from ..architecture.LeNet import LeNet, LeNet2


RES_MONITOR_PRETIME = 200


class Problem:
    def __init__(self, settings):
        self.override = settings.override
        self.sub_dirs = settings.sub_dirs
        self.cfg_train = settings.cfg_train
        self.cfg_heuristic = settings.cfg_heuristic
        self.cfg_verify = settings.cfg_verify
        self.logger = settings.logger
        self._setup_(settings)

    def _setup_(self, settings):
        self.seed = settings.seed
        torch.manual_seed(settings.seed)
        np.random.seed(settings.seed)
        torch.set_printoptions(threshold=100000)

        self.__name__ = self.Utility.get_hashed_name(
            [self.cfg_train, self.cfg_heuristic, self.seed]
        )
        self.model_path = os.path.join(
            self.sub_dirs["model_dir"], f"{self.__name__}.onnx"
        )
        self.train_log_path = os.path.join(
            self.sub_dirs["train_log_dir"], f"{self.__name__}.txt"
        )

        cfg = self.cfg_train
        use_gpu = False if "gpu" not in cfg else cfg["gpu"]
        if use_gpu and settings.task == "T":
            assert torch.cuda.is_available()
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        use_cuda = use_gpu and torch.cuda.is_available()
        amp = False if "amp" not in cfg else cfg["amp"]
        self.amp = use_cuda and amp
        if use_cuda:
            self.logger.info("CUDA enabled.")
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.artifact = globals()[cfg["artifact"]](
            data_mode=cfg["adv_train"],
            batch_size=cfg["batch_size"],
            test_batch_size=cfg["test_batch_size"],
            use_cuda=self.device,
        )
        self.train_loader, self.test_loader = self.artifact.get_data_loader()

        if self.cfg_train["net_name"] in ["NetS", "NetM", "NetL"]:
            self.model = ReLUNet(
                self.artifact,
                self.cfg_train["net_layers"],
                self.logger,
                self.device,
                self.amp,
            ).to(self.device)
        elif self.cfg_train["net_name"] == "LeNet":
            self.model = LeNet(
                self.artifact,
                self.logger,
                self.device,
                self.amp,
            ).to(self.device)
        elif self.cfg_train["net_name"] == "LeNet2":
            self.model = LeNet2(
                self.artifact,
                self.logger,
                self.device,
                self.amp,
            ).to(self.device)
        else:
            raise NotImplementedError(
                f"Unsupported architecture: {self.cfg_train['net_name']}"
            )

        self.logger.info(f"Network:\n{self.model}")
        self.logger.info(f"# ReLUs: {self.model.nb_ReLUs}")

    # Training ...
    def _setup_train(self):
        if "save_log" in self.cfg_train and self.cfg_train["save_log"]:
            self.logger.debug(f"Saving training log to: {self.train_log_path}")

            file_handler = logging.FileHandler(self.train_log_path, "w")
            self.logger.addHandler(file_handler)

        # setup train data collectors
        self.train_BS_points = []
        self.train_loss = []
        self.test_accuracy = []

        # configure heuristics
        if "stable_estimator" in self.cfg_train:
            self.stable_estimators = get_stability_estimators(
                self.cfg_train["stable_estimator"], self.model
            )
            self.train_stable_ReLUs = {x: [] for x in self.stable_estimators}
            self.logger.info(
                f"Train stability estimators: {list(self.stable_estimators.keys())}"
            )
        else:
            self.stable_estimators = None
            self.train_stable_ReLUs = None
            self.logger.info("No activated train stability estimators.")

        self.model._setup_heuristics(self.cfg_heuristic)

    def _trained(self):
        # TODO: account for log file and # epochs
        # print(self.train_log_path)
        if not os.path.exists(self.train_log_path):
            trained = False  # os.path.exists(self.model_path)
        else:
            trained = (
                len(
                    [
                        x
                        for x in open(self.train_log_path, "r").readlines()
                        if "[Test] epoch: " in x
                    ]
                )
                == self.cfg_train["epochs"]
            )
        return trained

    def train(self):
        if self._trained() and not self.override:
            self.logger.info(f"Skipping trained network. {self.__name__}")
        else:
            self._setup_train()

            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.cfg_train["lr"]
            )
            self.LR_decay_scheduler = StepLR(
                self.optimizer, step_size=1, gamma=self.cfg_train["gamma"]
            )

            if self.amp:
                self.logger.info("CUDA AMP enabled.")
                self.amp_scaler = torch.cuda.amp.GradScaler()

            for epoch in range(1, self.cfg_train["epochs"] + 1):
                self._train_epoch(epoch)
                self._test_epoch(epoch)

                # save model
                if self.cfg_train["save_model"]:
                    self.logger.debug(f"Saving model: {self.model_path}")
                    dummy_input = torch.randn(
                        [1] + self.artifact.input_shape, device=self.device
                    )
                    torch.onnx.export(
                        self.model, dummy_input, self.model_path, verbose=False
                    )
                    torch.save(self.model.state_dict(), f"{self.model_path[:-5]}.pt")

                    if self.cfg_train["save_intermediate"]:
                        torch.onnx.export(
                            self.model,
                            dummy_input,
                            f"{self.model_path[:-5]}.{epoch}.onnx",
                            verbose=False,
                        )
                        torch.save(
                            self.model.state_dict(),
                            f"{self.model_path[:-5]}.{epoch}.pt",
                        )

                self.LR_decay_scheduler.step()

                self._plot_train()

    def _train_epoch(self, epoch):
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # Use AMP
            # equivalent to:
            # with torch.cuda.amp.autocast(enabled=self.amp):
            # to disable warning when cuda==false and amp==true.
            with torch.autocast(self.device.type, enabled=self.amp):

                output_pre_softmax = self.model(data)
                output = F.log_softmax(output_pre_softmax, dim=1)

                loss = F.nll_loss(output, target)

                # [H] RS loss
                if (
                    self.cfg_heuristic
                    and "rs_loss" in self.cfg_heuristic
                    and self.Utility.heuristic_enabled_epochwise(
                        epoch,
                        self.cfg_heuristic["rs_loss"]["start"],
                        self.cfg_heuristic["rs_loss"]["end"],
                    )
                ):

                    rs_loss = self.model.run_heuristics(
                        "rs_loss", data=data, test_loader=self.test_loader
                    )
                    loss += rs_loss * self.cfg_heuristic["rs_loss"]["weight"]

                if loss.isnan():
                    # raise ValueError("Loss is NaN.")
                    self.logger.warn("LOSS is Nan.")

                if self.amp:
                    self.amp_scaler.scale(loss).backward()
                    self.amp_scaler.step(self.optimizer)
                    self.amp_scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

            # [H] bias shaping
            if (
                self.cfg_heuristic
                and "bias_shaping" in self.cfg_heuristic
                and batch_idx != 0
                and self.Utility.heuristic_enabled_epochwise(
                    epoch,
                    self.cfg_heuristic["bias_shaping"]["start"],
                    self.cfg_heuristic["bias_shaping"]["end"],
                )
            ):
                # print('before', self.model.estimate_stable_ReLU(self.cfg_train['ReLU_estimation']), self.test_loader)
                if self.model.run_heuristics(
                    "bias_shaping",
                    data=data,
                    epoch=epoch,
                    batch=batch_idx,
                    test_loader=self.test_loader,
                ):
                    BS_point = len(self.train_loader) * (epoch - 1) + batch_idx
                    self.train_BS_points += [BS_point]
                # print('after', self.model.estimate_stable_ReLU(self.cfg_train['ReLU_estimation']), self.test_loader)

            # [H] pruning
            # prune after entire epoch trained
            # using pre-activation values of last mini-batch
            if (
                self.cfg_heuristic
                and "prune" in self.cfg_heuristic
                and self.Utility.heuristic_enabled_epochwise(
                    epoch,
                    self.cfg_heuristic["prune"]["start"],
                    self.cfg_heuristic["prune"]["end"],
                )
            ):
                re_arched = self.model.run_heuristics(
                    "prune",
                    data=data,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    total_batches=len(self.train_loader),
                    test_loader=self.test_loader,
                    total_epoch=self.cfg_train["epochs"],
                )
                if re_arched and "SIP" in self.stable_estimators:
                    self.stable_estimators["SIP"].init_inet()

            if batch_idx % self.cfg_train["log_interval"] == 0:

                # log test accuracy
                self.logger.info(
                    f"[Train] epoch: {epoch:3} batch: {batch_idx:5} {100.*batch_idx/len(self.train_loader):5.2f}% Loss: {loss.item():10.6f}"
                )
                # log stable ReLUs
                if self.stable_estimators is not None:
                    se_str = ""
                    for se in self.stable_estimators:
                        self.stable_estimators[se].propagate(
                            data=data, test_loader=self.test_loader
                        )
                        stable_le_0, stable_ge_0 = self.stable_estimators[
                            se
                        ].get_stable_ReLUs()

                        stable_le_0 = sum(
                            [
                                torch.mean(x.type(torch.float32), axis=-1)
                                for x in stable_le_0
                            ]
                        )
                        stable_ge_0 = sum(
                            [
                                torch.mean(x.type(torch.float32), axis=-1)
                                for x in stable_ge_0
                            ]
                        )
                        batch_stable_ReLU = stable_le_0 + stable_ge_0

                        relu_accuracy = batch_stable_ReLU / self.model.nb_ReLUs

                        se_str += f"{se}: {relu_accuracy*100:.2f}% "

                        self.train_stable_ReLUs[se] += [relu_accuracy.cpu()]
                else:
                    se_str = "Disabled."
                self.logger.info(f"[Train] Stable ReLUs: {se_str}")
                # print(f"{torch.cuda.memory_allocated() / 1024 / 1024 / 1024:.2f} Gb")
            else:
                if self.train_stable_ReLUs is not None:
                    for x in self.train_stable_ReLUs:
                        self.train_stable_ReLUs[x] += [self.train_stable_ReLUs[x][-1]]

            self.train_loss += [loss.item()]

            gc.collect()
            torch.cuda.empty_cache()

    def _test_epoch(self, epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                output = F.log_softmax(output, dim=1)

                test_loss += F.nll_loss(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        test_accuracy = correct / len(self.test_loader.dataset)
        self.test_accuracy += [test_accuracy]

        self.logger.info(
            f"[Test] epoch: {epoch:3} loss: {test_loss:10.6f}, accuracy: {test_accuracy*100:.2f}%"
        )

        if self.stable_estimators is not None:
            se_str = ""
            for se in self.stable_estimators:
                self.stable_estimators[se].propagate(
                    data=data, test_loader=self.test_loader
                )
                stable_le_0, stable_ge_0 = self.stable_estimators[se].get_stable_ReLUs()
                stable_le_0 = sum(
                    [torch.mean(x.type(torch.float32), axis=-1) for x in stable_le_0]
                )
                stable_ge_0 = sum(
                    [torch.mean(x.type(torch.float32), axis=-1) for x in stable_ge_0]
                )
                batch_stable_ReLU = stable_le_0 + stable_ge_0
                relu_accuracy = batch_stable_ReLU / self.model.nb_ReLUs
                se_str += f"{se}: {relu_accuracy*100:.2f}% "
        else:
            se_str = "Disabled."
        self.logger.info(f"[Test] Stable ReLUs: {se_str}\n")
        # print(f"{torch.cuda.memory_allocated() / 1024 / 1024 / 1024:.2f} Gb")

    def test(self):
        assert self._trained()
        self._setup_train()
        self.logger.info(f"Loading model from {self.model_path[:-5]}.pt")
        self.model.load_state_dict(
            torch.load(f"{self.model_path[:-5]}.pt", map_location=self.device)
        )
        self._test_epoch(0)

    def _plot_train(self):
        # draw training progress plot

        # # stable relu
        if self.train_stable_ReLUs is not None:
            X1 = range(len(list(self.train_stable_ReLUs.values())[0]))
            Y1 = self.train_stable_ReLUs
        else:
            X1 = []
            Y1 = []

        # test accuracy
        X3 = (np.array(range(len(self.test_accuracy))) + 1) * len(self.train_loader)
        Y3 = self.test_accuracy

        # train loss
        X4 = range(len(self.train_loss))
        Y4 = self.train_loss

        # bs points
        X2 = self.train_BS_points
        Y2 = np.zeros(len(X2))

        # max_safe_relu = sum([self.model.activation[layer].view(
        #    self.model.activation[layer].size(0), -1).shape[-1] for layer in self.model.activation])

        p_plot = ProgressPlot()
        # max_safe_relu))
        p_plot.draw_train(
            X1,
            Y1,
            X2,
            Y2,
            (0, self.cfg_train["epochs"] * len(self.train_loader)),
            (0, 1),
        )
        p_plot.draw_accuracy(X3, Y3, X4, Y4, (0, 1))
        p_plot.draw_grid(
            x_stride=self.cfg_train["epochs"] * len(self.train_loader) / 5, y_stride=0.2
        )

        title = f"# {self.__name__}"
        path = os.path.join(self.sub_dirs["figure_dir"], self.__name__ + ".pdf")
        p_plot.save(title, path)
        p_plot.clear()

    # Verification ...
    def _setup_verification(self):
        target_model = (
            self.cfg_verify["target_model"]
            if "target_model" in self.cfg_verify
            else None
        )

        if (
            "save_intermediate" not in self.cfg_train
            or not self.cfg_train["save_intermediate"]
            and target_model not in [None, "last"]
        ):
            raise ValueError(
                "'save_intermediate = true' is required for verification model selection strategy other than 'last' epoch."
            )

        target_epoch = self.Utility.get_target_epoch(target_model, self.train_log_path)

        self.veri_log_path = os.path.join(
            self.sub_dirs["veri_log_dir"],
            f"{self.__name__}_e={target_epoch}_{self.Utility.get_verification_postfix(self.cfg_verify)}.txt",
        )

        veri_log_name = Problem.Utility.get_hashed_name(
            [
                self.cfg_train,
                self.cfg_heuristic,
                self.cfg_verify,
                target_epoch,
                self.seed,
            ]
        )
        self.veri_log_path = os.path.join(
            self.sub_dirs["veri_log_dir"], f"{veri_log_name}.txt"
        )

        return target_epoch

    def _verified(self):
        assert self._trained()
        # TODO: account for verification completion

        return os.path.exists(self.veri_log_path)

    def verify(self):
        cfg = self.cfg_verify
        prop = cfg["property"]
        eps = cfg["epsilon"]
        veri_framework = cfg["verifier"].split(":")[0]
        verifier = cfg["verifier"].split(":")[1]
        debug = cfg["debug"] if "debug" in cfg else None
        save_log = cfg["save_log"] if "save_log" in cfg else None

        target_epoch = self._setup_verification()

        if self._verified() and save_log and not self.override:
            self.logger.info("Skipping verified problem.")
        else:
            if (
                "save_intermediate" not in self.cfg_train
                or not self.cfg_train["save_intermediate"]
            ):
                model_path = self.model_path
            else:
                model_path = f"{self.model_path[:-5]}.{target_epoch}.onnx"
            self.logger.debug(f"Target model: {model_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Missing model file: {model_path}")

            # generate property
            self.logger.info("Generating property ...")
            prop_path = self.artifact.gen_property(
                prop, eps, self.sub_dirs["property_dir"]
            )

            # compose DNNV command
            res_monitor_path = os.path.join(
                os.environ["DNNV"], "tools", "resmonitor.py"
            )
            cmd = f"python3 {res_monitor_path} -T {cfg['time']+RES_MONITOR_PRETIME} -M {cfg['memory']}"

            if veri_framework in ["DNNV", "DNNVWB"]:
                cmd += f" ./tools/run_{veri_framework}.sh"
            else:
                raise NotImplementedError()

            cmd += f" {prop_path} -N N {model_path}"

            if "eran" in verifier:
                cmd += f" --eran --eran.domain {verifier.split('_')[1]}"
            else:
                cmd += f" --{verifier}"

            if debug:
                cmd += " --debug"

            if save_log:
                veri_log_file = open(self.veri_log_path, "a")
            else:
                veri_log_file = sys.stdout

            self.logger.info("Executing DNNV ...")
            self.logger.debug(cmd)
            self.logger.debug(f"Verification output path: {veri_log_file}")

            # Run verification twice to account for performance issues
            # TODO: fix later
            self.logger.info("Dry run ...")
            if save_log:
                open(self.veri_log_path, "w").write("********Dry_Run********\n")
            else:
                print("********Dry_Run********\n")
            sp = subprocess.Popen(
                cmd, shell=True, stdout=veri_log_file, stderr=veri_log_file
            )
            rc = sp.wait()
            assert rc == 0

            self.logger.info("Wet run ...")
            if save_log:
                open(self.veri_log_path, "a").write("********Wet_Run********\n")
            else:
                print("********Wet_Run********\n")
            # 2. Wet run
            sp = subprocess.Popen(
                cmd, shell=True, stdout=veri_log_file, stderr=veri_log_file
            )
            rc = sp.wait()
            assert rc == 0

    def analyze(self):
        # analyze train
        # analyze verification
        assert self._trained()
        self._setup_verification()
        assert self._verified()
        self.logger.debug(f"Analyzing log: {self.veri_log_path}")
        veri_ans, veri_time = self.Utility.analyze_veri_log(
            self.veri_log_path, timeout=self.cfg_verify["time"]
        )
        if veri_ans and veri_time:
            self.logger.info(f"Result: {veri_ans}, {veri_time}s.")
        else:
            self.logger.info(f"Failed.")

    def _save_meta(self):
        ...

    class Utility:
        @staticmethod
        def get_hashed_name(identifiers):
            identifiers_ = []
            for x in identifiers:
                if isinstance(x, dict):
                    x = {i: x[i] for i in sorted(x.keys())}
                identifiers_ += [str(x)]
            return str(hash("".join(identifiers_)) + sys.maxsize + 1)

        def get_name(cfg_train, cfg_heuristic, seed):
            name = f"A={cfg_train['artifact']}"
            name += f"_N={cfg_train['net_name']}"
            epsilon = ""
            for x in cfg_train["stable_estimator"]:
                if x in ["SAD", "NIP", "SIP"]:
                    epsilon = f":_eps={cfg_train['stable_estimator'][x]['epsilon']}"
                    break
            name += epsilon

            def parse_start_end(x):
                return (
                    str(x["start"]).replace(" ", "")
                    + ":"
                    + str(x["end"]).replace(" ", "")
                )

            if cfg_heuristic:
                for h in cfg_heuristic:
                    x = cfg_heuristic[h]

                    if h == "bias_shaping":
                        if x["mode"] == "standard":
                            m = "S"
                        elif x["mode"] == "distribution":
                            m = "D"
                        else:
                            raise NotImplementedError
                        i = f":i{x['intensity']}" if "intensity" in x else ""
                        o = f":o{x['occurrence']}" if "occurrence" in x else ""
                        p = f":p{x['pace']}" if "pace" in x else ""
                        d = f":d{x['decay']}" if "decay" in x else ""

                        name += f"_BS={m}{i}{o}{p}{d}:{parse_start_end(x)}"

                    elif h == "rs_loss":
                        if x["mode"] == "standard":
                            m = "S"
                        else:
                            raise NotImplementedError
                        name += f"_RS={m}:{x['weight']}:{parse_start_end(x)}"

                    elif h == "prune":
                        if x["mode"] == "structure":
                            m = "S"
                        else:
                            raise NotImplementedError

                        if "re_arch" not in x:
                            r = "-"
                        elif x["re_arch"] == "standard":
                            r = "S"
                        elif x["re_arch"] == "last":
                            r = "L"
                        else:
                            raise NotImplementedError

                        name += f"_PR={m}:{r}:{x['sparsity']}:{parse_start_end(x)}"

                    else:
                        assert False
                    # only works with 1 ReLU estimator
                    # TODO: fix me, don't use descriptive naming, use HASH instead.
                    se = f'{list(x["stable_estimator"].keys())[0]}'
                    se += (
                        f':({x["stable_estimator"]["epsilon"]})'
                        if "epsilon" in x["stable_estimator"]
                        else ""
                    )
                name += f":{se}"
            name += f"_seed={seed}"
            return name

        @staticmethod
        def get_verification_postfix(cfg_verify):
            print("Deprecated.")
            p = cfg_verify["property"]
            e = cfg_verify["epsilon"]
            v = cfg_verify["verifier"]
            return f"P={p}_E={e}_V={v}"

        @staticmethod
        def heuristic_enabled_epochwise(epoch, start, end):
            assert type(start) == type(end)
            if isinstance(start, int):
                enabled = start <= epoch <= end
            elif isinstance(start, list):
                enabled = any([x[0] <= epoch <= x[1] for x in zip(start, end)])
            else:
                assert False
            return enabled

        @staticmethod
        def get_target_epoch(target_model, train_log_path):
            lines = [
                x.strip()
                for x in open(train_log_path, "r").readlines()
                if "[Test]" in x
            ]
            test_lines = [x for x in lines if "[Test] epoch" in x]
            relu_lines = [x for x in lines if "[Test] Stable ReLUs" in x]
            assert len(test_lines) == len(relu_lines)
            test_accs = np.array(
                [float(x.strip().split()[-1][:-1]) for x in test_lines]
            )

            if not target_model or target_model == "last":
                target_epoch = len(test_lines)
            elif re.search("^best test accuracy of last .* epochs", target_model):
                x = int(target_model.split()[-2])
                max_idx = np.where(test_accs == np.max(test_accs[-x:]))
                target_epoch = max_idx[0][0] + 1

            elif re.search("^is .*", target_model):
                target_epoch = int(target_model.split()[-1])

            elif target_model.startswith("best") or target_model.startswith("top"):
                assert False, "Need to fix. Which estimator to use?"
                acc_test = np.array([float(x.strip().split()[-1][:-1]) for x in lines])
                acc_relu = np.array(
                    [float(x.strip().split()[-1][:-1]) / 100 for x in lines]
                )

                if target_model.startswith("best"):
                    if target_model == "best test accuracy":
                        target_epoch = np.argmax(acc_test) + 1
                    elif target_model == "best relu accuracy":
                        target_epoch = np.argmax(acc_relu) + 1
                    else:
                        assert False
                else:
                    threshold = float(target_model.split()[1])

                    if target_model.endswith("test accuracy"):
                        max_test = np.max(acc_test)
                        # print(acc_test, threshold, max_test)
                        candidates = np.where(acc_test >= max_test - threshold)
                        # print(candidates)
                        max_relu = np.max(acc_relu[candidates])
                        # print(max_relu)

                        candidates = np.where(acc_relu == max_relu)
                        # print(candidates)
                        max_test = np.max(acc_test[candidates])
                        # print(max_test)
                        set_relu = set(np.where(acc_relu == max_relu)[0].tolist())
                        set_test = set(np.where(acc_test == max_test)[0].tolist())
                        # print(set_relu)
                        # print(set_test)
                        final_candidates = set_relu.intersection(set_test)
                        # assert len(final_candidates) == 1
                        target_epoch = final_candidates.pop() + 1

                    elif target_model.endswith("relu accuracy"):
                        max_relu = np.max(acc_relu)
                        candidates = np.where(acc_relu >= max_relu - threshold)
                        # print(candidates)
                        max_test = np.max(acc_test[candidates])
                        # print(max_relu)

                        candidates = np.where(acc_test == max_test)
                        # print(candidates)
                        max_relu = np.max(acc_relu[candidates])
                        # print(max_test)
                        set_test = set(np.where(acc_test == max_test)[0].tolist())
                        set_relu = set(np.where(acc_relu == max_relu)[0].tolist())
                        # print(set_relu)
                        # print(set_test)
                        final_candidates = set_test.intersection(set_relu)
                        # assert len(final_candidates) == 1
                        target_epoch = final_candidates.pop() + 1
                        # print(target_epoch)
                    else:
                        assert False
            else:
                assert False, f"Unknown target_model: {target_model}"

            return target_epoch

        @staticmethod
        def analyze_veri_log(log_path, timeout=None):
            lines = open(log_path, "r").readlines()
            lines.reverse()
            # wet_run_idx = lines.index("********Wet_Run********\n")
            # lines = lines[: wet_run_idx + 1]
            veri_ans = None
            veri_time = None
            for l in lines:
                if "  result: " in l:
                    if "result: NeurifyError(Return code: -11)" in l:
                        veri_ans = "error"
                    elif "Invalid counter example" in l:
                        veri_ans = "error"
                    elif "Error(Return code: 1)" in l:
                        veri_ans = "error"
                    else:
                        veri_ans = l.strip().split()[-1]
                    break

                elif "  time: " in l:
                    veri_time = float(l.strip().split()[-1])

                elif "Timeout" in l:
                    veri_ans = "timeout"
                    veri_time = float(l.strip().split()[-3])
                    break

                elif "Out of Memory" in l:
                    veri_ans = "memout"
                    veri_time = None  # float(l.strip().split()[-3])
                    break

                elif "CANCELLED" in l:
                    break

            # assert veri_ans
            # assert veri_time
            if timeout and veri_time:
                if veri_time > timeout:
                    veri_ans = "timeout"
                    veri_time = timeout

            return veri_ans, veri_time
