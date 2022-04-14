# OCTOPUS
Oh-Crazy Training-Oriented Provable Unified System(OCTOPUS) for DNN verification.

## 1.How to use?
### 1.1. Setup
1. Initialize environment, `python3 -m venv .venv`.
2. Activate environment, `source .env.d/openenv.sh`.
3. Install dependencies, `pip install -r .env.d/requirements.txt`.

### 1.2. Singular Execution
1. Activate environment, `source .env.d/openenv.sh`.
2. Use `octopus CONFIG_FILE TASK [OPTIONS]` to run octopus.

    **TASKS**

        train: train network with the heuristics.

        verify: verify the trained network.

        analyze: analyze training and verification.

        all:  sequentially run the above three tasks.

    **CONFIG_FILE**
    TODO: describe here.

    **OPTIONS**

        --seed x: seed pytorch and numpy with x. Default is 0.
        --debug: print debug log.
        --dumb: silence.

### 1.3. Benchmark Execution

## 2. PLAN:
#### 2.1. Attic:
* [x] explore ways to incoordinate the unsafe ReLUs into back propagation. [D,N]
* [x] explore safe/unsafe ReLUs according to each image/label/class.
* [x] add parameters regulating the heuristics, e.g. % of unsafe ReLU, labeled unsafe ReLU, and grade of unsafe ReLU.
* [x] explore sparse networks about pruning techniques vs. original networks in terms of Safe/Unsafe ReLUs.
* [x] bias shaping for all layers. [D]
* [x] explore Safe/Unsafe ReLUs in terms of properties. [N]
* [x] refinement phase. [D]
* [ ] early stopping criteria. [N]
* [x] explore verification vs bias shaping. [D]

### 2.2. W1[D]:
* [x] switch from arguments to config files with toml.
* [x] setup interfaces for BS/RSLoss/Pruning/ReArc/AdvTrain ...
* [x] re-architecture the MODELS to accept unified functions.
* [x] refactor code base to support all the changes.
* [x] fix RS loss.
* [x] a unified way to control heuristics pre/during/post training.
* [x] a unified way to save logs/models/meta-data.
* [x] a unified pipeline with DNNV/analysis.

### 2.3. W2[D]:
* [x] Enhance efficiency of RSLoss/BS.
* [x] Multiple ReLU estimations. TD/VS/IP ...
* [x] Explore when to apply RSLoss/BS.
* [x] Explore RSLoss/BS to verification.
* [ ] BS decay/weights of RSLoss.
* [ ] Explore combinations of RSLoss/BS to verification.

### 2.4. W3[D]: