# OCTOPUS
Oh-Crazy Training-Oriented Provable Unified System(OCTOPUS) for DNN verification, is a framework that allows training neural networks for faster verification, as defined in the [Training for Verification: Increasing Neuron Stability to Scale DNN Verification](link) paper.


## 1. Overview

![](overview.png|width=500)


## 2. How to use OCTOPUS?
### 2.1. Setup
#### 2.1.1 Training Setup
1. Install dependencies: `python3`
2. Initialize environment: `python3 -m venv .venv`.
3. Activate environment: `source .env.d/openenv.sh`.
4. Install packages: `pip install -r .env.d/requirements.txt`.
5. (Optional) Install SIP: `git clone `
#### 2.1.2 Verification Setup
1. Install DNNV at path: `./lib/DNNV`.

### 2.2. OCTOPUS as Singular Execution
1. Activate environment, `source .env.d/openenv.sh`.
2. Use `octopus CONFIG_FILE TASK [OPTIONS]` to run octopus.

    **TASKS**
        T(rain): train network with stabilizers defined in the .

        verify: verify the trained network.

        analyze: analyze training and verification.

        all:  sequentially run the above three tasks.

    **CONFIG_FILE**
        
        TODO: describe here.

    **OPTIONS**

        --seed x: seed pytorch and numpy with x. Default is 0.
        --debug: print debug log.
        --dumb: silence.

### 2.3. OCTOPUS+
OCTOPUS+ is a OCTOPUS benchmark tool.

### 2.4. 

## 2. Features
* [ ] Network
  * [x] FC Nets
  * [ ] Conv Nets
* [x] Heuristics
  * [x] Bias shaping
    * [x] greedy distance
    * [ ] cumulative distance
    * [ ] greedy distribution: raw count
  * [x] RS loss
  * [x] Pruning
* [ ] ReLU estimations: 
  * [ ] Training distribution
  * [ ] PGD distribution
  * [ ] Interval propagation
  * [ ] Advanced Interval propagation
  * [ ] Symbolic Interval propagation
* [ ] Verification
  * [x] DNNV
  * [ ] DNNF


## 3. Known Issues
1. CUDA AMP doesn't work with RS Loss.