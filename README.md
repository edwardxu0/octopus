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

### 1.3. OCTOPUS+
OCTOPUS+ is a OCTOPUS benchmark tool.


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