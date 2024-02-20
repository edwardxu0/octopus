# [The OCTOPUS Framework]Training for Verification: Increasing Neuron Stability to Scale DNN Verification
Oh-Crazy Training-Oriented Provable Unified System(OCTOPUS) for DNN verification, is a framework that allows training neural networks for faster verification, as defined in the [Training for Verification: Increasing Neuron Stability to Scale DNN Verification](link_to_paper_to_be_added) paper. The purpose of this research artifact is to demonstrate the functionality of the proposed methods described in the paper and to replicate the evaluation results presented in the paper. This artifact contains 1)guidelines on installing the OCTOPUS framework with scripts; 2)a tutorial on training an example neural network with stabilizers enabled and verifying the resulting network over robustness property; 3)instructions and scripts to fully replicate the research results from scratch; and 4)scripts to generate the plots and tables used in the paper and appendix. The tool is written in the Python language. We expect users to be familiar with the Python language, neural network training, and neural network verification.

## 1. Overview
### 1.1 TL;DR
    To learn about OCTOPUS, we recommend starting with the following steps:
    - For installation of the framework, follow Sections 2.1, steps 1, 2, 3, 4, 5, and 8.
    - For trying out the OCTOPUS tutorial, follow the complete Section 3.
    - For result replication, follow Section 5.3.
    - (Optional)You are welcome to read and test the full set of instructions. Just be aware that to fully replicate the results from scratch requires large amounts of computing resources and disk space.

### 1.2 Directories
  What are the directories in this repo?
  ```
    - configs: stores the configuration files of OCTOPUS for test purposes.
    - data: dataset for network training.
    - lib: external libraries used by OCTOPUS.
    - octopus: source code of OCTOPUS.
    - octopus+: source code of OCTOPUS+.
    - results: stores the results.
    - results_precompiled: precompiled and cached results from the raw data in the pandas dataframe format used in the paper as well as expected results to the tutorials in this readme.
    - results_figs: stores the generated plots.
    - s1/s2/s3: scripts and configs to execute the full set of experiments used in the evaluation section of the paper.
    - scripts: scripts to install, test, and run experiments.
    - tools: miscellaneous tools to make plots and tables.
  ```

## 2. Installation

### 2.1 Quick Installation Guidelines
We have pre-cloned all the necessary repositories in this artifact for ease of use. Please refer to the full installation guide for future updates of the tool and its dependencies.
1. Install [anaconda](https://docs.anaconda.com/free/anaconda/install/linux/) or [miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)(Recommended for space-saving) by following their instructions. Make sure that `which conda` returns the correct answer, e.g., "/home/tacas23/miniconda3/bin/conda". If you added a new virtual disk, you can mount the disk and install Conda on it for space reasons, e.g., "/media/tacas23/work/miniconda3".

2. Set up the `$CONDA_HOME` environment variable to be the home directory of the Conda installation, e.g., `export CONDA_HOME=/home/tacas23/miniconda3`. Check that `echo $CONDA_HOME` returns the correct result, e.g., "/home/tacas23/miniconda3". Do this every time you open a terminal or add it to the bashrc file.

3. Install the OCTOPUS Conda environment: `./scripts/2.1_install_octopus.sh` It may take some time, depending on the Internet connection. If corrupted packages are found, try to remove them `rm -rf $CONDA_HOME/pkgs/[CORRUPTED]` and rerun the installation script. The following indicates that the environment has been successfully installed.
    ```
    done                                                                                   
    
    #                                                                
    # To activate this environment, use
    #
    #     $ conda activate octopus
    #
    # To deactivate an active environment, use
    #
    #     $ conda deactivate
   ```
   
4. Install the SwarmHost DNN verification framework: `./scripts/2.1_install_swarmhost.sh`  After finishing, we can install the following verification tools. Watch for space limitations if not adding a virtual disk, you can use `./scripts/0_clean_cache.sh` to free up space taken by conda and pip.

5. (Recommended) NNEnum: `./scripts/2.1_install_nnenum.sh`

6. (Optional) ABCROWN: `./scripts/2.1_install_abcrown.sh`

7. (Optional) MN-Bab: `./scripts/2.1_install_mnbab.sh`

8. Run `conda env list` will list the installed environments. 
   ```
    # conda environments:
    #
    base                  *  /home/tacas23/miniconda3
    abcrown                  /home/tacas23/miniconda3/envs/abcrown
    mnbab                    /home/tacas23/miniconda3/envs/mnbab
    nnenum                   /home/tacas23/miniconda3/envs/nnenum
    octopus                  /home/tacas23/miniconda3/envs/octopus
    swarmhost                /home/tacas23/miniconda3/envs/swarmhost
   ```
   Note that this doesn't guarantee that the environments are correctly installed in some extreme cases. They might still be broken due to corrupted packages when downloading, keyboard interruption while installing, etc. If training or verification doesn't run as expected, try reinstalling the environments using the above scripts.
   
   **Note Sections 2.2 and 2.3 are for general installation purposes; no need to follow for this artifact.**

### 2.2 Full Guidelines on Installation for Training(Ignore)
1. Install [anaconda](https://docs.anaconda.com/free/anaconda/install/linux/) or [miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) by following their instructions. Make sure `conda` is in system path and set up the `$CONDA_HOME` environment variable to be the home directory of Conda, e.g., `export CONDA_HOME=[PATH_TO_CONDA_MAIN_FOLDER]`
2. Install make and cmake: `sudo apt install make cmake`
3. Install the OCTOPUS Conda environment: `conda env create -n octopus -f .env.d/octopus.yml`
5. Install the optional dependencies to the `lib` folder.
6. (Optional) Install SIP estimator: `git clone https://github.com/edwardxu0/SIP.git lib/SIP`
7. (Optional) Install ALR/ALRo estimator: `git clone https://github.com/Verified-Intelligence/auto_LiRPA lib/auto_LiRPA`
8. (Optional) Install NeuralSAT: `git clone https://github.com/dynaroars/neuralsat lib/neuralsat`
### 2.3 Full Guidelines on Installation of the Verification Framework(Ignore)
1. Install the SwarmHost verification framework: `git clone https://github.com/edwardxu0/SwarmHost.git lib/SwarmHost`. Install its Conda environment `./scripts/2.1_install_swarmhost.sh`. Then we can install the following verifiers and their Conda environments.

2. (Recommended) NNenum: `git clone https://github.com/stanleybak/nnenum.git lib/SwarmHost/lib/nnenum` and install the Conda environment `./scripts/2.1_install_nnenum.sh`.

3. (Optional) ABCrown: `git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN.git lib/SwarmHost/lib/abcrown` and install the Conda environment `./scripts/2.1_install_abcrown.sh`.

4. (Optional) Mn-Bab: `git clone https://github.com/eth-sri/mn-bab.git lib/SwarmHost/lib/mnbab `  and checkout the main branch, `git checkout main`, and and install the Conda environment `./scripts/2.1_install_mnbab.sh`.

   



## 3. Tutorial on OCTOPUS

1. Activate environment, `source .env.d/openenv.sh`.

2. Use `octopus -h` to access the helps. You are expected to see the following:
   ```
       ____  ________________  ____  __  _______
      / __ \/ ____/_  __/ __ \/ __ \/ / / / ___/
    / / / / /     / / / / / / /_/ / / / /\__ \ 
    / /_/ / /___  / / / /_/ / ____/ /_/ /___/ / 
    \____/\____/ /_/  \____/_/    \____//____/  
                                                
    usage: octopus [-h] [--seed SEED] [--result_dir RESULT_DIR] [--override] [--debug] [--dumb] [--version] configs {T,V,A,AA}
   
    Generate Verifiable Networks
   
    positional arguments:
      configs               Configurations file.
      {T,V,A,AA}       Select tasks to perform, including [T]rain, [V]erify, [A]nalyze, [AA]ll 'bove.
   
    options:
      -h, --help            show this help message and exit
      --seed SEED           Random seed.
      --result_dir RESULT_DIR
                            Result directory.
      --override            Overrides training/verification tasks.
      --debug               Print debug log.
      --dumb                Silent mode.
      --version             show program's version number and exit
   ```

### 3.1. Flags

OCTOPUS uses this format to execute `octopus [CONFIG_FILE] [TASK] [OPTIONS]` .

    **CONFIG_FILE**
      The main configuration file in the "toml" format that contains the training parameters, stabilizers, and verification parameters, e.g., "configs/test.toml"
    
    **TASK**
        T: train the network with stabilizers defined in the config file.
        V: verify the trained network.
        A: analyze training and verification. (The save_log option needs to be on in the config file.)
        AA: sequentially run the above three tasks.
    
    **OPTIONS**
        --seed x: seed pytorch and numpy with x. Default is 0.
        --debug: print debug log.
        --dumb: silence.
        ...

### 3.2 Train Networks with Neuron Stabilizers
1. Use `octopus configs/test.toml T` to train a network with specs defined in the file: `config/test.toml`. In this case, we will train a fully connected network on the MNIST dataset with 2 layers and 32 neurons in each layer. It enables three neuron stabilizers that are run in parallel: Bias Shaping with the SDD estimator, RS Loss with the SIP estimator, and Stable Pruning with the ALR estimator.
2. If training is successful, a result folder named `results/test` will be generated. Check the `results/test/figure/*.pdf` file after finishing at least one epoch for the training information, e.g., training loss, test accuracy, and number of estimated stable neurons.
3. When training is finished, i.e., the command line prompts `Mission Complete.`, check if there exist ONNX model files in the `model` folder, if so, we can now proceed to verification.  Here are some expected training results: `results_precompiled/3.2_test`.
4. Use `octopus configs/test.toml V` to run verification. In this case, we use the "NNenum" verifier, and the results will be either SAFE(UNSAT)/UNSAFE(SAT) based on the resulting network. In my case, the result is `Result: network is SAFE`. In some extreme cases, the network will timeout depending on the resource. The expected result can be found at `results_precompiled/3.2_verification_log_nnenum.txt`. 
5. Now we have successfully trained and verified a network with stabilizers turned on. Feel free to tweak the settings in the config file `configs/test.toml` to try other parameters, e.g., change the network architecture; disable a stabilizer by commenting out its relative sections; tune parameters of stabilizers; and try other verifiers(requires installation of the tool). Verification results of "ABCrown" and "MNBab" can be found at `results_precompiled/3.2_verification_log_abcrown.txt` and `results_precompiled/3.2_verification_log_mnbab.txt`. You can also try the `configs/cifar2020_2_255.toml` config to train a CIFAR network, and verifying the CIFAR network requires more memory.
6. When you are checking the results folder for information, here is a list of useful descriptions:
    ```
    The file names are the hashes of the training and verification parameters.
    -figure: saves the training progress such as loss/accuracy plots
    -model: keeps the neural network models;
    -train_log: keeps the training logs(if is saved)
    -property: saves the vnnlib properties
    -veri_configs: stores the configuration file of verifiers(if is needed)
    -veri_log: keeps the verification log(if is saved)
    ```

## 4. OCTOPUS+
OCTOPUS+ is a benchmarking tool for OCTOPUS. It can be used to execute OCTOPUS benchmarks that systematically vary the datasets, network architectures, stabilizer, estimator, epsilons, properties, verifiers, and seeds. We used this tool for the experiments presented in the evaluation of the paper.

1. Activate environment, `source .env.d/openenv.sh`.
2. Use `octopus+ -h` for the help menu.

### 4.1. Flags

OCTOPUS+ uses this format to execute `octopus+ [BASE_CONFIG] [BENCHMARK_SETTINGS] [TASK] [OPTIONS]`.

    **BASE_CONFIG**
       The same .toml file that is used in OCTOPUS for training a single network, e.g., "s1/s1.toml"
    
    **BECNHMARK_SETTINGS**
       The benchmark settings that define the parameters to vary in the benchmark, e.g., "s1/__init__.py" and "s1/s1.py".
    
    **TASK**
        T: train all networks defined in the benchmark.
        V: verify all the properties defined in the benchmark.
        A: analyze training and verification results.
        
    **OPTIONS**
        --override            Overrides existing train/verify/analyze tasks.
        --go                  Dry run or not?
        --debug               Print debug log.
        --dumb                Silent mode.

### 4.2 Try out OCTOPUS+
The following steps can be used to execute the experiments on the M2 artifact that was used in the paper. Note that we do NOT expect those commands to finish in a reasonable amount of time on a local machine, due to training hundreds of neural networks and verifying tens of thousands of properties. Note that the training process also requires an nVidia GPU that supports CUDA.

1. Use `octopus+ s1/s1.toml s1/s1.py T --go` to train neural networks the benchmark.
2. Use `octopus+ s1/s1.toml s1/s1.py V --go` to verify the problems in the benchmark
3. Use `octopus+ s1/s1.toml s1/s1.py A --go` to analyze the raw results.


## 5 Result Replication
It requires a large amount of computing power and time to replicate our results presented in the paper starting from scratch. It took us more than 1,858 hours to train the neural networks with GTX 1080 ti GPUs and 1,052 hours to verify the properties on clusters with Intel Xeon Gold 6130 CPUs. However, we have collected all the raw results and stored them in the `results` folder for further analysis.

### 5.1 Full Instructions [Not Recommended]
If anyone is really interested in replicating the results from scratch, please use the following script:
`./scripts/5.1_replicate_results_from_scratch.sh`

### 5.2 Analyze the Raw Results [Not Recommended]
After fully training and verifying the benchmarks in 5.1, use the following command to analyze the results:
`./scripts/5.2_analyze_results_from_scratch.sh`

We have saved all the training and verification log files in the `results` folder. Note that we have to omit the neural network model files due to space issues. To fully analyze the results, it requires thousands of networks with their intermediate snapshots, which count up to 100+Gb space.

To try out the result analyzer in OCTOPUS+ without training and verification from scratch, go to the `result` folder, unzip the `s?.zip` files, and execute the above command. This also takes a few minutes, depending on the hardware, due to the large number of test cases. The expected log file can be found at `results_precompiled/5.2_analyze_raw_results.txt`, and some expected files can be found at `results_precompiled/5.2_results`. 

### 5.3 Analyze the Pre-compiled and Cached Results [Recommended]
To be able to reasonably replicate the results as shown in the paper and appendix, we have analyzed, pre-compiled, and cached the raw results into the PANDAS dataframe format stored in `results_precompiled\*feather`. We can try to generate the plots and tables used in the paper.

#### 5.3.1 Generating Plots
To make all the plots used in the paper, including, stable neurons vs training, problems solved vs training, training time, and verification speedups, use the following tool:
`./tools/star_gate.py scatter --root results_precompiled m2 m6 c3`
The generated plots will be saved to the `results_figs` folder. The expected resulting plots can be found at `results_precompiled/5.3.1_Plots`.

Here are some guidelines on how to read the plot file names:
```
[Aggregation](Prefix)(_Midfix)(_Posfix)
  -Aggregation: aggregates on network architectures and/or verifiers
    -A: aggregates on both, i.e., an average on all benchmarks
    -N: aggregates on the verifier, i.e., an average on various network architectures
    -V: aggregates on the network architectures i.e., an average on various verifiers
    -NV: no aggregation.

  -Prefix: type of plot
    -SN: stable neurons vs. test accuracy
    -SP: solved problems vs. test accuracy
    -VS: verification speedup

  -Midfix: the parameter that did not aggregate on network architectures and/or verifiers, i.e., M2, M6, and C3; a-b-CROWN, MN-Bab, and NNEnum.

  -Postfix: misc
    -s: means scatter plot
    -l: log scale
    -_: '_' in the end means no legend for the plot
```

#### 5.3.2 Generating Table
To replicate the tables used in the appendix, use the following commands:
1. Test Accuracy and Stable Neurons, ``./tools/star_gate.py st --root results_precompiled m2 m6 c3``
2. Problems Solved and Verification Time, `./tools/star_gate.py vt --root results_precompiled m2 m6 c3`

The expected results can be found at `./results_precompiled/5.3.2_training_table.txt` and `./results_precompiled/5.3.2_verification_table.txt` respectively.

## Acknowledgements
This material is based in part upon work supported by National Science Foundation awards 1900676, 2019239, 2129824, 2217071, and 2312487.

We greatly appreciate your enthusiasm for OCTOPUS. Should you need any assistance or guidance in utilizing or expanding OCTOPUS, please feel free to reach out to us.
