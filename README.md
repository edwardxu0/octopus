# OCTOPUS
Obsessive Crazy Training-Oriented Proving Ultimate System(OCTOPUS) for DNN verification.

## How to use?
### Setup
1. Initialize environment, `python3 -m venv .venv`.
2. Activate environment, `source .env.d/openenv.sh`.
3. Install dependencies, `pip install -r .env.d/requirements.txt`.

### Execution
1. Activate environment, `source .env.d/openenv.sh`.
2. Use `octopus config task` to run octopus.

    **train**: train network with the heuristics.

    **verify**: TODO
    **analyze**: TODO

3. `config`
    TODO: describe here.

## TODO:
* [x] explore ways to incoordinate the unsafe ReLUs into back propagation. [D,N]
* [x] explore safe/unsafe ReLUs according to each image/label/class.
* [x] add parameters regulating the heuristics, e.g. % of unsafe ReLU, labeled unsafe ReLU, and grade of unsafe ReLU.
* [x] explore sparse networks about pruning techniques vs. original networks in terms of Safe/Unsafe ReLUs.
* [x] bias shaping for all layers. [D]
* [x] explore Safe/Unsafe ReLUs in terms of properties. [N]
* [x] refinement phase. [D]
* [ ] early stopping criteria. [N]
* [x] explore verification vs bias shaping. [D]

## Plan W1[D]:
* [ ] switch from arguments to config files with toml.
* [ ] setup interfaces for BS/RSLoss/Pruning/ReArc/AdvTrain ...
* [ ] re-architecture the MODELS to accept unified functions.
* [ ] refactor code base to support all the changes.
* [ ] fix RS loss.
* [ ] a unified way to control heuristics pre/during/post training.
* [ ] a unified way to save logs/models/meta-data.
* [ ] a unified pipeline with DNNV/analysis.