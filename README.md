# Reducing-the-number-of-unsafe-ReLu

## How to use?

1. Run `python src/main.py --help` for detailed usage.

2. Use `python src/main.py [task] ...` to do some analyze over the pre activation values.

    **train**: train network with the "--save-model" flag to save models and raw data into the "results" path.

    **analyze_sign**: analyze the pre activation values during the training and testing phase. Results are saved at "./meta/activation_train.txt" and "./meta/activation_test.txt". It computes the sign of pre activation values in the order of: positive, negative, and zero.

    **analyze_bounds**: analyze the pre activation values during the training and testing phase. Results are saved at "./meta/train_bounds.csv" and "./meta/test_bounds.csv". It computes the number of safe and unsafe ReLUs during training and testing.
    
    **analyze_class**: analyze the pre activation values seperated by each output class during the training and testing phase. Results are saved at "./meta/activation_byclass_train.txt" and "./meta/activation_byclass_test.txt". It combines the pre activation values for similar class inputs and computes the sign of pre activation values in the order of: positive, negative, and zero.
    
    **analyze_class_bounds**: analyze the pre activation values seperated by each output class during the training and testing phase. Results are saved at "./meta/train_bounds_class.csv" and "./meta/test_bounds_class.csv". It combines the pre activation values for similar class inputs computes the number of safe and unsafe ReLUs during training and testing.

3. Run RURH(Reducing Unsafe ReLU Heuristics) with the `rurh` option.
`python ./src/main.py train --rurh standard`
    **None**: no heuristics.
    **standard**: basic heuristic that modifies the bias of all global unsafe ReLUs.
    **...**: ...

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