#!/bin/bash

# 5.3.1 Generate plots
./tools/star_gate.py scatter --root results_precompiled m2 m6 c3

# 5.3.2 Generate table of Test Accuracy and Stable Neurons
./tools/star_gate.py st --root results_precompiled m2 m6 c3

# 5.3.2. Generate table of Problems Solved and Verification Time
./tools/star_gate.py vt --root results_precompiled m2 m6 c3
