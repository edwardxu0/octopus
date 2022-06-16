## 1. Challenges
1. Unable to recover training.
2. Combine heuristics.

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

### 2.3. W2[D]:
* [x] a unified way to control heuristics pre/during/post training.
* [x] a unified way to save logs/models/meta-data.
* [x] a unified pipeline with DNNV/analysis.
* [x] Enhance efficiency of RSLoss/BS.
* [x] Multiple ReLU estimations. TD/VS/IP ...
* [x] Explore when to apply RSLoss/BS.
* [x] Explore RSLoss/BS to verification.

### 2.4. W3[D]:
* [x] Logging with octopus+.
* [x] Save slurm scripts separately.
* [x] Separate host and slurm logs.
* [ ] Refine BS from numpy to torch.
* [x] CUDA AMP.
* [x] BS decay/weights of RSLoss.
* [x] Explore combinations of RSLoss/BS to verification.

### 2.5. W4[D]:
* [x] BS distribution.
* [x] Refine pruning.
* [x] Study top 1, 3, 5.

### 2.5. W5[D]:
* [ ] Report.
* [ ] 
* [ ] 