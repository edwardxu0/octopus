import numpy as np

sleep_train = 0
sleep_verify = 0
platform = "slurm"

# train_nodes = ["cheetah01"]  # change octopus+ L:267
# train_nodes_ex = "ai01,ai07,ai08,lynx07,lynx08,lynx09,lynx10,sds01,sds02,lotus,titanx01,titanx02,titanx03,titanx04,titanx05,titanx06,lynx11,lynx12,affogato11,affogato12,affogato15,adriatic01,adriatic02,adriatic03,adriatic04,adriatic05,adriatic06,ristretto02,ristretto03,ristretto04,cheetah01,jaguar01,jaguar02,jaguar03"
# train_nodes_ex = "ai05,ai09,ai10,jinx01,jinx02,titanx03"
train_nodes = ["jaguar03"]

veri_nodes = ["doppio" + x for x in ["01", "02", "03", "04", "05"]]

seeds = [*range(1)]
props = [*range(50)]


# epsilons = np.linspace(2, 10, 5) / 100  # mnist
epsilons = np.linspace(2, 100, 50) / 1000  # mnist
# epsilons = np.linspace(2, 2, 1) / 1000  # mnist
print(epsilons, print(len(epsilons)))

# epsilons = np.linspace(0.5, 5, 10) / 256 # cifar
# epsilons = np.linspace(3, 3, 1) / 100  # mnist

# verifiers = ["DNNV:eran_deeppoly", "DNNV:nnenum", "DNNV:neurify", "DNNV:marabou"]
# verifiers = ["SH:abcrown"]
# verifiers = ["SH:mnbab"]

verifiers = ["SH:abcrown", "SH:mnbab", "SH:abcrown2", "SH:nnenum"]