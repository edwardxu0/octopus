import numpy as np

train_sleep_time = 20
verify_sleep_time = 0

# train_nodes = ["cheetah01"]  # change octopus+ L:267
train_nodes_ex = "ai01,ai07,ai08,lynx07,lynx08,lynx09,lynx10,sds01,sds02,lotus,titanx01,titanx02,titanx03,titanx04,titanx05,titanx06,lynx11,lynx12,affogato11,affogato12,affogato15,adriatic01,adriatic02,adriatic03,adriatic04,adriatic05,adriatic06,cheetah01,cheetah03,jaguar01,jaguar02,jaguar03,jaguar04"

veri_nodes = ["doppio" + x for x in ["01", "02", "03", "04", "05"]]

nb_train_nodes = 1
nb_verify_nodes = 1

seeds = [*range(3)]
props = [*range(5)]

# epsilons = np.linspace(2, 10, 5) / 100

epsilons = np.linspace(0.5, 5, 10) / 256

# verifiers = ["DNNV:eran_deeppoly", "DNNV:nnenum", "DNNV:neurify", "DNNV:marabou"]
verifiers = ["DNNV:eran_deeppoly", "DNNV:nnenum", "DNNVWB:neurify", "DNNV:marabou"]
# verifiers= ["DNNV:marabou"]
