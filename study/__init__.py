
import numpy as np


sleep_time = 0

# train_nodes = ["cheetah01"]  # change octopus+ L:267
train_nodes_ex = "ai01,ai07,ai08,lynx07,lynx08,lynx09,lynx10,sds01,sds02,lotus,titanx01,titanx02,titanx03,titanx04,titanx05,titanx06,lynx11,lynx12,affogato11,affogato12,adriatic01,adriatic02,adriatic03,adriatic04,adriatic05,adriatic06,ristretto02,ristretto03,ristretto04,cheetah01,jaguar01"

veri_nodes = ["doppio" + x for x in ["01", "02", "03", "04", "05"]]

seeds = [*range(3)]
props = [*range(10)]

epsilons = np.linspace(2, 10, 5) / 100

verifiers = ["DNNV:eran_deeppoly", "DNNV:nnenum", "DNNVWB:neurify", "DNNV:marabou"]
