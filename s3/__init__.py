import numpy as np

platform = "local"
sleep_train = 0
sleep_verify = 0

seeds = [*range(5)]
props = [*range(10)]

epsilons = np.linspace(18, 26, 5) / 10000

print(f"seeds: |{len(seeds)}|, props: |{len(props)}|")
print(f"epsilons: |{len(epsilons)}|, {epsilons}")

verifiers = ["SH:abcrown2", "SH:mnbab", "SH:nnenum"]