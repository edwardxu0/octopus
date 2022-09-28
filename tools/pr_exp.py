#!/usr/bin/env python
import os
import sys


def main():

    train_nodes_ex = "ai01,ai07,ai08,lynx08,lynx09,lynx10,sds01,sds02,lotus,titanx01,titanx02,titanx03,titanx04,titanx05,titanx06,lynx11,lynx12,affogato12,adriatic01,adriatic02,adriatic03,adriatic04,adriatic05,adriatic06"

    ests = ["sdd", "sad", "nip", "sip"]
    items = [*range(1, 6)]

    for i in items:
        for e in ests:

            cmd = f"python -m octopus conpr2/pr_{e}{i}.toml T --debug --override"

            log_path = f"pr_exp/log/{e}{i}.txt"
            lines = [
                "#!/bin/sh",
                f"#SBATCH --job-name=O.T",
                f"#SBATCH --output={log_path}",
                f"#SBATCH --error={log_path}",
                f"#SBATCH --partition=gpu",
                "#SBATCH --gres=gpu:1",
                f"#SBATCH --exclude={train_nodes_ex}",
                "cat /proc/sys/kernel/hostname",
                "source .env.d/openenv.sh",
                "echo CUDA_DEVICE: $CUDA_VISIBLE_DEVICES",
                cmd,
            ]

            lines = [x + "\n" for x in lines]
            open("temp.slurm", "w").writelines(lines)

            slurm_cmd = "sbatch temp.slurm"

            if sys.argv[1] == 'slurm':
                print(slurm_cmd)
                os.system(slurm_cmd)
            else:
                print(cmd)
                os.system(cmd)


if __name__ == "__main__":
    main()
