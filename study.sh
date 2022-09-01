#!/usr/bin/bash
python -m octopus+ e5/e5p3a.toml e5/e5p3a.py {$1} --slurm --go
python -m octopus+ e5/e5p3b.toml e5/e5p3b.py {$1} --slurm --go

python -m octopus+ e6/e6p3a.toml e6/e6p3a.py {$1} --slurm --go
python -m octopus+ e6/e6p3b.toml e6/e6p3b.py {$1} --slurm --go

python -m octopus+ e7/e7p3a.toml e7/e7p3a.py {$1} --slurm --go
python -m octopus+ e7/e7p3b.toml e7/e7p3b.py {$1} --slurm --go

python -m octopus+ e8/e8p3a.toml e8/e8p3a.py {$1} --slurm --go
python -m octopus+ e8/e8p3b.toml e8/e8p3b.py {$1} --slurm --go