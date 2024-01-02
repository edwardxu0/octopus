#!/bin/bash

# For the M2 artifact
python -m octopus+ s1/s1.toml s1/s1.py A --go
# For the M6 artifact
python -m octopus+ s2/s2.toml s2/s2.py A --go
# For the C3 artifact
python -m octopus+ s3/s3.toml s3/s3.py A --go
