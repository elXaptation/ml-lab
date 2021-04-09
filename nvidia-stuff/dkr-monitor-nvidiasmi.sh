#!/usr/bin/bash
docker run --rm -it --gpus all nvidia/cuda:11.2.1-base nvidia-smi -l 2 

