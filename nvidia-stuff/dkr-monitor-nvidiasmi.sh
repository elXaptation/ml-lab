#!/usr/bin/bash
docker run --rm -it --gpus all nvidia/cuda:11.2.1-base /bin/bash -c "watch -d -n5 nvidia-smi"

