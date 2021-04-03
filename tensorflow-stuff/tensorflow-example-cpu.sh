#!/usr/bin/bash
docker run -it --rm tensorflow/tensorflow python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
