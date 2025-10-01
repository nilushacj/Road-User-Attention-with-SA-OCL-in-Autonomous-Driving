#!/bin/bash

source activate slot-inference-env

srun python -m videosaur.inference_unsupervised --config /PATH/TO/DIR/configs/inference/waymo_tfds.yml #TODO: change