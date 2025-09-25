#!/bin/bash
#SBATCH --time=30:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:v100:1  
#SBATCH --output=videosaur-inference-%j.log #TODO: change
module load scicomp-python-env/2024-01

python3 -c "import tensorflow as tf; print(tf.__version__)"

srun python -m videosaur.inference_unsupervised --config /scratch/work/jayawin1/article_4/Priority_RUs/configs/inference/waymo_tfds.yml #TODO: change