#!/bin/bash

#SBATCH --partition=spgpu2
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-cpu=11GB
#SBATCH --account=
#SBATCH --time=00-08:00:00 
#SBATCH --output=/path_to_logs/%u/%j-%x.log

echo $1 $2
echo "Job started at" `date`
python src/run_inference.py --model_name_or_path "$1" --input_path "$2"
echo "Job Completed at" `date`