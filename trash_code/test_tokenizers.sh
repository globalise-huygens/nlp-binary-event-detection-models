#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --output=tokenizertest.out

module load 2022
module load Python/3.10.4-GCCcore-11.3.0
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0

#module load 2023
#module load Python/3.11.3-GCCcore-12.3.0
#module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

#Create output directory on scratch
#mkdir -p "$TMPDIR"/data

#Copy data to scratch
#scp -r $HOME/globalise/operation_autumn/data "$TMPDIR"/data

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.


# Define the array
python test_tokenizers.py