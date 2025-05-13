#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=2:00:00
#SBATCH --output=outfile_snellius.out

module load 2022
module load Python/3.10.4-GCCcore-11.3.0
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
#module load 2023
#module load Python/3.11.3-GCCcore-12.3.0
#module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

#Create output directory on scratch
mkdir -p "$TMPDIR"/data

#Copy data to scratch
scp -r $HOME/globalise/operation_autumn/data "$TMPDIR"/data

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.


# Define the array
inv_nrs=('1160' '1066' '7673' '11012' '9001' '1348' '4071' '1090' '1430' '2665' '1439' '1595' '2693' '3476' '8596') #, '3598')
modelnames=('globalise/GloBERTise' 'bert-base-multilingual-cased' 'emanjavacas/GysBERT-v2' 'FacebookAI/xlm-roberta-base' 'pdelobelle/robbert-v2-dutch-base' 'emanjavacas/GysBERT' 'GroNLP/bert-base-dutch-cased' 'emanjavacas/MacBERTh' 'FacebookAI/roberta-base' 'google-bert/bert-base-cased')
seeds=(23052024 21102024 553311 6834 888)


# Loop over seeds
for seed in "${seeds[@]}"
do
    # Loop over the models
    for modelname in "${modelnames[@]}"
    do
        # Loop over inventory numbers, e.g., datasplits
        for inv_nr in "${inv_nrs[@]}"
        do
          # Run the Python script with the current values (seed+model+datasplit combination)
          python finetune_with_click.py \
              --seed=$seed \
              --inv_nr="$inv_nr" \
              --root_path="data/json_per_doc/" \
              --tokenizername="$modelname" \
              --modelname="$modelname" \
              --label_list="['O', 'I-event']"
        done
    done
done

