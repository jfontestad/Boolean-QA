#!/bin/bash

#SBATCH -A danielk_gpu
#SBATCH --partition a100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name="HW6 CS 601.471/671 homework"

module load anaconda

#init virtual environment if needed
#conda create -n toy_classification_env python=3.7

conda activate toy_classification_env # open the Python environment

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 faiss-gpu=1.7.3 cudatoolkit=11.3 -c pytorch -c nvidia
pip install -r requirements.txt # install Python dependencies
#conda install -c conda-forge faiss-gpu

#runs your code
#srun python classification.py  --experiment "overfit" --small_subset False --device cuda --model "bert-base-uncased" --batch_size "32" --lr 1e-4 --num_epochs 9
srun python rag_classification.py  --experiment "rag" --small_subset True --device cuda --model "facebook/rag-token-nq" --batch_size "1" --lr 1e-4 --num_epochs 3
#srun python bloomz.py  --experiment "bloomz" --model "bigscience/bloomz-560m"
#srun python petals_bloomz.py --experiment "petals" --model "bigscience/bloomz-560m"
