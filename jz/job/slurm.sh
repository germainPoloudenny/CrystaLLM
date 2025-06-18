#!/bin/bash
#SBATCH --job-name=dp_n_atoms
#SBATCH --output=jz/logs/v.out
#SBATCH --error=jz/logs/v.err

#SBATCH --partition=gpu_p2
#SBATCH -C v100
#SBATCH -A nxk@v100

#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --hint=nomultithread

module purge
module load pytorch-gpu/py3/2.4.0
module load git
git checkout 810c5290b0ea11732b4ddb351d006d7b364649ea

export PYTHONPATH=$PYTHONPATH:/lustre/fswork/projects/rech/nxk/uvv78gt/

# Exécution du script Python avec torchrun
python train.py -log_dir dp_n_atoms
