#!/bin/bash

# Vérification des arguments
if [ $# -lt 2 ]; then
  echo "Usage: $0 <job_name> <gpu_type: v|a|h>"
  exit 1
fi

JOB_NAME="$1"
GPU_TYPE="$2"
SLURM_SCRIPT="jz/job/slurm.sh"

# Début de génération du fichier slurm.sh
echo "#!/bin/bash" > "$SLURM_SCRIPT"

# Partie 1 : Nom du job, fichiers de log
cat <<EOF >> "$SLURM_SCRIPT"
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=jz/logs/${GPU_TYPE}.out
#SBATCH --error=jz/logs/${GPU_TYPE}.err

EOF

# Partie 2 : Configuration GPU
if [[ "$GPU_TYPE" == "h" ]]; then
  cat <<EOF >> "$SLURM_SCRIPT"
#SBATCH --partition=gpu_p6
#SBATCH -C h100
#SBATCH -A nxk@h100

EOF
  ARCH="h100"
elif [[ "$GPU_TYPE" == "a" ]]; then
  cat <<EOF >> "$SLURM_SCRIPT"
#SBATCH --partition=gpu_p5
#SBATCH -C a100
#SBATCH -A nxk@a100

EOF
  ARCH="a100"
elif [[ "$GPU_TYPE" == "v" ]]; then
  cat <<EOF >> "$SLURM_SCRIPT"
#SBATCH --partition=gpu_p2
#SBATCH -C v100
#SBATCH -A nxk@v100

EOF
  ARCH="v100"
else
  echo "Unknown GPU type: $GPU_TYPE. Use v, a, or h."
  exit 1
fi

cat <<EOF >> "$SLURM_SCRIPT"
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --hint=nomultithread
EOF

# Partie 4 : Chargement des modules
COMMIT_HASH=$(git rev-parse HEAD)
cat <<EOF >> "$SLURM_SCRIPT"

module purge
EOF

# Charger le module arch uniquement si ARCH n'est pas vide
if [[ "$GPU_TYPE" != "v" ]]; then
  echo "module load arch/$ARCH" >> "$SLURM_SCRIPT"
fi

cat <<EOF >> "$SLURM_SCRIPT"
module load pytorch-gpu/py3/2.4.0
module load git
source crystallm_venv/bin/activate
git checkout $COMMIT_HASH

# Exécution du script Python avec torchrun
torchrun --nproc_per_node=1  bin/train.py dtype=float16 --config config/0.yaml
EOF

chmod +x "$SLURM_SCRIPT"
sbatch "$SLURM_SCRIPT"