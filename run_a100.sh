#!/bin/bash
#SBATCH -J Dreamer                    # Job name
#SBATCH --ntasks=1
#SBATCH --time=8:0:0
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100:1
#SBATCH --account=gts-agarg35
#SBATCH -q embers                               # QOS Name
#SBATCH --output=slurm_out/Report-%j.out

# Setup
mkdir -p slurm_out

module load anaconda3/2022.05.0.1               # Load module dependencies
conda activate dreamer

echo "Running the following command:"
echo $@

srun $@