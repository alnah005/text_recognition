#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --mail-type=ALL
#SBATCH --mem=128GB
#SBATCH --job-name=label_generation
#SBATCH --mail-user=alnah005@umn.edu
#SBATCH -p amdsmall                                            

cd $SLURM_SUBMIT_DIR
module load singularity
pwd
singularity exec --nv -i ../raccoon_identification/Triplet_Loss_Framework/nv_od_local_v1.sif ~/text_recognition/commands.sh
