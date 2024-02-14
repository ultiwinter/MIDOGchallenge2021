#!/bin/bash -l
#SBATCH --job-name=my_dpdl_training  # ADAPT
#SBATCH --clusters=tinygpu
#SBATCH --ntasks=1

#SBATCH --gres=gpu:a100:2
#SBATCH --partition=a100

#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --mail-type=end,fail
#SBATCH --time=24:00:00  # ADAPT TO YOUR NEEDS
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

source ~/.bashrc

# Set proxy to access internet from the node
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

module purge
module load python
module load cuda
module load cudnn

# Conda
source activate seminar_dpdlv2

# create a temporary job dir on $TMPDIR
echo $TMPDIR
WORKDIR="$TMPDIR/$SLURM_JOBID"
mkdir $WORKDIR
cd $WORKDIR

# cp /home/janus/iwb6-datasets/MIDOG2021.tar.gz .  # NOTE: FOR LONG TRAININGS: COPY THE DATA ON THE NODE AND ADAPT THE PATH IN DPDL_DEFAULTS
# tar xzf MIDOG2021.tar.gz 

cd $WORKDIR

# copy input file from location where job was submitted, and run
cp -r ${SLURM_SUBMIT_DIR}/. .
mkdir -p output/

# Run training script
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda);"
# command for training
python training.py -m train --nepochs 3000 -p 512 -b 30 -lr 0.0001 --logdir output/

# command for testing
# python training.py -m test --checkptfile '<replace with path to checkpt.ckpt>'

# Create a directory on $HOME and copy the results from our training
# note: you can adapt this to copy the data also to work to avoid filling up your home
mkdir ${HOME}/$SLURM_JOB_ID
cp -r ./output/. ${HOME}/$SLURM_JOB_ID


