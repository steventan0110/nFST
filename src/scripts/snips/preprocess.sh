#!/bin/bash
#SBATCH -A jeisner1
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4GB
#SBATCH --output=/home/wtan12/nFST/output/slurm/snips_preprocess.out
#SBATCH --time=72:00:00


module load anaconda
conda activate nfst
AGNOSTIC=False
LATENT=False
AFTER=False
ROOT="/home/wtan12/nFST/src"
export PYTHONPATH="/home/wtan12/nFST"

python $ROOT/main_snips.py \
    output_dir="/scratch4/jeisner1/nfst" \
    output_mapping="/home/wtan12/nFST/src/preprocess/snips/mapping.snips" \
    tag_mapping="/home/wtan12/nFST/src/preprocess/snips/mapping.out" \
    fairseq_ckpt="/data/jeisner1/nfst/fairseq_ckpt/snips/agnostic-${AGNOSTIC}/latent-${LATENT}/after-${AFTER}" \
    agnostic=$AGNOSTIC \
    latent=$LATENT \
    after=$AFTER \
    gpu=0 \
    do_preprocess=True \
    do_train=False \
    do_fairseq=False \
    do_decode=False \
    do_eval=False