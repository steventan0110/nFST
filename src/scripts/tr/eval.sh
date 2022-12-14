#!/bin/bash
#SBATCH -A jeisner1
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --output=/home/wtan12/nFST/output/slurm/tr_eval_128.out
#SBATCH --time=72:00:00


module load anaconda
conda activate nfst

ROOT="/home/wtan12/nFST/src"
export PYTHONPATH="/home/wtan12/nFST"
lang=ur
python -u $ROOT/main.py \
    output_dir="/scratch4/jeisner1/nfst" \
    input_mapping="/home/wtan12/nFST/src/preprocess/tr/dict.cmudict.eng.txt" \
    output_mapping="/home/wtan12/nFST/src/preprocess/tr/${lang}.graphs.sym" \
    language=$lang \
    fairseq_ckpt="/data/jeisner1/nfst/fairseq_ckpt/tr/checkpoints/${lang}" \
    gpu=0 \
    do_preprocess=False \
    do_train=False \
    do_fairseq=False \
    do_decode=False \
    do_eval=True