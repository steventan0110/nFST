#!/bin/bash
#SBATCH -A jeisner1_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4GB
#SBATCH --output=/home/wtan12/nFST/output/slurm/tr_train_68pretrain.out
#SBATCH --time=48:00:00

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
    gpu=1 \
    do_preprocess=False \
    do_train=True \
    do_fairseq=False \
    do_decode=False \
    do_eval=False