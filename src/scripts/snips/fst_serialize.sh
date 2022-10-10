AGNOSTIC=$1
LATENT=$2
AFTER=$3
ROOT="/home/wtan12/nFST/src"
export PYTHONPATH="/home/wtan12/nFST"

python $ROOT/main_snips.py \
    output_dir="/export/c06/wtan12/nfst" \
    output_mapping="/home/wtan12/seq-samplers/snips/nfst/mapping.snips" \
    tag_mapping="/home/wtan12/seq-samplers/snips/nfst/mapping.out" \
    fairseq_ckpt="/home/kitsing/scratch/data/snips/fairseq-scripts/checkpoints/snips/agnostic-${AGNOSTIC}/latent-${LATENT}/after-${AFTER}" \
    agnostic=$AGNOSTIC \
    latent=$LATENT \
    after=$AFTER \
    gpu=0 \
    do_preprocess=False \
    do_train=False \
    do_fairseq=True \
    do_decode=False \
    do_eval=False