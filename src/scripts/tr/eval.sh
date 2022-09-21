ROOT="/home/wtan12/nFST/src"
export PYTHONPATH="/home/wtan12/nFST"
lang=ur
python $ROOT/main.py \
    output_dir="/export/c06/wtan12/nfst" \
    input_mapping="/home/kitsing/scratch/data/cmudict/fairseq-scripts/data-bin/cmudict/dict.cmudict.eng.txt" \
    output_mapping="/home/wtan12/seq-samplers/tr/nfst/${lang}.graphs.sym" \
    language=$lang \
    fairseq_ckpt="/export/c01/kitsing/dakshina-small-fairseq/checkpoints/${lang}" \
    sub_size=500 \
    gpu=0 \
    do_preprocess=False \
    do_train=False \
    do_fairseq=False \
    do_decode=False \
    do_eval=True