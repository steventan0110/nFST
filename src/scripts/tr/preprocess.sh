ROOT="/home/wtan12/easy-latent-seq/easy_latent_seq"
export PYTHONPATH="/home/wtan12/easy-latent-seq"
lang=ur
python $ROOT/main.py \
    output_dir="/export/c06/wtan12/nfst" \
    input_mapping="/home/kitsing/scratch/data/cmudict/fairseq-scripts/data-bin/cmudict/dict.cmudict.eng.txt" \
    output_mapping="/home/wtan12/seq-samplers/tr/nfst/${lang}.graphs.sym" \
    language=$lang \
    fairseq_ckpt=tx ${lang}" \
    sub_size=500 \
    gpu=0