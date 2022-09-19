USERNAME="wtan12"
LANG="tr-filter"
LANGUAGE=ur
SPLIT="valid"
PANPHON=0
SYLLABIC=0
UNICODE=0
O_I_WEIGHTS=0
NO_SUBSTITUTION=1
RESERVED_SYLLABIC=0
RESERVED_VOCAB_SIZE=150
IS_SBATCH=0
NUM_CANDIDATES=20
SEED="42"
USE_T9=1
HARD=0
FAT_FINGER=0
DECOUPLED_WFST=0
SPECIFY_VOCAB=/home/${USERNAME}/seq-samplers/tr/nfst/ursd-g2p.vocab
SCRIPT_DIR=/home/${USERNAME}/seq-samplers/tr/scripts/clsp
HYPERPARAMS=$SCRIPT_DIR/hyperparams-filter.sh

# specific job
PREPROCESS=false
TRAIN=fals 
SERIALIZE_NPZ=false
DECODE=false
EVALUATE=false

conda activate easy-latent-seq

if ${PREPROCESS}; then
  echo "preprocessing"
  bash $SCRIPT_DIR/preprocess.sh ${LANG} ${UNICODE} ${O_I_WEIGHTS} ${NO_SUBSTITUTION} ${RESERVED_SYLLABIC} ${RESERVED_VOCAB_SIZE} \
    ${USE_T9} ${HARD} ${FAT_FINGER} ${DECOUPLED_WFST} ${SPECIFY_VOCAB} ${IS_SBATCH} ${HYPERPARAMS} ${LANGUAGE} "compose"
fi

if ${TRAIN}; then
  echo "training"
  bash $SCRIPT_DIR/train_script.sh ${LANG} ${UNICODE} ${O_I_WEIGHTS} ${NO_SUBSTITUTION} ${RESERVED_SYLLABIC} ${RESERVED_VOCAB_SIZE} \
  ${USE_T9} ${HARD} ${FAT_FINGER} ${DECOUPLED_WFST} ${SPECIFY_VOCAB} ${IS_SBATCH} ${HYPERPARAMS} ${LANGUAGE}
fi

echo "finding best parameters"
source $SCRIPT_DIR/common.sh ${LANG} ${UNICODE} ${O_I_WEIGHTS} ${NO_SUBSTITUTION} ${RESERVED_SYLLABIC} ${RESERVED_VOCAB_SIZE} \
  ${USE_T9} ${HARD} ${FAT_FINGER} ${DECOUPLED_WFST} ${SPECIFY_VOCAB} ${IS_SBATCH} ${HYPERPARAMS} ${LANGUAGE}
if [ ${USERNAME} == 'wtan12' ]; then
  CHECKPOINT_PATH="/export/b08/wtan/${LANG}/${LANGUAGE}/outputs/${LANG}/training/${TRAINING_CONFIG}/default"
else
  echo "set checkpoint path"
fi
BEST_PARAMS_CMD=(python ${HOME}/seq-samplers/t9_scripts/find_best_parameters.py "${CHECKPOINT_PATH}" )

echo "Checkpoint best path: " $CHECKPOINT_PATH
BEST_PARAMS="$(${BEST_PARAMS_CMD[*]})"
echo "${BEST_PARAMS}"

echo "finding best seq2seq hyperparams"
FIND_CMD=(python ${HOME}/seq-samplers/nre_scripts/find_lowest_wer.py /export/c01/kitsing/dakshina-small-fairseq/checkpoints/${LANGUAGE} --find-best --t9 --use-suffix)
FAIRSEQ_OUTPUT_RERANK="$(${FIND_CMD[*]})/${SPLIT}.out.${NUM_CANDIDATES}"
FAIRSEQ_OUTPUT_REF="$(${FIND_CMD[*]})/${SPLIT}.out.ref"

FIND_WFST_CMD=(python /export/c01/kitsing/2020/task1/baselines/fst/gen_best_hyps_filtered.py --checkpoint-prefix /export/c01/kitsing/2020/task1/baselines/fst/checkpoints-dakshina-small/${LANGUAGE} --language ${LANGUAGE} --check-start 6 --check-end 10 --template "/export/c01/kitsing/2020/task1/baselines/fst/checkpoints-dakshina-small/hyps/${LANGUAGE}-{best_run}.hyps.txt.invert")
WFST_OUTPUT="$(${FIND_WFST_CMD[*]})"
echo "wFST Hypothesis file" $WFST_OUTPUT

echo ${FAIRSEQ_OUTPUT_REF}
if ${SERIALIZE_NPZ}; then
  echo "getting wfst hyp fsts"
  #bash $SCRIPT_DIR/populate_wfst_fsts.sh ${LANG} ${UNICODE} ${O_I_WEIGHTS} ${NO_SUBSTITUTION} ${RESERVED_SYLLABIC} \
  #  ${RESERVED_VOCAB_SIZE} ${USE_T9} ${HARD} ${FAT_FINGER} ${DECOUPLED_WFST} ${SPECIFY_VOCAB} ${IS_SBATCH} ${HYPERPARAMS} ${LANGUAGE} \
  #  ${SPLIT} ${FAIRSEQ_OUTPUT_RERANK} "compose" ${WFST_OUTPUT}

  echo "getting seq2seq baseline fsts - hypotheses"
  bash $SCRIPT_DIR/populate_baseline_fsts.sh ${LANG} ${UNICODE} ${O_I_WEIGHTS} ${NO_SUBSTITUTION} ${RESERVED_SYLLABIC} \
    ${RESERVED_VOCAB_SIZE} ${USE_T9} ${HARD} ${FAT_FINGER} ${DECOUPLED_WFST} ${SPECIFY_VOCAB} ${IS_SBATCH} ${HYPERPARAMS} ${LANGUAGE} \
    ${SPLIT} ${FAIRSEQ_OUTPUT_RERANK} "compose"

  echo "getting seq2seq baseline fsts - gold"
  bash $SCRIPT_DIR/populate_baseline_fsts.sh ${LANG} ${UNICODE} ${O_I_WEIGHTS} ${NO_SUBSTITUTION} ${RESERVED_SYLLABIC} \
    ${RESERVED_VOCAB_SIZE} ${USE_T9} ${HARD} ${FAT_FINGER} ${DECOUPLED_WFST} ${SPECIFY_VOCAB} ${IS_SBATCH} ${HYPERPARAMS}  ${LANGUAGE}\
    ${SPLIT} ${FAIRSEQ_OUTPUT_REF} "compose"
fi

if ${DECODE}; then
  echo "getting marginalized probs for all transduction candidates"
  bash $SCRIPT_DIR/decode.sh ${LANG} ${UNICODE} ${O_I_WEIGHTS} ${NO_SUBSTITUTION} ${RESERVED_SYLLABIC} ${RESERVED_VOCAB_SIZE} \
    ${USE_T9} ${HARD} ${FAT_FINGER} ${DECOUPLED_WFST} ${SPECIFY_VOCAB} ${IS_SBATCH} ${HYPERPARAMS} ${LANGUAGE} \
    ${BEST_PARAMS} 0
fi

if ${EVALUATE}; then
  echo "evaluating dev"
  if [ ${USERNAME} == 'wtan12' ]; then
	  nfst_path=/export/b08/wtan/${LANG}/${LANGUAGE}/outputs/${LANG}/training/${TRAINING_CONFIG}/decoded
  else
	  echo "define nfst path"
  fi

  python /home/${USERNAME}/seq-samplers/tr/nfst/rerank_fairseq_output.py \
    --phones-sym ${HOME}/seq-samplers/tr/nfst/${LANGUAGE}.graphs.sym \
    --serialize-prefix ${SERIALIZE_PREFIX} --filter \
    --ref ${FAIRSEQ_OUTPUT_REF} --out ${FAIRSEQ_OUTPUT_RERANK} \
    --nfst-results-path ${nfst_path} --language ${LANGUAGE} \
    --ignore-unk --vocab ${SPECIFY_VOCAB} --bos 19997 --eos 19998 --pad 19999
fi



