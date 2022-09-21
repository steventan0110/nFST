#!/usr/bin/env python
from src.util.preprocess_util import Vocab, Utils
from os.path import basename, exists
from scipy.special import logsumexp
import numpy as np
import logging
import re
from collections import defaultdict

logger = logging.getLogger("Rerank[Eval]")


def load_nfst_results(prefix, fname):
    bname = basename(fname)
    to_load = f"{prefix}/{bname}.decoded"
    assert exists(to_load), to_load
    try:
        with open(to_load, mode="r") as fh:
            loaded = float(fh.read().strip())
            return loaded
    except Exception as e:
        logger.info(to_load)
        raise e


def get_prob(m, prefix):
    def string_to_vocab(s):
        return [Vocab.lookup(_) for _ in s]

    fairseq_prob, fst_prob = [], []
    fairseq_prob_gold, fst_prob_gold = [], []
    count_gold = 0
    for (k, v) in m.items():
        if len(v) != 21 and len(v) != 20:
            logger.info("skip bad pair")
            continue
        # retrieve score for all pairs
        src_vocab = string_to_vocab(k)
        # compute normalized prob for fairseq
        f_score_raw = np.array([t[1] for t in v]).astype(float)
        f_score = f_score_raw - logsumexp(f_score_raw)
        fairseq_score = []
        fst_score = []
        gold_idx = None
        for (idx, (tgt, score, isRef)) in enumerate(v):
            tgt_vocab = string_to_vocab(tgt)
            hash_path = Utils.get_hashed_name(src_vocab, tgt_vocab, prefix)
            filename = f"{hash_path}.npz.decoded"
            if exists(filename):
                with open(filename, "r") as f:
                    prob = float(f.read().strip())
                    fst_score.append(prob)
                    fairseq_score.append(f_score[idx])
                    if isRef == 1:
                        gold_idx = len(fst_score) - 1
        fst_score_raw = np.array(fst_score).astype(float)
        fst_score_normalize = fst_score_raw - logsumexp(fst_score_raw)
        for i in range(len(fst_score_normalize)):
            if gold_idx is not None and i == gold_idx:
                fairseq_prob_gold.append(np.exp(fairseq_score[i]))
                fst_prob_gold.append(np.exp(fst_score_normalize[i]))
                if np.exp(fst_score_normalize[i]) >= 0.5:
                    count_gold += 1
            else:
                fairseq_prob.append(np.exp(fairseq_score[i]))
                fst_prob.append(np.exp(fst_score_normalize[i]))
    logger.info(
        "Out of {} gold pairs, fst ranks {} as 1st choice".format(
            len(fst_prob_gold), count_gold
        )
    )
    return fairseq_prob, fst_prob, fairseq_prob_gold, fst_prob_gold


def get_baseline(m, n):
    out = []
    lang_id_mistake = 0
    for (k, v) in m.items():
        # sort the translation based on score
        if len(v) < 20:
            logger.info("skip bad pair ", v)
            continue
        reorder_v = sorted(v, key=lambda x: x[1])
        order = [0 for _ in range(len(v))]
        for i in range(len(v)):
            if reorder_v[i][2] == 1:
                order[i] = 1
                gold_lang_id = reorder_v[i][0][0]
                if reorder_v[0][0][0] != gold_lang_id:
                    lang_id_mistake += 1
                break
        out.append(order)
    logger.info(f"Number of lang id mistake from fairseq: {lang_id_mistake}")
    return out


def ner_baseline(m):
    """return the highest prob sentence from fariseq output"""
    out = []

    def tag_to_vocab(s):
        return [Vocab.lookup(_) for _ in s]

    for (k, v) in m.items():
        if len(v) != 21 and len(v) != 20:
            logger.info("skip bad pair ", v)
            continue
        else:
            reorder_v = sorted(v, key=lambda x: x[1])
            gold_tgt = None
            for temp_item in v:
                if temp_item[2] == 1:  # found gold sent
                    gold_tgt = temp_item[0]
            assert gold_tgt is not None
            best_hyp = reorder_v[0][0]
            out.append((gold_tgt, best_hyp))
    return out


def ner_rerank(m, prefix):
    def string_to_vocab(s):
        return [Vocab.lookup(_) for _ in s]

    out = []
    for (k, v) in m.items():
        if len(v) != 21 and len(v) != 20:
            logger.info("skip bad pair")
            continue
        # retrieve score for all pairs
        src_vocab = string_to_vocab(k)
        gold_tgt = None
        max_prob = -float("inf")
        best_hyp = None
        for (idx, (tgt, _, isRef)) in enumerate(v):
            if isRef == 1:
                gold_tgt = tgt
                if len(v) == 21:
                    # hypothesis has no gold output, should not expose the oracle to fst as well
                    continue
            tgt_vocab = string_to_vocab(tgt)
            hash_path = Utils.get_hashed_name(src_vocab, tgt_vocab, prefix)
            filename = f"{hash_path}.npz.decoded"
            if exists(filename):
                with open(filename, "r") as f:
                    prob = float(f.read().strip())
                    if prob > max_prob:
                        max_prob = prob
                        best_hyp = tgt
            else:
                logger.info("hash file for {} does not exist".format(k))
        assert gold_tgt is not None
        if best_hyp is None:
            logger.info("no string other than gold is accepted")
            continue
        out.append((gold_tgt, best_hyp))
    return out


def get_prob_for_all_sentences(ref, out):
    # TODO: need to take into account translation direction of fairseq
    def convert_string_vocab(sent):
        temp = "".join(sent.rstrip().split(" "))
        temp = re.sub("<<unk>>", "", temp)
        temp = re.sub("<unk>", "", temp)
        return re.sub("<SP>", " ", temp)

    def convert_tag_vocab(sent):
        out = []
        for tag in sent.rstrip().split(" "):
            if tag.startswith("madeupword") or tag == "<unk>" or tag == "<<unk>>":
                continue
            out.append(tag)
        return out

    m = defaultdict(list)
    with open(ref, "r") as ref_h:
        data = ref_h.read()
        src, tgt, hypo = None, None, None

    for (idx, sent) in enumerate(data.split("\n")):
        if sent.startswith("S"):
            src = convert_string_vocab(sent.split("\t")[1])
        elif sent.startswith("T"):
            tgt = convert_tag_vocab(sent.split("\t")[1])
        elif sent.startswith("H"):
            score = sent.split("\t")[1]
            m[src].append((tgt, score, 1))  # 1 meaning this is the gold translation

    # now we start populating all test results
    with open(out, "r") as f:
        data = f.read()
    src, tgt, hypo = None, None, None
    for (idx, sent) in enumerate(data.split("\n")):
        if sent.startswith("S"):
            src = convert_string_vocab(sent.split("\t")[1])
        elif sent.startswith("T"):
            tgt = convert_tag_vocab(sent.split("\t")[1])
        elif sent.startswith("H"):
            score = sent.split("\t")[1]
            hyp = convert_tag_vocab(sent.split("\t")[2])
            if src not in m:
                logger.info(f"Preprocess err, {tgt} not in gold standard translation")
                continue
            if hyp == tgt:
                logger.info("hypothesis matches gold tgt, skip")
                continue
            m[src].append((hyp, score, 0))  # 0 meaning this is not the gold translation
    return m


def get_rerank(m, prefix, output_mapping, forward, filter=True):
    def string_to_vocab(s, use_mapping, filter):
        if use_mapping:
            if filter:
                out = []
                if s[0] == "<ur>" or s[0] == "<sd>":
                    out.append(Vocab.lookup(s[0]))
                else:
                    out.append(Vocab.lookup(output_mapping.inverse[s[0]]))
                for tok in s[1:]:
                    if tok in "<ur>" or tok in "<sd>":
                        continue
                    out.append(Vocab.lookup(output_mapping.inverse[tok]))
                return out
            return [Vocab.lookup(output_mapping.inverse[_]) for _ in s]
        else:
            return [Vocab.lookup(_) for _ in s]

    out = []
    num_exist_file = 0
    lang_id_mistake = 0
    oracle_log_prob = 0
    num_oracle_sent = 0
    for (k, v) in m.items():
        if len(v) < 20:
            logger.info("skip bad pair")
            continue
        # retrieve score for all pairs
        # if forward, src is other language and we need to use output mapping to convert it
        src_vocab = string_to_vocab(k, forward, filter)
        probs = []
        unaccepted_files_count = 0
        best_tgt = None
        best_prob = -float("inf")
        oracle_tgt = None
        for (idx, (tgt, _, isRef)) in enumerate(v):
            if isRef == 1:
                oracle_tgt = tgt
                oracle_lang_id = tgt[0]
            tgt_vocab = string_to_vocab(tgt, not forward, filter)
            hash_path = (
                Utils.get_hashed_name(tgt_vocab, src_vocab, prefix)
                if not forward
                else Utils.get_hashed_name(src_vocab, tgt_vocab, prefix)
            )
            # hash_path = basename(hash_path)
            # filename = f'{prefix}/{hash_path}'
            filename = f"{hash_path}.npz.decoded"
            if exists(filename):
                num_exist_file += 1
                with open(filename, "r") as f:
                    prob = float(f.read().strip())
                    probs.append(prob)
                    if prob > best_prob:
                        best_tgt = tgt
                    if isRef:
                        oracle_log_prob += prob
                        num_oracle_sent += 1
            else:
                logger.info("hash file for {} does not exist".format(k))
                probs.append(-float("inf"))
                unaccepted_files_count += 1
        if unaccepted_files_count == len(v):
            logger.info("Encounter the situation that even ref is not serialized")
            continue

        order = [0 for _ in range(len(v))]
        # assert max_idx != -1 # only possible if non of the file is loaded
        assert len(probs) == len(v)
        sorted_index = np.argsort(np.array(probs))
        for idx, pos in enumerate(reversed(sorted_index)):
            if idx == 0:
                if v[pos][0][0] != oracle_lang_id:
                    lang_id_mistake += 1
            if v[pos][2] == 1:
                order[idx] = 1
                # TODO: debug/interpretation purpose
                # if idx != 0:
                #     logger.info(
                #         "src: {}\n predicted tag: {}\n best tag: {}\n".format(
                #             k, best_tgt, oracle_tgt
                #         )
                #     )
                break
        out.append(order)
    logger.info(
        f"Gold sentence log prob sum: {oracle_log_prob}, logprob avg: {oracle_log_prob / num_oracle_sent}"
    )
    logger.info(f"{num_exist_file} files that are found")
    logger.info(f"{lang_id_mistake} lang id mistake by nfst")
    return out


class Rerank:
    def __init__(self, args) -> None:
        self.cfg = args.preprocess
        self.limit = int(args.limit)
        self.sub_size = args.sub_size
        self.task = args.task
        self.language = args.language
        self.control_symbols = list(self.cfg.control_symbols.split(","))
        self.output_mapping = Utils.load_mapping(args.output_mapping)
        self.input_mapping = args.input_mapping
        Vocab.load(args.vocab)
        # important paths used
        self.serialize_prefix = args.serialize_prefix
        self.prefix = args.prefix
        self.npz_path = args.serialize_fst_path
        self.decode_path = args.decode_prefix
        self.forward = False  # true if other -> eng
        self.k = 20  # FIXME: add this to config
        # add test split after model selection is done
        from src.decode.tr_serialize import find_best_ckpt

        for split in ["valid"]:
            (fairseq_ref, fairseq_hyp) = find_best_ckpt(args.fairseq_ckpt, split)
            logger.info(f"Using fairseq file: {fairseq_ref}")
            fairseq_result_map = get_prob_for_all_sentences(fairseq_ref, fairseq_hyp)
            self.compute_and_print_results(fairseq_result_map, split)

    def compute_and_print_results(self, result_map, split):
        baseline_outcome = get_baseline(result_map, self.k)
        filter = "filter" in self.task
        reranked_outcome = get_rerank(
            result_map,
            f"{self.decode_path}/{split}",
            self.output_mapping,
            self.forward,
            filter,
        )

        logger.info(f"MRR baseline: {mean_reciprocal_rank(baseline_outcome)}")
        logger.info(f"MRR reranked: {mean_reciprocal_rank(reranked_outcome)}")
        logger.info(f"TOP-1 baseline: {top1_rank(baseline_outcome)}")
        logger.info(f"TOP-1 reranked: {top1_rank(reranked_outcome)}")


def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
            rs: Iterator of relevance scores (list or numpy) in rank order
                    (first element is the first item)
    Returns:
            Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1.0 / (r[0] + 1) if r.size else 0.0 for r in rs])


def top1_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1.0 if r.size and r[0] == 0 else 0.0 for r in rs])
