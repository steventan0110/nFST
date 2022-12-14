from src.util.preprocess_util import Vocab, Utils
from os.path import basename, exists
from scipy.special import logsumexp
import numpy as np
import logging
import re
from collections import defaultdict
from seqeval.metrics import f1_score, classification_report

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
            hash_path = Utils.get_hashed_name(tgt_vocab, src_vocab, prefix)
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
    for (k, v) in m.items():
        # sort the translation based on score
        if len(v) < 20:
            print("skip bad pair ")
            continue
        reorder_v = sorted(v, key=lambda x: x[1])
        order = [0 for _ in range(len(v))]
        for i in range(len(v)):
            if reorder_v[i][2] == 1:
                order[i] = 1
                break
        out.append(order)
    return out


def ner_baseline(m):
    """return the highest prob sentence from fariseq output"""
    out = []

    def tag_to_vocab(s):
        return [Vocab.lookup(_) for _ in s]

    for (k, v) in m.items():
        if len(v) < 20:
            print("skip bad pair")
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


def ner_rerank(m, prefix, gold_set):
    from os.path import exists

    def string_to_vocab(s):
        return [Vocab.lookup(_) for _ in s]

    out = []
    for (k, v) in m.items():
        if len(v) < 20:
            print("skip bad pair")
            continue
        # retrieve score for all pairs
        src_vocab = string_to_vocab(k)
        gold_tgt = None
        max_prob = -float("inf")
        best_hyp = None
        for (idx, (tgt, _, isRef)) in enumerate(v):
            if isRef == 1:
                gold_tgt = tgt
                if k not in gold_set:
                    # hypothesis has no gold output, should not expose the oracle to fst as well
                    continue
            tgt_vocab = string_to_vocab(tgt)
            hash_path = Utils.get_hashed_name(tgt_vocab, src_vocab, prefix)
            filename = f"{hash_path}.npz.decoded"
            if exists(filename):
                with open(filename, "r") as f:
                    prob = float(f.read().strip())
                    if prob > max_prob:
                        max_prob = prob
                        best_hyp = tgt
            # else:
            #     print("hash file for {} does not exist".format(k))
        assert gold_tgt is not None
        if best_hyp is None:
            print("no string other than gold is accepted")
            continue
        out.append((gold_tgt, best_hyp))
    return out


def get_prob_for_all_sentences(ref, out, tag_mapping, slot=None):
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
            out.append(tag_mapping[int(tag)])
        return out

    m = defaultdict(list)
    gold_available_set = []
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
            if src == " ":
                print("skip bad pair")
                continue
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
                print("Preprocess err, {} not in gold standard translation".format(tgt))
                continue
            if src == " ":
                print("skip bad pair")
                continue
            if hyp == tgt:
                print("hypothesis matches gold tgt, skip")
                gold_available_set.append(src)
                continue
            m[src].append((hyp, score, 0))  # 0 meaning this is not the gold translation

    # add slotrefine data
    if slot is not None:
        with open(slot, "r") as fr:
            data = fr.read()
        src, tgt, hypo = None, None, None
        for (idx, sent) in enumerate(data.split("\n")):
            if sent.startswith("S"):
                src = convert_string_vocab(sent.split("\t")[1])
            elif sent.startswith("H"):
                score = sent.split("\t")[1]
                hyp = convert_tag_vocab(sent.split("\t")[2])
                if src not in m:
                    print(
                        "Preprocess err, {} not in gold standard translation".format(
                            tgt
                        )
                    )
                    continue
                if src == " ":
                    print("skip bad pair")
                    continue
                # check for duplicates
                is_duplicate = False
                for list_item in m[src]:
                    if list_item[0] == hyp:
                        # duplicate, do not add
                        is_duplicate = True
                        break
                if not is_duplicate:
                    m[src].append((hyp, score, 0))
    return m, gold_available_set


def get_rerank(m, prefix):
    def string_to_vocab(s):
        return [Vocab.lookup(_) for _ in s]

    out = []
    for (k, v) in m.items():
        if len(v) < 20:
            print("skip bad pair")
            continue
        # retrieve score for all pairs
        src_vocab = string_to_vocab(k)
        probs = []
        unaccepted_files_count = 0
        best_tgt = None
        best_prob = -float("inf")
        oracle_tgt = None
        for (idx, (tgt, _, isRef)) in enumerate(v):
            if isRef == 1:
                oracle_tgt = tgt
            tgt_vocab = string_to_vocab(tgt)
            hash_path = Utils.get_hashed_name(tgt_vocab, src_vocab, prefix)
            # hash_path = basename(hash_path)
            # filename = f'{prefix}/{hash_path}'
            filename = f"{hash_path}.npz.decoded"
            if exists(filename):
                with open(filename, "r") as f:
                    prob = float(f.read().strip())
                    probs.append(prob)
                    if prob > best_prob:
                        best_tgt = tgt
            else:
                # print("hash file for {} does not exist".format(k))
                probs.append(-float("inf"))
                unaccepted_files_count += 1
        if unaccepted_files_count == len(v):
            print("Encounter the situation that even ref is not serialized")
            continue

        order = [0 for _ in range(len(v))]
        # assert max_idx != -1 # only possible if non of the file is loaded
        assert len(probs) == len(v)
        sorted_index = np.argsort(np.array(probs))
        for idx, pos in enumerate(reversed(sorted_index)):
            if v[pos][2] == 1:
                order[idx] = 1
                if idx != 0:
                    print(
                        "src: {}\n predicted tag: {}\n best tag: {}\n".format(
                            k, best_tgt, oracle_tgt
                        )
                    )
                break
        out.append(order)
    return out


def update_I_tag(tags):
    # TODO: need to update for the special BIO tag
    prev_B_type = None
    out = []
    for tag in tags:
        if tag.startswith("B-"):
            prev_B_type = tag.split("-")[1]
        elif tag == "I":
            tag = "I-" + prev_B_type if prev_B_type is not None else "O"
        elif tag != "O":
            continue  # ignore the tag since we meet intent here
        out.append(tag)
    return out


def compute_oracle_rerank(m, gold_set):
    all_f1 = []
    for k, v in m.items():
        if len(v) < 20:
            # skip bad pair
            continue
        elif k in gold_set:
            # fairseq has the gold standard output as its hypothesis, the oracle should always find that as upperbound
            gold_tag = None
            for (tgt, _, isRef) in v:
                if isRef == 1:
                    gold_tag = tgt
                    break
            gold = update_I_tag(gold_tag)
            all_f1.append((gold, gold))
        else:
            max_f1 = -float("inf")
            pair = None
            # find the gold sample
            gold_tag = None
            for (tgt, _, isRef) in v:
                if isRef == 1:
                    gold_tag = tgt
                    break

            for (tgt, _, isRef) in v:
                if isRef == 1:
                    continue
                gold = update_I_tag(gold_tag)
                hyp = update_I_tag(tgt)
                if len(gold) > len(hyp):
                    gold = gold[: len(hyp)]
                elif len(gold) < len(hyp):
                    hyp = hyp[: len(gold)]
                score = f1_score([gold], [hyp])
                if score > max_f1:
                    max_f1 = score
                    pair = (gold, hyp)
            all_f1.append(pair)
    y_true = []
    y_pred = []
    for p1, p2 in all_f1:
        y_true.append(p1)
        y_pred.append(p2)
    return f1_score(y_true, y_pred)


class Rerank:
    def __init__(self, args) -> None:
        self.cfg = args.preprocess
        self.limit = int(args.limit)
        self.task = args.task
        self.serialize_prefix = args.serialize_prefix
        self.prefix = args.prefix
        self.control_symbols = list(self.cfg.control_symbols.split(","))
        self.output_mapping = Utils.load_mapping(args.output_mapping)
        self.tag_mapping = Utils.load_mapping(args.tag_mapping)
        self.agnostic = args.agnostic
        self.latent = args.latent
        self.after = args.after
        Vocab.load(args.vocab)
        # important paths used
        self.serialize_prefix = args.serialize_prefix
        self.prefix = args.prefix
        self.npz_path = args.serialize_fst_path
        self.decode_path = args.decode_prefix

        self.forward = False  # true if other -> eng
        self.k = 20
        from src.decode.snips_serialize import find_best_ckpt

        for split in ["valid"]:
            (fairseq_ref, fairseq_hyp) = find_best_ckpt(args.fairseq_ckpt, split)
            logger.info(f"Using fairseq file: {fairseq_ref}")
            fairseq_result_map, gold_available_set = get_prob_for_all_sentences(
                fairseq_ref, fairseq_hyp, self.tag_mapping
            )
            self.compute_results(fairseq_result_map, gold_available_set, split)

    def compute_results(self, result_map, gold_set, split):
        # compute oracle ranker f1 as the upperbound
        oracle_reranker_f1 = compute_oracle_rerank(result_map, gold_set)
        # generate baseline from fairseq
        baseline_outcome = get_baseline(result_map, self.k)
        baseline_outcome_no_oracle = ner_baseline(result_map)

        # retrieve result by FST and generate rerank array
        reranked_outcome = get_rerank(result_map, f"{self.decode_path}/{split}")
        reranked_outcome_no_oracle = ner_rerank(
            result_map, f"{self.decode_path}/{split}", gold_set
        )
        print(f"MRR baseline: {mean_reciprocal_rank(baseline_outcome)}")
        print(f"MRR rerank: {mean_reciprocal_rank(reranked_outcome)}")
        print()
        print(f"P@1 baseline: {top1_rank(baseline_outcome)}")
        print(f"P@1 rerank: {top1_rank(reranked_outcome)}")
        print()
        print(f"F1 baseline: {f1(baseline_outcome_no_oracle)}")
        print(f"F1 rerank: {f1(reranked_outcome_no_oracle)}")
        print(f"F1 Oracle (upperbound): {oracle_reranker_f1}")


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


def f1(input):
    """compute f1 score given an array of (gold, hyp) pair"""

    y_true = []
    y_pred = []
    for (gold, hyp) in input:
        # hacky way to make the import package compute correct f1 score
        gold = update_I_tag(gold)
        hyp = update_I_tag(hyp)
        if len(gold) > len(hyp):
            gold = gold[: len(hyp)]
        elif len(gold) < len(hyp):
            hyp = hyp[: len(gold)]
        y_true.append(gold)
        y_pred.append(hyp)
    score = f1_score(y_true, y_pred)
    # score = classification_report(y_true, y_pred)
    # print(score)
    return score
