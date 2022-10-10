#!/usr/bin/env python
from typing import *
import logging
from pathlib import Path
import os
import multiprocessing as mp
from src.util.preprocess_util import Vocab, Utils
from src.preprocess.preprocess import Preprocess
from src.preprocess.snips import PreprocessSnips
from src.fsm.snips import Snips
import re

logger = logging.getLogger(__name__)


def find_best_ckpt(path, split):
    def find_loss(target_folder):
        best_loss = float("inf")
        for _, _, files in os.walk(target_folder):
            for file in files:
                if "-valid.res" in file:
                    with open(f"{target_folder}/{file}", "r") as f:
                        wer_line = f.readline()
                        wer_score = float(wer_line.rstrip().split("\t")[-1])
                    if wer_score < best_loss:
                        best_loss = wer_score
            break
        return best_loss

    for _, ckpt_dirs, _ in os.walk(path):
        best_loss = float("inf")
        for ckpt in ckpt_dirs:
            loss = find_loss(f"{path}/{ckpt}")
            if loss < best_loss:
                best_loss = loss
                best_ckpt = f"{path}/{ckpt}"
        break
    return (f"{best_ckpt}/{split}.out.ref", f"{best_ckpt}/{split}.out.20")


class SerializeSnips(Preprocess):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.limit = int(args.limit)
        self.task = args.task
        self.serialize_prefix = args.serialize_prefix
        self.prefix = args.prefix
        self.control_symbols = list(self.cfg.control_symbols.split(","))
        self.output_mapping = Utils.load_mapping(args.output_mapping)
        self.tag_mapping = Utils.load_mapping(args.tag_mapping)
        self.serialize_fst_path = args.serialize_fst_path
        self.agnostic = args.agnostic
        self.latent = args.latent
        self.after = args.after
        # define fst machine tau
        self.tau = Snips(
            args.output_mapping,
            self.cfg.vocab,
            self.control_symbols,
            self.cfg.bos,
            self.cfg.eos,
            self.cfg.pad,
            self.agnostic,
            self.latent,
            self.after,
        ).get_machine()
        self.vocab = Vocab.load(self.cfg.vocab)
        self.bos, self.eos, self.pad = Utils.lookup_control(
            self.cfg.bos, self.cfg.eos, self.cfg.pad
        )
        logger.info(f"Serialize fairseq output with {self.cfg.cpu_count} cpus")

        for split in ["valid", "test"]:
            (ref_output, hyp_output) = find_best_ckpt(args.fairseq_ckpt, split)
            logger.info(f"Serialize fst from reference: {ref_output}")
            self.serialize_under_mpp(ref_output, split)
            logger.info(f"Serialize fst from hypothesis: {hyp_output}")
            self.serialize_under_mpp(hyp_output, split)

    def add_hyp_to_bucket(self, hypotheses, counter, buckets, src, path, split):
        if len(hypotheses) > 0:
            current_bucket = counter % len(buckets)
            for h in hypotheses:
                # eng -> other, src is en, h is other
                buckets[current_bucket][-1].append(
                    (h, Utils.get_hashed_name(h, src, path), src, split)
                )

    def serialize_under_mpp(self, fairseq_output, split):
        src = None
        hypotheses = []
        buckets = []
        for _ in range(self.cfg.cpu_count):
            buckets.append(
                (
                    self.bos,
                    self.eos,
                    self.pad,
                    self.tau,
                    self.cfg.vocab,
                    [],
                )
            )
        counter = 0
        path = f"{self.serialize_fst_path}/{split}"
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(fairseq_output, "r") as fh:
            for l in fh:
                if l.startswith("S-"):
                    counter += 1
                    self.add_hyp_to_bucket(
                        hypotheses, counter, buckets, src, path, split
                    )
                    src = self.process_line(l, "S")
                    hypotheses = []
                elif l.startswith("H-"):
                    hyp = self.process_line(l, "H")
                    hypotheses.append(hyp)
            if hypotheses:
                self.add_hyp_to_bucket(hypotheses, counter, buckets, src, path, split)

        with mp.Pool(self.cfg.cpu_count) as mpp:
            if len(buckets) == 1:
                PreprocessSnips.serialize(buckets[0])
            else:
                mpp.map(PreprocessSnips.serialize, buckets)

    def process_line(self, l, type) -> str:
        out = []
        if type == "S" or type == "T":
            sent = l.split("\t")[1]
            temp = "".join(sent.rstrip().split(" "))
            temp = re.sub("<<unk>>", "", temp)
            temp = re.sub("<unk>", "", temp)
            temp = re.sub("<SP>", " ", temp)
            for char in temp:
                out.append(Vocab.lookup(char))
        else:
            sent = l.split("\t")[2]
            tags = sent.rstrip().split(" ")
            for tag in tags:
                if tag.startswith("madeupword") or tag == "<unk>" or tag == "<<unk>>":
                    continue
                tag_decode = self.tag_mapping[int(tag)]
                out.append(Vocab.lookup(tag_decode))

        return out
