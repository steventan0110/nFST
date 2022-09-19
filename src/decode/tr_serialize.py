#!/usr/bin/env python
from typing import *
import logging
from pathlib import Path
import os
import re
import multiprocessing as mp
from easy_latent_seq.util.preprocess_util import Vocab, Utils
from easy_latent_seq.preprocess.preprocess import Preprocess
from easy_latent_seq.preprocess.tr import PreprocessTR
from easy_latent_seq.fsm.tr import TR
from tqdm import tqdm
from os.path import exists
import re

logger = logging.getLogger("SerializeTR")


def find_loss(path):
    best_loss = float("inf")
    for _, _, files in os.walk(path):
        for file in files:
            if "best_loss" in file:
                loss = file.split("best_loss_")[1]
                loss = float(loss[:-3])
                if loss < best_loss:
                    best_loss = loss
        break
    return best_loss


def find_best_ckpt(path, split):
    for _, ckpt_dirs, _ in os.walk(path):
        best_loss = float("inf")
        for ckpt in ckpt_dirs:
            loss = find_loss(f"{path}/{ckpt}")
            if loss < best_loss:
                best_loss = loss
                best_ckpt = f"{path}/{ckpt}"
        break
    return (f"{best_ckpt}/{split}.out.ref", f"{best_ckpt}/{split}.out.20")


class SerializeTR(Preprocess):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.limit = int(args.limit)
        self.task = args.task
        self.serialize_prefix = args.serialize_prefix
        self.serialize_fst_path = args.serialize_fst_path
        self.prefix = args.prefix
        self.language = args.language
        self.control_symbols = list(self.cfg.control_symbols.split(","))
        self.output_mapping = Utils.load_mapping(args.output_mapping)
        self.input_mapping = args.input_mapping
        # define fst machine tau
        self.tau = TR(
            args.output_mapping,
            args.input_mapping,
            self.cfg.vocab,
            self.control_symbols,
            self.cfg.bos,
            self.cfg.eos,
            self.cfg.pad,
            self.language,
        ).get_machine(args.machine_type)
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
                    self.add_hyp_to_bucket(
                        hypotheses, counter, buckets, src, path, split
                    )

        with mp.Pool(self.cfg.cpu_count) as mpp:
            if len(buckets) == 1:
                PreprocessTR.serialize(buckets[0])
            else:
                mpp.map(PreprocessTR.serialize, buckets)

    def process_line(self, l, type) -> list:
        def rm_unk(str):
            temp = "".join(str.rstrip().split(" "))
            temp = re.sub("<<unk>>", "", temp)
            temp = re.sub("<unk>", "", temp)
            temp = re.sub("<SP>", " ", temp)
            return temp

        out = []
        if self.task.startswith("tr-filter"):
            if type == "S":
                sent = l.split("\t")[1]
                for char in rm_unk(sent):
                    out.append(Vocab.lookup(char))
            elif type == "T":
                pass  # not used in the codebase
            else:  # hypothesis in other language, need to keep label <ur>/<sd>
                sent = l.split("\t")[2]
                tokens = sent.rstrip().split(" ")
                tags = tokens[1:]
                output_tag = []
                for tag in tags:
                    if (
                        tag.startswith("madeupword")
                        or tag == "<unk>"
                        or tag == "<<unk>>"
                        or tag == "<sd>"
                        or tag == "<ur>"
                    ):
                        continue
                    output_tag.append(self.output_mapping.inverse[tag])
                if tokens[0] == "<ur>" or tokens[0] == "<sd>":
                    out.append(Vocab.lookup(tokens[0]))
                else:
                    out.append(Vocab.lookup(self.output_mapping.inverse[tokens[0]]))
                for item in output_tag:
                    out.append(Vocab.lookup(item))
        else:
            if type == "S" or type == "T":
                sent = l.split("\t")[1]
                for char in rm_unk(sent):
                    out.append(Vocab.lookup(char))
            else:
                sent = l.split("\t")[2]
                tags = sent.rstrip().split(" ")
                output_tag = []
                for tag in tags:
                    if (
                        tag.startswith("madeupword")
                        or tag == "<unk>"
                        or tag == "<<unk>>"
                    ):
                        continue
                    output_tag.append(self.output_mapping.inverse[tag])
                for item in output_tag:
                    out.append(Vocab.lookup(item))
        return out
