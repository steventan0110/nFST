import os
from pathlib import Path
import logging
import re
import multiprocessing as mp
from src.util.preprocess_util import Vocab, Utils
from src.preprocess.preprocess import Preprocess
from src.fsm.tr import TR
from tqdm import tqdm
import numpy as np
from numpy import load, savez_compressed
from os.path import exists
import tqdm

logger = logging.getLogger("PreprocessTR")


class PreprocessTR(Preprocess):
    def __init__(self, args) -> None:
        super().__init__(args)
        if not exists(os.path.dirname(self.cfg.vocab)):
            Path(os.path.dirname(self.cfg.vocab)).mkdir(parents=True, exist_ok=True)

        logger.info("Defining FST")
        self.limit = int(args.limit)
        self.sub_size = str(args.sub_size)
        self.task = args.task
        self.serialize_prefix = args.serialize_prefix
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

        # vocab should already be constructed from snips' fst
        assert os.path.exists(self.cfg.vocab)
        self.vocab = Vocab.load(self.cfg.vocab)
        logger.info("Start Preprocessing input into NPZ")
        # self.bos is vocab mapping of bos (which is access by self.cfg.bos)
        self.bos, self.eos, self.pad = Utils.lookup_control(
            self.cfg.bos, self.cfg.eos, self.cfg.pad
        )

        Path(self.serialize_prefix).mkdir(parents=True, exist_ok=True)
        base_machine = f"{self.serialize_prefix}/base.npz"
        if not os.path.exists(base_machine):
            logger.info("serialize the abstract mfst machine")
            self.serialize_z(base_machine, self.cfg.wfsa_length_prior)
        logger.info(f"Serialize input data with {self.cfg.cpu_count} cpus")

        with mp.Pool(self.cfg.cpu_count) as mpp:
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
            self.serialize_under_mpp(buckets, mpp)

    def serialize_under_mpp(self, buckets, mpp):
        for dataset_split in ("train", "dev", "test"):
            if self.task.startswith("tr-filter"):
                file_path = f"{self.prefix}/{dataset_split}/{self.language}_{dataset_split}_filter_10_{self.sub_size}.tsv"
            else:
                file_path = (
                    f"{self.prefix}/{dataset_split}/{self.language}_{dataset_split}.tsv"
                )
            with open(file_path, "r") as f:
                for idx, line in enumerate(tqdm.tqdm(f)):
                    if idx > self.limit:
                        break
                    other, en = line.rstrip().split("\t")
                    if self.task.startswith("tr-filter"):
                        # remove the prepended mark for language
                        other_encode = [
                            Vocab.lookup(self.output_mapping.inverse[_])
                            for _ in other.replace("<ur>", "").replace("<sd>", "")
                        ]
                        # add back preprended mark for language
                        if "<ur>" in other:
                            other_encode.insert(0, Vocab.lookup("<ur>"))
                        else:
                            other_encode.insert(0, Vocab.lookup("<sd>"))
                    else:
                        other_encode = [
                            Vocab.lookup(self.output_mapping.inverse[_]) for _ in other
                        ]
                    en_encoded = [Vocab.lookup(_) for _ in en.split(" ")]
                    # for debug purpose
                    # print([Vocab.r_lookup(_) for _ in other_encode])
                    # print([Vocab.r_lookup(_) for _ in en_encoded])
                    b_idx = idx % len(buckets)
                    buckets[b_idx][-1].append(
                        (
                            other_encode,
                            f"{self.serialize_prefix}/{dataset_split}.{idx}",
                            en_encoded,
                            dataset_split,
                        )
                    )

            if len(buckets) == 1:
                self.serialize(buckets[0])
            else:
                mpp.map(self.serialize, buckets)

    @staticmethod
    def serialize(params):
        bos, eos, pad, tau, vocab_path, work = params
        Vocab.load(vocab_path)
        Vocab.freeze()
        vocab_size = Vocab.size()
        for w in work:
            other, name, en, dataset_split = w
            print(f"serializing {name}.npz")

            try:
                if exists(f"{name}.npz"):
                    try:
                        loaded = load(f"{name}.npz")
                        del loaded
                        continue
                    except Exception as e:
                        print(f"re serializing {name}.npz")
                x = Utils.create_from_string(en, semiring_class=tau.semiring)
                y = Utils.create_from_string(other, semiring_class=tau.semiring)

                en_encoded = np.array(
                    [_ for _ in en]
                    + [
                        eos,
                    ],
                )
                other_encoded = np.array(
                    [_ for _ in other]
                    + [
                        eos,
                    ],
                )

                xT = x.compose(tau)
                assert xT.num_states > 0, f"{en}\t{en_encoded}"
                xTy = xT.compose(y)
                # xTy._string_mapper = tau._string_mapper[1]
                assert xTy.num_states > 0
                (matrices, _, _, _, _,) = Preprocess.composed_to_matrices(
                    bos,
                    xTy,
                    eos,
                    pad,
                    vocab_size,
                    fst_fname=f"{name}.{dataset_split}.clamped.fst",
                    serialize_mfst=True,
                )
                (free_matrices, _, _, _, _,) = Preprocess.composed_to_matrices(
                    bos,
                    xT,
                    eos,
                    pad,
                    vocab_size,
                    fst_fname=f"{name}.{dataset_split}.free.fst",
                    serialize_mfst=True,
                )

                savez_compressed(
                    f"{name}.npz",
                    num_emission=matrices[0],
                    num_transition=matrices[1],
                    denom_emission=free_matrices[0],
                    denom_transition=free_matrices[1],
                    gs=en_encoded,
                    ps=other_encoded,
                )
            except Exception as e:
                print(e)
                logger.error(
                    f"warning: {name}.npz cannot be serialized. en: {en}\tother: {other}"
                )

            if not exists(f"{name}.npz"):
                logger.warning("{} not exist after serialization".format(name))
