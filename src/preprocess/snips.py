from genericpath import exists
import os
from pathlib import Path
import logging
import re
from omegaconf import DictConfig
import multiprocessing as mp
from src.util.preprocess_util import Vocab, Utils
from src.preprocess.preprocess import Preprocess
from src.fsm.snips import Snips
from tqdm import tqdm
import numpy as np
from numpy import savez_compressed, load
from itertools import islice

logger = logging.getLogger("PreprocessSnips")


class PreprocessSnips(Preprocess):
    def __init__(self, args) -> None:
        super().__init__(args)
        if not exists(os.path.dirname(self.cfg.vocab)):
            Path(os.path.dirname(self.cfg.vocab)).mkdir(parents=True, exist_ok=True)

        logger.info("Defining FST")
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

        # vocab should already be constructed from snips' fst
        assert os.path.exists(self.cfg.vocab)
        Vocab.load(self.cfg.vocab)
        print(Vocab.size())

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
        for dataset_split in ("train", "valid", "test"):
            file_path = f"{self.prefix}/{dataset_split}"
            print("Serialize File Path: ", file_path)
            with open(f"{file_path}.snips.in") as en_fh, open(
                f"{file_path}.snips.out"
            ) as c_fh:
                for idx, (en_l, c_l) in enumerate(
                    tqdm(islice(zip(en_fh, c_fh), self.limit), disable=True)
                ):
                    if en_l == "\n" or c_l == "\n":
                        continue
                    tags = []
                    ciphers = c_l.strip()
                    for tag in ciphers.rstrip().split(" "):
                        # map from number back to string, then we can query in vocab
                        tags.append(self.tag_mapping[int(tag)])

                    eng = en_l.strip()
                    eng = "".join(eng.rstrip().split(" "))
                    eng = re.sub("<unk>", "", eng)
                    eng = re.sub("<SP>", " ", eng)

                    ci = [Vocab.lookup(_) for _ in tags]
                    en = [Vocab.lookup(_) for _ in eng]
                    b_idx = idx % len(buckets)
                    buckets[b_idx][-1].append(
                        (
                            ci,
                            f"{self.serialize_prefix}/{dataset_split}.{idx}",
                            en,
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
            tag, name, en, dataset_split = w
            # print(f"serializing {name}.npz")

            try:
                if exists(f"{name}.npz"):
                    try:
                        loaded = load(f"{name}.npz")
                        del loaded
                        continue
                    except Exception as e:
                        print(f"re serializing {name}.npz")
                x = Utils.create_from_string(en, semiring_class=tau.semiring)
                y = Utils.create_from_string(tag, semiring_class=tau.semiring)

                en_encoded = np.array(
                    [_ for _ in en]
                    + [
                        eos,
                    ],
                )
                other_encoded = np.array(
                    [_ for _ in tag]
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
                if False:  # verbose
                    decode_en = [Vocab.r_lookup(_) for _ in en]
                    decode_tag = [Vocab.r_lookup(_) for _ in tag]
                    logger.warning(
                        f"warning: {name}.npz cannot be serialized. en: {decode_en}\tother: {decode_tag}"
                    )

            if not exists(f"{name}.npz"):
                logger.warning("{} not exist after serialization".format(name))
