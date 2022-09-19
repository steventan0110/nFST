from genericpath import exists
import os
from pathlib import Path
import logging
import re
from omegaconf import DictConfig
import multiprocessing as mp
from easy_latent_seq.util.preprocess_util import Vocab, Utils
from easy_latent_seq.preprocess.preprocess import Preprocess
from easy_latent_seq.fsm.snips import Snips
from tqdm import tqdm
from numpy import savez_compressed
from itertools import islice

logger = logging.getLogger("PreprocessSnips")


class PreprocessSnips(Preprocess):
    def __init__(self, args) -> None:
        super().__init__(args)
        if not os.path.exists(os.path.dirname(self.cfg.vocab)):
            os.mkdir(os.path.dirname(self.cfg.vocab))

        logger.info("Defining FST")
        self.control_symbols = list(self.cfg.control_symbols.split(","))
        self.tau = Snips(
            args.tag_mapping,
            self.cfg.vocab,
            "",
            self.control_symbols,
            self.cfg.bos,
            self.cfg.eos,
            self.cfg.pad,
        ).agnostic_latent_after_fst
        # vocab should already be constructed from snips' fst
        assert os.path.exists(self.cfg.vocab)
        self.vocab = Vocab.load(self.cfg.vocab)
        logger.info("Start Preprocessing input into NPZ")
        self.tag_mapping = Utils.load_mapping(args.tag_mapping)
        self.output_mapping = Utils.load_mapping(args.phone_sym)
        self.bos, self.eos, self.pad = Utils.lookup_control(
            self.cfg.bos, self.cfg.eos, self.cfg.pad
        )
        self.serialize_prefix = args.serialize_prefix
        Path(self.serialize_prefix).mkdir(parents=True, exist_ok=True)
        base_machine = f"{self.serialize_prefix}/base.npz"
        if not os.path.exists(base_machine):
            logger.info("serialize the abstract mfst machine")
            self.serialize_z(base_machine, self.cfg.wfsa_length_prior)
        logger.info(f"Serialize input data with {self.cfg.cpu_count} cpus")

        # with mp.Pool(self.cfg.cpu_count) as mpp:
        #     buckets = []
        #     for _ in range(self.cfg.cpu_count):
        #         buckets.append(
        #             (
        #                 self.cfg.vocab,
        #                 [],
        #             )
        #         )
        #     self.serialize_under_mpp(buckets, args.limit, mpp, self.serialize_prefix)

    def serialize_under_mpp(self, buckets, limit, mpp, serialize_prefix):
        for dataset_split in ("train", "valid", "test"):
            file_path = f"{self.prefix}/{dataset_split}"
            print("Serialize File Path: ", file_path)
            with open(f"{file_path}.snips.in") as en_fh, open(
                f"{file_path}.snips.out"
            ) as c_fh:
                for idx, (en_l, c_l) in enumerate(
                    tqdm.tqdm(islice(zip(en_fh, c_fh), limit))
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
