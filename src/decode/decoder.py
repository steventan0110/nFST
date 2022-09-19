import sys
import os
import logging
from src.modules.lightning import JointProb
from src.preprocess.preprocess import Preprocess
from src.util.preprocess_util import Utils, Vocab
from src.trainer.tr_trainer import flatten_args
from glob import glob
from tqdm import tqdm
from os.path import basename, exists


logger = logging.getLogger("Decoder")


def find_loss(path):
    if not exists(path):
        return (None, None)
    best_loss = float("inf")
    for _, _, files in os.walk(path):
        for file in files:
            if "val_loss" in file:
                loss = file.split("val_loss=")[1]
                loss = float(loss[:-5])
                if loss < best_loss:
                    best_loss = loss
                    best_ckpt = file
        break
    return (best_loss, best_ckpt)


def find_best_ckpt(path):
    for _, ckpt_dirs, _ in os.walk(path):
        best_loss = float("inf")
        for ckpt in ckpt_dirs:  # different version
            (loss, best_file) = find_loss(f"{path}/{ckpt}/checkpoints")
            print(loss, ckpt)
            if loss is None:
                # sometimes a folder is empty due to bad training
                continue
            if loss < best_loss:
                best_loss = loss
                best_ckpt = f"{path}/{ckpt}/checkpoints/{best_file}"
        break
    return best_ckpt


class Decoder:
    def __init__(self, args) -> None:
        self.args = flatten_args(args)
        Vocab.load(args.vocab)
        self.vocab_size = Vocab.size()
        self.args.bos, self.args.eos, self.args.pad = Utils.lookup_control(
            args.preprocess.bos, args.preprocess.eos, args.preprocess.pad
        )
        self.model_root = f"{args.training_outputs}/default"
        self.best_ckpt = find_best_ckpt(self.model_root)
        self.decode_prefix = args.decode_prefix

        logger.info(f"Decoding fst with model ckpt {self.best_ckpt}")
        for split in ["valid", "test"]:
            npz_path = f"{args.serialize_fst_path}/{split}"
            self.decode(npz_path, split)

    def decode(self, path, split):
        model = JointProb.load_from_checkpoint(
            self.best_ckpt, args=self.args, strict=True
        ).to("cpu")
        model.eval()

        filenames = glob(f"{path}/*.npz")
        for fname in tqdm(filenames, disable=None):
            bfname = basename(fname)
            decoded_fname = f"{self.decoded_prefix}/{split}/{bfname}.decoded"
            if not exists(decoded_fname):
                try:
                    prob, mark = model.decode_from_npz(
                        fname, vocab_size=self.vocab_size, pad=self.args.pad
                    )
                    with open(decoded_fname, mode="w") as fh:
                        fh.write(f"{prob}\n")
                    # tokens = [Vocab.r_lookup(_) for _ in mark.tolist()]
                    # logger.info(tokens)
                except Exception as e:
                    print(f"cannot decode {fname}: {e}")
            else:
                logger.info("sequence already decoded")
            break