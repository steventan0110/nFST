import os
import logging
from pathlib import Path
from src.modules.lightning import JointProb
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
        Vocab.load(args.vocab)
        args.vocab_size = Vocab.size()
        self.args = flatten_args(args)
        self.args.bos, self.args.eos, self.args.pad = Utils.lookup_control(
            args.preprocess.bos, args.preprocess.eos, args.preprocess.pad
        )
        self.model_root = f"{args.training_outputs}/lightning_logs"
        self.best_ckpt = find_best_ckpt(self.model_root)
        self.decode_prefix = args.decode_prefix

        logger.info(f"Decoding fst with model ckpt {self.best_ckpt}")
        self.model = JointProb.load_from_checkpoint(
            self.best_ckpt, "cpu", args=self.args
        )
        self.model.eval()
        for split in ["valid", "test"]:
            npz_path = f"{args.serialize_fst_path}/{split}"
            self.decode(npz_path, split)

    def decode(self, path, split):
        Path(f"{self.decode_prefix}/{split}").mkdir(parents=True, exist_ok=True)
        filenames = glob(f"{path}/*.npz")
        visualize = True
        for fname in tqdm(filenames, disable=True):
            bfname = basename(fname)
            decoded_fname = f"{self.decode_prefix}/{split}/{bfname}.decoded"

            if (not exists(decoded_fname)) or visualize:
                try:
                    prob, mark = self.model.decode_from_npz(
                        fname, vocab_size=self.args.vocab_size, pad=self.args.pad
                    )
                    if visualize:
                        tokens = [Vocab.r_lookup(_) for _ in mark.tolist()]
                        logger.info(tokens)
                    else:
                        with open(decoded_fname, mode="w") as fh:
                            fh.write(f"{prob}\n")
                except Exception as e:
                    logger.error(f"cannot decode {fname}: {e}")
            else:
                logger.info("sequence already decoded")
