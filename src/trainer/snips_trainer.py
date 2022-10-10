import sys
import os

try:
    home_path = os.environ["HOME"]
except KeyError:
    home_path = "/home/wtan12"
sys.path.insert(1, os.path.join(home_path, "seq-samplers"))
import pytorch_lightning as pl
from dataset_readers.lightning_task1_reader import T9FSADataModule
from modules.lightning import JointProb
from pytorch_lightning import loggers as pl_loggers
from nre.preprocess_npz_t9 import PreprocessNPZT9
import torch

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.plugins import DDPPlugin


def main():
    from argparse import ArgumentParser
    from modules.util import Utils, Vocab

    parser = ArgumentParser()
    parser.add_argument("--training-outputs", default="/tmp/nre-logs/", type=str)
    parser.add_argument("--resume-from-checkpoint", default=None, type=str)
    parser.add_argument("--compose-checkpoint", default=False, action="store_true")
    parser.add_argument("--nro-model", default=None, type=str)
    parser.add_argument("--pretrain-model", default=None, type=str)
    parser.add_argument("--pretrain-model-type", default=None, type=str)
    parser = PreprocessNPZT9.add_argparse_args(parser)
    parser = T9FSADataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = JointProb.add_model_specific_args(parser)
    args = parser.parse_args()
    print(args)
    tb_logger = pl_loggers.TensorBoardLogger(args.training_outputs)

    from modules.own_t9 import OwnT9

    Vocab.load(args.vocab_mapping)
    args.vocab_size = Vocab.size()
    args.bos, args.eos, args.pad = Utils.lookup_control(args.bos, args.eos, args.pad)
    if int(args.precision) == 16:
        torch.set_default_dtype(torch.float16)

    model_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        filename=f"{args.tilde_p_choice}-{args.estimator}"
        + "-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        verbose=True,
    )

    data = T9FSADataModule.from_argparse_args(args)
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=20,
        verbose=True,
    )
    if args.compose_checkpoint:
        HARD_CODE_BTYPE = ["B-album", "B-track", "B-artist"]
        model = JointProb.load_from_checkpoint(args.nro_model, args=args, strict=False)
    else:
        model = JointProb(args)

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        callbacks=[model_checkpoint, early_stop],
        resume_from_checkpoint=args.resume_from_checkpoint,
        plugins=DDPPlugin(find_unused_parameters=True),
    )
    trainer.fit(model, data)
    return


if __name__ == "__main__":
    main()
