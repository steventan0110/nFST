from typing import Dict
from src.util.dataset_reader import T9FSADataModule
from src.util.preprocess_util import Utils, Vocab, dotdict
from src.modules.lightning import JointProb

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import loggers as pl_loggers
import logging

logger = logging.getLogger("Trainer")


def flatten_args(args):
    """max depth=2 so no need to write recursive fnc"""
    ret = dotdict({})
    for k, v in args.items():
        if isinstance(v, Dict):
            for k_sub, v_sub in v.items():
                if k_sub in ret:
                    logger.warning(f"Duplicate args defined in hydra: {k_sub}")
                ret[k_sub] = v_sub
        else:
            if k in ret:
                logger.warning(f"Duplicate args defined in hydra: {k}")
            ret[k] = v
    return ret


class Trainer:
    def __init__(self, args) -> None:
        self.args = args
        Vocab.load(args.vocab)
        args.vocab_size = Vocab.size()
        # all special token needs to converted into vocab's space

        cfg = flatten_args(args)
        cfg.bos, cfg.eos, cfg.pad = Utils.lookup_control(cfg.bos, cfg.eos, cfg.pad)
        tb_logger = pl_loggers.TensorBoardLogger(cfg.training_outputs)

        data = T9FSADataModule(
            prefix=cfg.prefix,
            lang=cfg.task,
            limit=cfg.limit,
            serialize_prefix=cfg.serialize_prefix,
            batch_size=cfg.batch_size,
            pad=cfg.pad,
            num_workers=cfg.cpu_count,
            vocab_size=cfg.vocab_size,
            proposal_distribution=cfg.proposal_dist,
        )
        model_checkpoint = ModelCheckpoint(
            monitor="val_loss",
            filename=f"{cfg.tilde_p_choice}-{cfg.estimator}"
            + "-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            verbose=True,
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=True,
        )
        model = JointProb(cfg)
        accelerator = "cpu" if int(cfg.gpu) == 0 else "gpu"
        # FIXME: hacky way to handle cpu-only case
        if accelerator == "cpu":
            trainer = pl.Trainer.from_argparse_args(
                cfg,
                logger=tb_logger,
                callbacks=[model_checkpoint, early_stop],
                accelerator=accelerator,
            )
        else:
            trainer = pl.Trainer.from_argparse_args(
                cfg,
                logger=tb_logger,
                callbacks=[model_checkpoint, early_stop],
                accelerator=accelerator,
                devices=int(cfg.gpu),
            )

        trainer.fit(model, data)
