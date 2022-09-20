import hydra
from typing import Dict
import logging
from omegaconf import DictConfig, OmegaConf
from src.preprocess.snips import PreprocessSnips
from src.preprocess.tr import PreprocessTR
from src.util.preprocess_util import dotdict
from src.trainer.tr_trainer import Trainer
from src.decode.tr_serialize import SerializeTR
from src.decode.decoder import Decoder
from src.evaluate.rerank import Rerank

logger = logging.getLogger(__name__)


def add_dot_operation(dict_cfg):
    for k, v in dict_cfg.items():
        if isinstance(v, Dict):
            dict_cfg[k] = dotdict(v)
    return dotdict(dict_cfg)


@hydra.main(config_path="./conf", config_name="tr.yaml")
def main(config: DictConfig):
    logger.info(OmegaConf.to_yaml(config, resolve=True))
    dict_cfg = OmegaConf.to_container(config, resolve=True)
    args = add_dot_operation(dict_cfg)

    if args.do_preprocess:
        PreprocessTR(args)
    if args.do_train:
        Trainer(args)
    if args.do_fairseq:
        SerializeTR(args)
    if args.do_decode:
        Decoder(args)
    # if args.do_eval:
    #     Rerank(args)


if __name__ == "__main__":
    main()
