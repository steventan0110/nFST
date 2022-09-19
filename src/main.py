import hydra
from typing import Dict
import logging
from omegaconf import DictConfig, OmegaConf
from src.preprocess.snips import PreprocessSnips
from src.preprocess.tr import PreprocessTR
from src.util.preprocess_util import dotdict
from src.trainer.tr_trainer import Trainer
from src.decode.tr_serialize import SerializeTR

logger = logging.getLogger(__name__)


def add_dot_operation(dict_cfg):
    for k, v in dict_cfg.items():
        if isinstance(v, Dict):
            dict_cfg[k] = dotdict(v)
    return dotdict(dict_cfg)


@hydra.main(config_path="./conf", config_name="debug.yaml")
def main(config: DictConfig):
    cfg = OmegaConf.to_yaml(config, resolve=True)
    logger.info(cfg)
    dict_cfg = OmegaConf.to_container(config, resolve=True)
    args = add_dot_operation(dict_cfg)
    PreprocessTR(args)
    # Trainer(args)
    # SerializeTR(args)


if __name__ == "__main__":
    main()
