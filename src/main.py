import hydra
from typing import Dict
import logging
from omegaconf import DictConfig, OmegaConf
from easy_latent_seq.preprocess.snips import PreprocessSnips
from easy_latent_seq.preprocess.tr import PreprocessTR
from easy_latent_seq.util.preprocess_util import dotdict
from easy_latent_seq.trainer.tr_trainer import Trainer
from easy_latent_seq.decode.tr_serialize import SerializeTR

logger = logging.getLogger(__name__)


def add_dot_operation(dict_cfg):
    for k, v in dict_cfg.items():
        if isinstance(v, Dict):
            dict_cfg[k] = dotdict(v)
    return dotdict(dict_cfg)


@hydra.main(config_path="./conf", config_name="tr.yaml")
def main(config: DictConfig):
    cfg = OmegaConf.to_yaml(config, resolve=True)
    logger.info(cfg)
    dict_cfg = OmegaConf.to_container(config, resolve=True)
    args = add_dot_operation(dict_cfg)
    PreprocessTR(args)
    # Trainer(args)
    SerializeTR(args)


if __name__ == "__main__":
    main()
