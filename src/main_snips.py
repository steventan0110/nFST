import hydra

from typing import Dict
import logging
from omegaconf import DictConfig, OmegaConf

# from src.preprocess.snips import PreprocessSnips
from src.util.preprocess_util import dotdict

# from src.evaluate.snips_rerank import Rerank

# from src.trainer.snips_trainer import Trainer
# from src.decode.snips_serialize import SerializeSnips

from src.decode.decoder import Decoder


logger = logging.getLogger("SNIPS[MAIN]")


def add_dot_operation(dict_cfg):
    for k, v in dict_cfg.items():
        if isinstance(v, Dict):
            dict_cfg[k] = dotdict(v)
    return dotdict(dict_cfg)


@hydra.main(config_path="./conf", config_name="snips.yaml")
def main(config: DictConfig):
    logger.info(OmegaConf.to_yaml(config, resolve=True))
    dict_cfg = OmegaConf.to_container(config, resolve=True)
    args = add_dot_operation(dict_cfg)
    # Trainer(args)
    # Rerank(args)
    # if args.do_preprocess:
    #     PreprocessSnips(args)

    # if args.do_train:
    #     Trainer(args)

    # if args.do_fairseq:
    #     SerializeSnips(args)

    if args.do_decode:
        Decoder(args)
    # if args.do_eval:
    #     Rerank(args)


if __name__ == "__main__":
    main()
