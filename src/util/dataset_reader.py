from argparse import ArgumentParser, Namespace
from os.path import exists
from typing import *
from xml.sax import default_parser_list

from numpy import load
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch import from_numpy

from src.util.preprocess_util import Utils
import tqdm
from itertools import islice


class FSADataset(Dataset):
    def __init__(
        self,
        list_of_machines: List[str],
        list_of_wfst_proposals: Optional[List[str]] = None,
        vocab_size=None,
        pad=None,
    ):
        self.l = list_of_machines
        self.proposals = list_of_wfst_proposals
        self.vocab_size = vocab_size
        self.pad = pad
        self.z = None

    def __getitem__(self, index: int):
        name = self.l[index]
        npz_fname = f"{name}.npz"
        if self.proposals is not None:
            wfst_name = self.proposals[index]
        else:
            wfst_name = None
        to_return = Utils.load_fsa_from_npz(
            npz_fname, wfst_name, self.vocab_size, self.pad
        )
        return to_return

    def __len__(self) -> int:
        return len(self.l)


class T9FSADataModule(LightningDataModule):
    def __init__(
        self,
        prefix,
        lang,
        sub_size=None,
        limit: Optional[int] = None,
        serialize_prefix: str = None,
        batch_size: int = 1,
        pad: int = 0,
        num_workers: int = 1,
        vocab_size=None,
        proposal_distribution: str = "ps",
        language: str = None,
    ):
        super().__init__()
        self.prefix = prefix
        self.lang = lang
        self.train, self.val, self.test = None, None, None
        self.serialize_prefix = serialize_prefix
        self.batch_size = batch_size
        self.pad = pad
        self.num_workers = num_workers
        self.language = language
        self.sub_size = sub_size
        # called only on 1 GPU
        loaded = self.load_from_splits(limit, pad, proposal_distribution, vocab_size)
        self.splits = loaded

    def load_from_splits(self, limit, pad, proposal_distribution, vocab_size):
        loaded = []
        if self.language is not None:
            # TODO: use duplicate codes for now for transliteration dataset
            for dataset_split in ("train", "dev", "test"):
                file_path = f"{self.prefix}/{dataset_split}/{self.language}_{dataset_split}_filter_10_{self.sub_size}.tsv"
                # FIXME: remove this to use train data
                # if dataset_split == "train":
                #     file_path = f"{self.prefix}/dev/{self.language}_dev_filter_10_{self.sub_size}.tsv"

                current_split = []
                current_fst_split = []

                with open(file_path, "r") as f:
                    for l_idx, line in enumerate(tqdm.tqdm(f)):
                        if l_idx > limit:
                            break
                        name = f"{self.serialize_prefix}/{dataset_split}.{l_idx}"
                        current_split.append(name)

                if proposal_distribution == "ps":
                    current_fst_split = None
                loaded.append(
                    FSADataset(
                        current_split,
                        vocab_size=vocab_size,
                        pad=pad,
                        list_of_wfst_proposals=current_fst_split,
                    )
                )
                print(f"split {dataset_split}: {len(current_split)}")
                print(f"file: {file_path}")
            return loaded
        else:
            for dataset_split in ("train", "valid", "test"):
                current_split = []
                current_fst_split = []
                file_path = f"{self.prefix}/{dataset_split}"
                # with open(f'{file_path}.en') as g_fh, open(f'{file_path}.cipher') as p_fh:
                if self.lang == "qa":
                    self.lang = "tydiqa"
                with open(f"{file_path}.{self.lang}.in") as g_fh, open(
                    f"{file_path}.{self.lang}.out"
                ) as p_fh:
                    for l_idx, (g_l, p_l) in enumerate(
                        tqdm.tqdm(islice(zip(g_fh, p_fh), limit))
                    ):
                        name = f"{self.serialize_prefix}/{dataset_split}.{l_idx}"
                        current_split.append(name)

                    if proposal_distribution == "ps":
                        current_fst_split = None
                    loaded.append(
                        FSADataset(
                            current_split,
                            vocab_size=vocab_size,
                            pad=pad,
                            list_of_wfst_proposals=current_fst_split,
                        )
                    )
                print(f"split {dataset_split}: {len(current_split)}")
                print(f"file: {file_path}")
            return loaded

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if stage != "test":
            self.train = self.splits[0]
            self.val = self.splits[1]
            self.test = self.splits[1]
        else:
            self.test = self.splits[2]

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=1,
            num_workers=1,
            pin_memory=True,
            collate_fn=self.collate,
        )

    def collate(self, batch):
        pad = self.pad
        from torch import from_numpy

        padded_list = []
        for i in range(6):
            padded_list.append(
                Utils.pad_sequence(
                    [_[i] for _ in batch], batch_first=True, padding_value=pad
                )
            )
        return tuple([from_numpy(_) for _ in padded_list])

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        parent_parser.add_argument("--batch-size", default=2, type=int)
        parent_parser.add_argument("--num-workers", default=4, type=int)
        parent_parser.add_argument("--normalized-fst-prefix", default=".", type=str)
        return parent_parser
