from typing import *
import numpy as np
from numpy import load
import pytorch_lightning as pl
import torch
import pickle
from torch.nn.modules import Embedding
from src.modules.transformer import TransformerLM, GPT2Wrapper
from src.util.preprocess_util import Vocab, Utils

# from src.modules import scorers, sampler, nre_scorers, transducers, queries
from src.modules.scorers import (
    GlobalCompositionNROScorer,
    StaticRNNScorer,
)
from src.modules.estimatros import Estimators
from src.modules.samplers import Sampler
from src.modules.scorers import FSAGRUScorer, FSAMaskScorer
from src.modules.transformer_scorer import GPTScorer, TransformerSampler
from src.modules.queries import QueryPiGivenXAndY, QueryPiGivenXAndYTransformer
from transformers import AdamW

import logging


logger = logging.getLogger("LightningTrain")


class ZTable:
    mapping = dict()
    exit_states = set()


def param_sum(model):
    p_sum = 0
    for p in model.parameters():
        p_sum += p.sum().item()
    return p_sum


class JointProb(pl.LightningModule):
    def __init__(self, args):
        super(JointProb, self).__init__()
        self.proposal_tuning_k = args.proposal_tuning_k
        self.estimator = args.estimator
        self.locally_normalized = args.locally_normalized
        self.warmup_steps = args.warmup_steps
        assert self.locally_normalized  # only work on this version

        self.tilde_p_choice = args.tilde_p_choice
        self.tilde_p_type = args.tilde_p_type
        self.p_type = args.p_type
        self.proposal_distribution = args.proposal_dist

        self.tune_p = args.tune_p
        self.tune_q = args.tune_q
        self.tune_num = args.tune_num
        self.tune_global = args.tune_global
        self.tune_denom = args.tune_denom

        self.tilde_p_type = args.tilde_p_type
        self.learning_rate = args.learning_rate
        self.tilde_p_lr = args.tilde_p_lr

        self.nro = args.nro
        self.vocab_size = args.vocab_size
        self.embeddings = Embedding(
            args.vocab_size,
            args.tilde_p_hid_dim,
            args.pad,
        )
        self.step = 0  # track number of updataes for printout

        if (
            args.tilde_p_choice == "wfst-nfst-composition"
            or args.tilde_p_choice == "wfst-nfst-composition-dummy"
        ):
            nfst_scorer = self.get_base(args, self.embeddings, nro=self.nro)
            self.nfst_scorer = nfst_scorer
            nre = self.nfst_scorer

        elif args.tilde_p_choice == "nfst-composition":
            nre = GlobalCompositionNROScorer(
                args.tilde_p_hid_dim,
                args.vocab_size,
                pad=args.pad,
                eos=args.eos,
                bos=args.bos,
                mask=args.mask,
            )
            self.nfst_scorer = nre

        self.tilde_p = nre
        self.pad = args.pad
        proposal_embedding = Embedding(
            args.vocab_size,
            args.hid_dim,
            args.pad,
        )
        if args.p_type == "beta":
            logger.info("Compute Beta prob for proposal")
            self.num_proposal_dist = FSAGRUScorer(
                args.hid_dim,
                args.vocab_size,
                bos=args.bos,
                eos=args.eos,
                pad=args.pad,
                insert_penalty=args.insert_penalty,
                insert_threshold=args.insert_threshold,
                length_threshold=args.length_threshold,
                length_penalty=args.length_penalty,
                max_length=args.max_length,
                embeddings=proposal_embedding,
                tied_embeddings=args.tied_proposal_embeddings,
                use_beta=True,  # change into parameters for computing Tree-LSTM later
            )
            self.proposal_modules = torch.nn.ModuleList(
                [
                    self.num_proposal_dist,
                ]
            )
            # self.num_sampler = TransformerSampler(self.num_proposal_dist)
            self.num_sampler = Sampler(self.num_proposal_dist)
        elif args.p_type == "transformer":
            logger.info("Using transformer for proposal")
            self.io_queries = QueryPiGivenXAndYTransformer(
                args.hid_dim,
                num_layers=args.num_layers,
                state_hid_dim=args.hid_dim,
                num_heads=args.num_heads,
                vocab_size=args.vocab_size,
                pad=args.pad,
                bos=args.bos,
                eos=args.eos,
                drop=args.dropout,
                max_length=args.max_length,
            )
            # TODO update scorer and sampler with transformer as well
            self.num_proposal_dist = FSAGRUScorer(
                args.hid_dim,
                args.vocab_size,
                bos=args.bos,
                eos=args.eos,
                pad=args.pad,
                insert_penalty=args.insert_penalty,
                insert_threshold=args.insert_threshold,
                length_threshold=args.length_threshold,
                length_penalty=args.length_penalty,
                max_length=args.max_length,
                embeddings=proposal_embedding,
                query=self.io_queries,
                tied_embeddings=args.tied_proposal_embeddings,
            )
            self.proposal_modules = torch.nn.ModuleList(
                [
                    self.num_proposal_dist,
                    self.io_queries,
                ]
            )
            # self.num_sampler = TransformerSampler(self.num_proposal_dist)
            self.num_sampler = Sampler(self.num_proposal_dist)
        else:
            self.io_queries = QueryPiGivenXAndY(
                args.hid_dim,
                num_layers=args.num_layers,
                state_hid_dim=args.hid_dim,
                embedder=proposal_embedding,
                vocab_size=args.vocab_size,
                pad=args.pad,
                bos=args.bos,
                eos=args.eos,
                drop=args.dropout,
            )
            self.num_proposal_dist = FSAGRUScorer(
                args.hid_dim,
                args.vocab_size,
                bos=args.bos,
                eos=args.eos,
                pad=args.pad,
                insert_penalty=args.insert_penalty,
                insert_threshold=args.insert_threshold,
                length_threshold=args.length_threshold,
                length_penalty=args.length_penalty,
                max_length=args.max_length,
                embeddings=proposal_embedding,
                query=self.io_queries,
                tied_embeddings=args.tied_proposal_embeddings,
            )
            self.proposal_modules = torch.nn.ModuleList(
                [
                    self.num_proposal_dist,
                    self.io_queries,
                ]
            )
            self.num_sampler = Sampler(self.num_proposal_dist)

        # check if pretrain ckpt changed the params
        # print("before loading ckpt")
        # print(param_sum(self.num_sampler.model))
        # print(param_sum(self.proposal_modules))
        if args.use_pretrain_proposal:
            num_proposal_dist, io_queries, proposal_modules = self.load_proposal(args)
            self.io_queries = io_queries
            self.num_proposal_dist = num_proposal_dist
            self.num_sampler = Sampler(num_proposal_dist)
            self.proposal_modules = proposal_modules
            # print("after loading ckpt")
            # print(param_sum(self.num_sampler.model))
            # print(param_sum(self.proposal_modules))

        self.k = args.k
        if args.prob_definition == "joint":
            z_name = f"{args.serialize_prefix}/base.npz"
            loaded = load(z_name)
            for key in (
                "z_emission",
                "z_transition",
                "local_state_matrices_emission",
                "local_state_matrices_transition",
            ):
                loaded_matrix = torch.from_numpy(loaded[key])
                reshaped_matrix = loaded_matrix.reshape(
                    (1, loaded_matrix.shape[0], loaded_matrix.shape[1])
                )
                logger.info(f"{key} has shape: {reshaped_matrix.shape}")
                self.register_buffer(key, reshaped_matrix)
            self.validation_z_set = False
        else:
            self.z_emission = None

        # FIXME: Composition with pre-trained checkpoints
        # machine_types = self.get_types_and_checkpoints(args)
        # self._pretrained_model_types = torch.nn.ModuleDict()
        # self._pretrained_model_types.update(machine_types)
        # self.set_pretrained()

        # self._pretrained_plang = self.get_plang_from_checkpoint(args)
        # if self._pretrained_plang is None:
        #     logger.warning("pretrain plang is None")
        # self._do_not_finetune_plang = False
        # if hasattr(args, "do_not_finetune_plang"):
        #     self._do_not_finetune_plang = args.do_not_finetune_plang
        # self.set_plang()

    def load_proposal(self, args):
        hard_code_path = "/scratch4/jeisner1/nfst/tr-filter-sub-20000/outputs/model/ur/p-type-transformer/tilde-p-type-LSTM/lr-1e-05/tilde-p-lr-0.001/hid-512/layers-6/tilde-p-hid-256/tilde-p-layers-2/drop-0.3/clip-5.0/bsz-8/max-seq-len-300/length-threshold-290/estimator-iwae/proposal-ps/nheads-8/k-64/tune-q-True/tied-mark-embeddings-False/label-smoothing-0.1/two-level-marks-False/tilde-p-choice-wfst-nfst-composition-dummy/lightning_logs/version_0/checkpoints/wfst-nfst-composition-dummy-iwae-epoch=10-val_loss=21.65.ckpt"
        from copy import deepcopy

        new_args = deepcopy(args)
        new_args["tilde_p_num_layers"] = 2
        new_args["tilde_p_type"] = "LSTM"
        new_args["use_pretrain_proposal"] = False
        # hard code to make the scorer consistent with prev checkpoitn
        new_args["tilde_p_hid_dim"] = 256

        pretrain_model = JointProb.load_from_checkpoint(
            hard_code_path, "cpu", args=new_args
        )
        # print(pretrain_model.num_proposal_dist)
        # print(pretrain_model.io_queries)
        # print(pretrain_model.proposal_modules)
        # raise RuntimeError
        return (
            pretrain_model.num_proposal_dist,
            pretrain_model.io_queries,
            pretrain_model.proposal_modules,
        )

    def set_plang(self):
        if (
            isinstance(self.nfst_scorer, GlobalCompositionNROScorer)
            and hasattr(self, "_pretrained_plang")
            and self._pretrained_plang is not None
        ):
            self.nfst_scorer.p_scorer = self._pretrained_plang
        if (
            isinstance(self.nfst_scorer, GlobalCompositionNROScorer)
            and self._do_not_finetune_plang
        ):
            print("freeze parameter of p_scorer because not finetuning plang")
            self.nfst_scorer.p_scorer.requires_grad_(False)

    def get_plang_from_checkpoint(self, args):
        if not hasattr(args, "pretrained_plang") or args.pretrained_plang is None:
            return None
        print(
            "load in plang, caution this might be wrong if you do not intend to compose module"
        )
        from copy import copy

        new_args = copy(args)
        try:
            del new_args.pretrained_plang
        except Exception as e:
            pass
        loaded_plang = JointProb.load_from_checkpoint(
            args.pretrained_plang, args=new_args, strict=False
        ).nfst_scorer.p_scorer
        return loaded_plang

    @staticmethod
    def get_base(args, embeddings: Optional[torch.nn.Module] = None, nro: bool = False):
        if nro:
            return GlobalCompositionNROScorer(
                args.tilde_p_hid_dim,
                args.vocab_size,
                bos=args.bos,
                pad=args.pad,
                eos=args.eos,
                max_length=args.max_length,
                embeddings=embeddings,
                locally_normalized=True,
                num_hidden_states=args.tilde_p_num_layers,
                rnn_type=args.tilde_p_type,
                tied_embeddings=args.tied_mark_embeddings,
                label_smoothing=args.label_smoothing,
                two_level_marks=args.two_level_marks,
                forced_type=args.force_type,
            )
        if args.tilde_p_type == "LSTM":
            return StaticRNNScorer(
                args.tilde_p_hid_dim,
                args.vocab_size,
                bos=args.bos,
                pad=args.pad,
                eos=args.eos,
                max_length=args.max_length,
                embeddings=embeddings,
                locally_normalized=True,
                num_hidden_states=args.tilde_p_num_layers,
                rnn_type=args.tilde_p_type,
                tied_embeddings=args.tied_mark_embeddings,
                label_smoothing=args.label_smoothing,
                two_level_marks=args.two_level_marks,
            )
        if args.tilde_p_type == "transformer":
            return GPT2Wrapper(args)

    def tune_proposal(
        self,
        numerator_emission: torch.Tensor,
        numerator_transition: torch.Tensor,
        denom_emission: torch.Tensor,
        denom_transition: torch.Tensor,
        gs,
        ps,
        wfst_emission: Optional[torch.Tensor] = None,
        wfst_transition: Optional[torch.Tensor] = None,
        tune_num: bool = True,
        tune_denom: bool = True,
        tune_global: bool = True,
    ):
        batch_size = numerator_transition.shape[0]
        self.num_sampler.set_masks(numerator_transition, numerator_emission)
        self.num_sampler.set_k(self.proposal_tuning_k)
        gs_expanded = (
            gs[:, None, :]
            .expand(-1, self.proposal_tuning_k, -1)
            .reshape(gs.shape[0] * self.proposal_tuning_k, -1)
        )
        gs_expanded_mask: torch.Tensor = self.get_mask(gs_expanded)
        ps_expanded = (
            ps[:, None, :]
            .expand(-1, self.proposal_tuning_k, -1)
            .reshape(ps.shape[0] * self.proposal_tuning_k, -1)
        )
        ps_expanded_mask: torch.Tensor = self.get_mask(ps_expanded)
        log_q_num, num_samples = self.num_sampler.sample(
            batch_size=batch_size * self.proposal_tuning_k,
            query_args={
                "to_encode_2": ps_expanded,
                "mask_2": ps_expanded_mask,
                "to_encode": gs_expanded,
                "to_encode_mask": gs_expanded_mask,
            },
        )
        # print(f"sample loss q: {log_q_num}")

        with torch.no_grad():
            tilde_p = self.tilde_p(self.num_sampler.stripping_pad(num_samples))
            # print(f"tilde_p: {tilde_p}")
            tilde_p_over_q = (tilde_p - log_q_num).reshape(
                batch_size, self.proposal_tuning_k
            )

            weights = torch.softmax(tilde_p_over_q, dim=1).detach()
            # # FIXME: use a threshold to prevent transformer from updating too early
            # if torch.mean(tilde_p_over_q) > 70:
            #     weights = weights.new_zeros(weights.shape)
        log_q_num: torch.Tensor = log_q_num.reshape(batch_size, self.proposal_tuning_k)

        to_return = {
            "num_loss": None,
            "global_loss": None,
            "denom_loss": None,
            "num_loss_rf": None,
            "global_loss_rf": None,
            "denom_loss_rf": None,
        }
        if tune_num:
            # print("debug tune q:")
            # print(log_q_num)
            # print(weights)
            to_return["num_loss"] = -(weights * log_q_num).sum(dim=1).mean()

        return to_return

    def log_marginalize(
        self,
        emission: torch.Tensor,
        transition: torch.Tensor,
        proposal: Sampler,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        k: Optional[int] = None,
    ):
        proposal.set_masks(transition=transition, emission=emission)
        k_prime = k
        biased = True
        query_args = dict()
        if x is not None:
            x_expanded, x_expanded_mask = self.expand_and_mask(x, k_prime)
            query_args["to_encode"] = x_expanded
            query_args["to_encode_mask"] = x_expanded_mask
        if y is not None:
            y_expanded, y_expanded_mask = self.expand_and_mask(y, k_prime)
            query_args["to_encode_2"] = y_expanded
            query_args["mask_2"] = y_expanded_mask
        if biased:
            log_marginalized, log_q, samples, log_w = Estimators.iwae(
                proposal,
                self.tilde_p,
                emission.shape[0],
                k_prime,
                self.step,
                query_args,
            )
            self.step += 1

        return log_marginalized, log_q, samples, log_w

    def forward(
        self,
        numerator_emission: torch.Tensor,
        numerator_transition: torch.Tensor,
        denom_emission: torch.Tensor,
        denom_transition: torch.Tensor,
        gs,
        ps,
        return_samples=False,
    ):

        """

        :param numerator_emission:
        :param numerator_transition:
        :param denom_emission:
        :param denom_transition:
        :return:
        """

        k = self.k
        # use iwae estimator, has k samples
        num_prob, _, num_samples, log_w = self.log_marginalize(
            transition=numerator_transition,
            emission=numerator_emission,
            proposal=self.num_sampler,
            x=gs,
            y=ps,
            k=k,
        )

        denom_prob = torch.zeros_like(num_prob)
        if return_samples:
            # only return the samples with highest prob
            log_w = log_w.squeeze()
            num_samples = num_samples.squeeze()
            best_sample = num_samples[torch.argmax(log_w)] if k > 1 else num_samples
            return num_prob, denom_prob, best_sample
        return num_prob, denom_prob

    def expand_and_mask(self, input, k):
        expanded = input[:, None, :].expand(-1, k, -1).reshape(input.shape[0] * k, -1)
        expanded_mask: torch.Tensor = self.get_mask(expanded)
        return expanded, expanded_mask

    def get_mask(self, to_mask):
        return (to_mask != self.num_sampler.model.__pad__).to(torch.get_default_dtype())

    def training_step(self, batch, batch_idx, optimizer_idx=0, **kwargs):
        self.validation_z_set = False
        first_six = batch[:6]
        if self.tune_p and self.tune_q:
            if optimizer_idx == 0:
                # print(param_sum(self.tilde_p))
                # print(param_sum(self.num_sampler.model))
                return self.train_q(batch, first_six)
            elif optimizer_idx == 1:
                # print(param_sum(self.tilde_p))
                # print(param_sum(self.num_sampler.model))
                return self.train_p(first_six)
            else:
                raise NotImplementedError
        elif self.tune_p:
            return self.train_p(first_six)
        elif self.tune_q:
            return self.train_q(batch, first_six)
        else:
            raise NotImplementedError

    def train_p(self, first_six):
        num_prob, denom_prob = self.forward(*first_six)
        self.log("train_num_prob", num_prob.detach().mean())
        p_loss = -(num_prob - denom_prob).mean()
        self.log("train_loss", p_loss.detach())
        return p_loss

    def train_q(self, batch, first_six):
        loss_dict = self.tune_proposal(
            *first_six,
            tune_num=self.tune_num,
            tune_denom=self.tune_denom,
            tune_global=self.tune_global,
        )
        q_loss = sum([_ for _ in loss_dict.values() if _ is not None])
        # print(f"q_loss: {q_loss}")
        if loss_dict["denom_loss"] is not None:
            self.log("kl_q_tilde_p", loss_dict["kl_q_tilde_p"])
        for k in ("num_loss", "denom_loss", "global_loss"):
            if loss_dict[k] is not None:
                self.log(f"q_{k}", loss_dict[k])
            if loss_dict[f"{k}_rf"] is not None:
                self.log(f"q_{k}_rf", loss_dict[f"{k}_rf"])
        return q_loss

    def validation_step(self, batch, batch_idx):
        if not self.validation_z_set:
            self.validation_z_set = True
        first_six = batch[:6]
        if self.tune_p:
            num_prob, denom_prob = self.forward(*first_six)
            val_loss = -(num_prob - denom_prob).mean()
            self.log("val_loss", val_loss)
            self.log(
                "val_denom",
                denom_prob,
            )
            self.log(
                "val_num",
                num_prob,
            )

        if self.tune_q:
            q_loss_dict = self.tune_proposal(
                *batch,
                tune_num=self.tune_num,
                tune_denom=self.tune_denom,
                tune_global=self.tune_global,
            )

            if q_loss_dict["denom_loss"] is not None:
                self.log("val_kl_q_tilde_p", q_loss_dict["kl_q_tilde_p"])
            for k in ("num_loss", "denom_loss", "global_loss"):
                if q_loss_dict[k] is not None:
                    self.log(f"val_q_{k}", q_loss_dict[k])
                if q_loss_dict[f"{k}_rf"] is not None:
                    self.log(f"val_q_{k}_rf", q_loss_dict[f"{k}_rf"])

    def configure_optimizers(self):
        from torch.optim import Adam
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        from transformers.optimization import get_polynomial_decay_schedule_with_warmup

        model_optimizer = AdamW(
            self.tilde_p.parameters(), lr=self.tilde_p_lr if self.tune_p else 0.0
        )
        proposal_optimizer = Adam(
            self.proposal_modules.parameters(),
            lr=self.learning_rate if self.tune_q else 0.0,
        )
        if self.tune_p:
            if self.tilde_p_type != "transformer":
                logger.info("Use ReduceLR scheduler for scorer")
                # using lstm
                model_optim_config = {
                    "optimizer": model_optimizer,
                    "scheduler": ReduceLROnPlateau(
                        model_optimizer, mode="min", verbose=True
                    ),
                    "monitor": "val_loss",
                }
            else:
                logger.info("Use warmup scheduler for scorer")
                model_optim_config = {
                    "optimizer": model_optimizer,
                    "lr_scheduler": {
                        "scheduler": get_polynomial_decay_schedule_with_warmup(
                            model_optimizer,
                            num_warmup_steps=self.warmup_steps,
                            num_training_steps=40000,
                            lr_end=1e-6,
                        ),
                        "interval": "step",
                        "frequency": 1,
                        "monitor": "val_loss",
                    },
                }
        if self.tune_q:
            if self.p_type != "transformer":
                logger.info("Use ReduceLR scheduler for proposal")
                proposal_optim_config = {
                    "optimizer": proposal_optimizer,
                    "scheduler": ReduceLROnPlateau(
                        proposal_optimizer,
                        mode="min",
                        verbose=True,
                    ),
                    "monitor": "val_q_num_loss",
                }
            else:
                logger.info("Use warmup scheduler for proposal")
                proposal_optim_config = {
                    "optimizer": proposal_optimizer,
                    "lr_scheduler": {
                        "scheduler": get_polynomial_decay_schedule_with_warmup(
                            proposal_optimizer,
                            num_warmup_steps=self.warmup_steps,
                            num_training_steps=40000,
                            lr_end=1e-6,
                        ),
                        "interval": "step",
                        "frequency": 1,
                        "monitor": "val_q_num_loss",
                    },
                }
        # return proposal_optim_config

        if self.tune_p and self.tune_q:
            return proposal_optim_config, model_optim_config
        elif self.tune_p:
            return model_optim_config
        elif self.tune_q:
            return proposal_optim_config
        else:
            return proposal_optim_config, model_optim_config

    def decode_from_npz(self, npz_path, vocab_size, pad):
        single_batch = tuple(
            [
                torch.from_numpy(_).unsqueeze(0)
                for _ in Utils.load_fsa_from_npz(npz_path, None, vocab_size, pad)
            ]
        )

        l = self.forward(*single_batch, return_samples=True)
        mark = l[2]
        prob = (l[0] - l[1]).flatten()[0].item()
        return prob, mark

    def set_pretrained(self):
        for machine_type in self._pretrained_model_types.keys():
            self.nfst_scorer.networks[machine_type] = self._pretrained_model_types[
                machine_type
            ]

    def get_types_and_checkpoints(self, args):
        if (
            not hasattr(args, "pretrain_model_type")
            or not hasattr(args, "pretrain_model")
            or args.pretrain_model_type is None
            or args.pretrain_model is None
        ):
            return dict()
        from copy import copy

        new_args = copy(args)
        try:
            del new_args.pretrain_model_type
            del new_args.pretrain_model
        except Exception as e:
            pass
        machine_types = args.pretrain_model_type.split(",")
        print("Using component model from types: ", machine_types)
        pretrain_model_checkpoints_paths = args.pretrain_model.split(",")
        networks_dict = dict()
        for machine_type, checkpoint in zip(
            machine_types, pretrain_model_checkpoints_paths
        ):
            networks_dict[machine_type] = JointProb.load_from_checkpoint(
                checkpoint, args=new_args, strict=False
            ).nfst_scorer.networks[machine_type]
        assert len(networks_dict) == len(pretrain_model_checkpoints_paths)
        return networks_dict
