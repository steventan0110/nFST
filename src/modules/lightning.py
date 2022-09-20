from typing import *
import numpy as np
from numpy import load
import pytorch_lightning as pl
import torch
import pickle
from torch.nn.modules import Embedding
from src.util.preprocess_util import Vocab, Utils

# from src.modules import scorers, sampler, nre_scorers, transducers, queries
from src.modules.scorers import (
    GlobalCompositionNROScorer,
    WFSTScorer,
    StaticRNNScorer,
)
from src.modules.estimatros import Estimators
from src.modules.samplers import Sampler
from src.modules.nros import NRO
from src.modules.scorers import FSAGRUScorer, FSAMaskScorer
from src.modules.queries import QueryPiGivenX, QueryPiGivenXAndY

import logging

logger = logging.getLogger("LightningTrain")


class ZTable:
    mapping = dict()
    exit_states = set()


class JointProb(pl.LightningModule):
    def __init__(self, args):
        super(JointProb, self).__init__()
        self.proposal_tuning_k = args.proposal_tuning_k
        self.estimator = args.estimator
        self.locally_normalized = args.locally_normalized
        assert self.locally_normalized  # only work on this version

        self.tilde_p_choice = args.tilde_p_choice
        self.proposal_distribution = args.proposal_dist

        self.tune_p = args.tune_p
        self.tune_q = args.tune_q
        self.tune_num = args.tune_num
        self.tune_global = args.tune_global
        self.tune_denom = args.tune_denom

        self.tilde_p_type = args.tilde_p_type
        self.learning_rate = args.learning_rate

        self.structured_entropy_regularization = float(
            args.structured_entropy_regularization
        )
        self.nro = args.nro
        self.embeddings = Embedding(
            args.vocab_size,
            args.tilde_p_hid_dim,
            args.pad,
        )

        self.register_buffer(
            "v_range",
            torch.arange(
                Vocab.size(),
            ),
        )
        self.pure_wfst_scorer = None
        self.mark_scores = None

        if (
            args.tilde_p_choice == "wfst-nfst-composition"
            or args.tilde_p_choice == "wfst-nfst-composition-dummy"
        ):
            # self.fixed_wfst_marks = torch.nn.Parameter(self.get_fixed_wfst_marks())
            # wrapped_wfst_scorer = WFSTScorer(
            #     pad=args.pad,
            #     eos=args.eos,
            #     bos=args.bos,
            #     scorer=self.score_fixed_wfst_marks,
            # )
            # wfst_scorer = NRO.nro_wfst_g(
            #     Vocab.lookup("wfst-start"),
            #     Vocab.lookup("wfst-end"),
            #     g=wrapped_wfst_scorer.evaluate_seq,
            #     pad=args.pad,
            #     substract_z=False,
            # )
            nfst_scorer = self.get_base(args, self.embeddings, nro=self.nro)
            self.nfst_scorer = nfst_scorer
            # self.wfst_scorer = wfst_scorer
            # composition_scoring = NRO.nro_compose_g(
            #     Vocab.lookup("cp1"),
            #     Vocab.lookup("cp2"),
            #     Vocab.lookup("cpstart"),
            #     Vocab.lookup("cpend"),
            #     args.pad,
            #     args.eos,
            #     self.wfst_scorer,
            #     self.nfst_scorer,
            # )

            # self.composition_scoring = composition_scoring
            if args.tilde_p_choice == "wfst-nfst-composition-dummy":
                # used wfst-nfst-composition-dummy in cipher case
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

        proposal_embeddings = Embedding(
            args.vocab_size,
            args.hid_dim,
            args.pad,
        )
        self.input_queries = QueryPiGivenX(
            args.hid_dim,
            num_layers=args.num_layers,
            state_hid_dim=args.hid_dim,
            embedder=proposal_embeddings,
            vocab_size=args.vocab_size,
            pad=args.pad,
            bos=args.bos,
            eos=args.eos,
            drop=args.dropout,
        )
        self.io_queries = QueryPiGivenXAndY(
            args.hid_dim,
            num_layers=args.num_layers,
            state_hid_dim=args.hid_dim,
            embedder=proposal_embeddings,
            vocab_size=args.vocab_size,
            pad=args.pad,
            bos=args.bos,
            eos=args.eos,
            drop=args.dropout,
        )

        self.global_queries = None
        self.global_proposal_dist = None
        self.denom_proposal_dist = FSAGRUScorer(
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
            embeddings=proposal_embeddings,
            query=self.input_queries,
            tied_embeddings=args.tied_proposal_embeddings,
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
            embeddings=proposal_embeddings,
            query=self.io_queries,
            tied_embeddings=args.tied_proposal_embeddings,
        )
        self.regularization_dist = FSAMaskScorer(
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
            embeddings=proposal_embeddings,
            tied_embeddings=args.tied_proposal_embeddings,
        )

        self.proposal_modules = torch.nn.modules.ModuleList(
            [
                self.denom_proposal_dist,
                self.num_proposal_dist,
                self.input_queries,
                self.io_queries,
            ]
        )
        logger.info(self.proposal_modules)
        self.num_sampler = Sampler(self.num_proposal_dist)
        self.global_sampler = None  # sampler.Sampler(self.global_proposal_dist, )
        self.regularization_sampler = Sampler(self.regularization_dist)

        self.denom_sampler = Sampler(
            self.denom_proposal_dist,
        )

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
        if wfst_emission is not None:
            (
                batch_size,
                cond_denom_emission,
                cond_denom_transition,
                denom_emission,
                denom_transition,
            ) = self.get_denom_matrices_and_batch_size(
                denom_emission,
                denom_transition,
                numerator_emission,
                numerator_transition,
                wfst_emission,
                wfst_transition,
            )
        else:
            (
                batch_size,
                cond_denom_emission,
                cond_denom_transition,
                denom_emission,
                denom_transition,
            ) = self.get_denom_matrices_and_batch_size(
                denom_emission,
                denom_transition,
                numerator_emission,
                numerator_transition,
            )
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
        num_samples_flattened = num_samples.reshape(
            batch_size * self.proposal_tuning_k, -1
        )
        with torch.no_grad():
            tilde_p = self.tilde_p(self.num_sampler.stripping_pad(num_samples))
            tilde_p_over_q = (tilde_p - log_q_num).reshape(
                batch_size, self.proposal_tuning_k
            )

            elbo: torch.Tensor = (tilde_p_over_q).detach()
            weights = torch.softmax(tilde_p_over_q, dim=1).detach()
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
            to_return["num_loss"] = -(weights * log_q_num).sum(dim=1).mean()

        if tune_global:
            self.global_sampler.set_masks(denom_transition, denom_emission)
            self.global_sampler.set_k(self.proposal_tuning_k)
            log_q_global = self.global_sampler.sample(
                batch_size * self.proposal_tuning_k, num_samples_flattened
            )[0]
            log_q_global = log_q_global.reshape(batch_size, self.proposal_tuning_k)
            to_return["global_loss"] = (weights * log_q_global).sum(dim=1).mean()

        if tune_denom:
            self.denom_sampler.set_masks(cond_denom_transition, cond_denom_emission)
            self.denom_sampler.set_k(self.proposal_tuning_k)
            log_q_denom = self.denom_sampler.sample(
                batch_size * self.proposal_tuning_k,
                num_samples_flattened,
                query_args={
                    "to_encode": gs_expanded,
                    "to_encode_mask": gs_expanded_mask,
                },
            )[0].reshape(batch_size, self.proposal_tuning_k)
            to_return["denom_loss"] = (weights * log_q_denom).sum(dim=1).mean()

            with torch.no_grad():
                log_q_prop, log_q_sampled = self.denom_sampler.sample(
                    batch_size * self.proposal_tuning_k,
                    query_args={
                        "to_encode": gs_expanded,
                        "to_encode_mask": gs_expanded_mask,
                    },
                )
                stripped = self.denom_sampler.stripping_pad(log_q_sampled)
                log_tilde_p = self.tilde_p(stripped)
                kl_q_tilde_p = torch.mean(log_q_prop - log_tilde_p)
                to_return["kl_q_tilde_p"] = kl_q_tilde_p

        return to_return

    def get_denom_matrices_and_batch_size(
        self,
        denom_emission,
        denom_transition,
        numerator_emission,
        numerator_transition,
        wfst_emission: Optional[torch.Tensor] = None,
        wfst_transition: Optional[torch.Tensor] = None,
    ):
        batch_size = numerator_transition.shape[0]
        cond_denom_emission = denom_emission.expand(batch_size, -1, -1)
        cond_denom_transition = denom_transition.expand(batch_size, -1, -1)
        if self.z_emission is not None:
            denom_emission = self.z_emission.expand(batch_size, -1, -1)
            denom_transition = self.z_transition.expand(batch_size, -1, -1)
        assert numerator_emission.shape[0] == batch_size
        assert denom_transition.shape[0] == batch_size
        assert denom_emission.shape[0] == batch_size
        if wfst_emission is None:
            return (
                batch_size,
                cond_denom_emission,
                cond_denom_transition,
                denom_emission,
                denom_transition,
            )
        wfst_emission = wfst_emission.expand(batch_size, -1, -1)
        wfst_transition = wfst_transition.expand(batch_size, -1, -1)
        assert wfst_emission.shape[0] == batch_size
        assert wfst_transition.shape[0] == batch_size
        return (
            batch_size,
            wfst_emission,
            wfst_transition,
            wfst_emission,
            wfst_transition,
        )

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
                proposal, self.tilde_p, emission.shape[0], k_prime, query_args
            )

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
        (
            batch_size,
            cond_denom_emission,
            cond_denom_transition,
            denom_emission,
            denom_transition,
        ) = self.get_denom_matrices_and_batch_size(
            denom_emission, denom_transition, numerator_emission, numerator_transition
        )
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
                return self.train_q(batch, first_six)
            elif optimizer_idx == 1:
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
        if self.structured_entropy_regularization > 0:
            p_loss -= self.structured_ent_reg()
        self.log("train_loss", p_loss.detach())
        return p_loss

    def train_q(self, batch, first_six):
        wfst_emission, wfst_transition = None, None
        if len(batch) > 6:
            wfst_emission, wfst_transition = batch[6], batch[7]
        loss_dict = self.tune_proposal(
            *first_six,
            wfst_emission=wfst_emission,
            wfst_transition=wfst_transition,
            tune_num=self.tune_num,
            tune_denom=self.tune_denom,
            tune_global=self.tune_global,
        )
        q_loss = sum([_ for _ in loss_dict.values() if _ is not None])
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
        free_sample_size = 32

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

        model_optimizer = Adam(
            self.tilde_p.parameters(), lr=self.learning_rate if self.tune_p else 0.0
        )
        proposal_optimizer = Adam(
            self.proposal_modules.parameters(),
            lr=self.learning_rate if self.tune_q else 0.0,
        )

        model_optim_config = {
            "optimizer": model_optimizer,
            "scheduler": ReduceLROnPlateau(model_optimizer, mode="min", verbose=True),
            "monitor": "val_loss",
        }
        proposal_optim_config = {
            "optimizer": proposal_optimizer,
            "scheduler": ReduceLROnPlateau(
                proposal_optimizer,
                mode="min",
                verbose=True,
            ),
            "monitor": "val_q_num_loss",
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

    def set_mark_scores(self, pad):
        embedded = self.embeddings(self.v_range)
        neg_inf = torch.log(
            torch.zeros_like(self.v_range, dtype=torch.get_default_dtype())
        )
        self.mark_scores = self.pure_wfst_scorer(embedded).reshape(self.v_range.shape)
        self.mark_scores = torch.nn.functional.log_softmax(self.mark_scores, dim=0)
        self.mark_scores = torch.where(
            self.v_range == pad, torch.zeros_like(self.mark_scores), self.mark_scores
        )
        self.mark_scores = torch.where(self.v_range == 0, neg_inf, self.mark_scores)

    def score_marks(self, t: torch.Tensor):
        return self.mark_scores[t]

    def score_fixed_wfst_marks(self, t: torch.Tensor):
        return self.fixed_wfst_marks[t]

    def get_fixed_wfst_marks(self):
        to_return = np.empty(Vocab.size())
        for k in range(Vocab.size()):
            w = 0.0
            try:
                r_lookedup = Vocab.r_lookup(k)
                if isinstance(r_lookedup, str) and r_lookedup.startswith("wfst#"):
                    w = float(r_lookedup.split("#")[1])
                    print(f"found key {r_lookedup}\t{w}")
            except KeyError as e:
                pass
            to_return[k] = w

        return torch.from_numpy(to_return).to(torch.get_default_dtype())

    def structured_ent_reg(self):
        k = 16
        assert self.structured_entropy_regularization > 0

        self.regularization_sampler.set_masks(
            transition=self.local_state_matrices_transition,
            emission=self.local_state_matrices_emission,
        )
        self.regularization_sampler.set_k(k)
        _, sampled = self.regularization_sampler.sample(
            k,
        )
        unnormalized_log_p = self.tilde_p(
            self.regularization_sampler.stripping_pad(sampled)
        )
        return (
            unnormalized_log_p.sum(dim=-1).mean()
            * self.structured_entropy_regularization
        )

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
