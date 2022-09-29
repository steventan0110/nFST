from typing import Optional, Tuple, Dict

import numpy as np
import torch
from torch.distributions import Categorical
from src.modules.scorers import LeftToRightScorer
from src.util.preprocess_util import Utils
from src.modules.scorers import FSAGRUScorer


def get_model(model) -> LeftToRightScorer:
    from torch.nn.parallel import DistributedDataParallel

    if isinstance(model, DistributedDataParallel):
        to_return: LeftToRightScorer = model.module
    else:
        to_return: LeftToRightScorer = model
    return to_return


class Sampler:
    def __init__(self, model: LeftToRightScorer):
        super(Sampler, self).__init__()
        self.model: LeftToRightScorer = get_model(model)
        # self.model = model

    @property
    def self_normalized(self):
        return self.model.self_normalized

    def set_masks(self, transition, emission):
        assert isinstance(self.model, FSAGRUScorer)
        self.model.set_masks(emission=emission, transition=transition)

    def set_k(self, k: int):
        self.model.set_k(k)

    def two_machine_sample(
        self,
        transition_1,
        emission_1,
        transition_2,
        emission_2,
        batch_size: int,
        log_p_s1_p1: Optional[torch.Tensor],
        sample_1: Optional[torch.Tensor],
    ):
        """

        :param transition_1:
        :param emission_1:
        :param transition_2:
        :param emission_2:
        :return:
        """

        # set first machine
        if log_p_s1_p1 is None:
            self.set_masks(transition_1, emission_1)
            log_p_s1_p1, sample_1 = self.sample(
                batch_size,
            )
        self.set_masks(transition_2, emission_2)
        log_p_s1_p2, _ = self.sample(batch_size, sample_1)

        log_p_s1 = torch.logsumexp(
            torch.stack((log_p_s1_p1, log_p_s1_p2), dim=1), dim=1
        ) - np.log(2.0)

        # set second machine
        self.set_masks(transition_2, emission_2)
        log_p_s2_p2, sample_2 = self.sample(
            batch_size,
        )
        self.set_masks(transition_1, emission_1)
        log_p_s2_p1, _ = self.sample(batch_size, sample_2)
        # FIXME we are replacing nan probs with -inf, which is probably not the most robust thing we can do...
        log_p_s2_p1.masked_fill_(log_p_s2_p1.isnan(), float("-inf"))
        log_p_s2 = torch.logsumexp(
            torch.stack((log_p_s2_p1, log_p_s2_p2), dim=1), dim=1
        ) - np.log(2.0)

        if sample_2.shape[1] > sample_1.shape[1]:
            paddings = torch.empty(
                batch_size,
                sample_2.shape[1] - sample_1.shape[1],
                device=sample_1.device,
            )
            paddings.fill_(self.model.__pad__)
            sample_1 = torch.cat((sample_1, paddings), dim=1)
        if sample_1.shape[1] > sample_2.shape[1]:
            paddings = torch.empty(
                batch_size,
                sample_1.shape[1] - sample_2.shape[1],
                device=sample_1.device,
            )
            paddings.fill_(self.model.__pad__)
            sample_2 = torch.cat((sample_2, paddings), dim=1)

        samples_to_return = torch.cat((sample_1, sample_2), dim=0)
        log_p = torch.cat((log_p_s1, log_p_s2), dim=0)
        return log_p, samples_to_return

    def score_and_mask(
        self, seq: torch.Tensor, bptt: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        scores sequences and outputs states / masks that are suitable for other encoding modules
        :param seq:
        :param bptt:
        :return: final state, states of all timesteps, mask, scores
        """
        model: LeftToRightScorer = self.model
        mask = seq != model.__pad__
        scores, hx = self.stateful_sample(
            batch_size=seq.shape[0],
            to_evaluate=seq,
            no_zs=True,
            detach_every=bptt,
            only_return_final_state=False,
        )
        final_hx = hx[:, -1, :]
        return final_hx, hx, mask, scores

    def all_reached_eos(self, candidates: torch.Tensor):
        """

        :param candidates:
        :return:
        """
        model: LeftToRightScorer = self.model
        # size = int(candidates.shape[0])
        s_pad = (candidates == model.__pad__).all()

        return s_pad

    def sample(
        self,
        batch_size: int,
        to_evaluate: Optional[torch.Tensor] = None,
        no_zs: bool = False,
        temperature: float = 1.0,
        query_args: Optional[Dict] = None,
    ):
        if isinstance(self.model, FSAGRUScorer):
            assert (
                batch_size
                == self.model.transition_k.shape[0]
                == self.model.emission_k.shape[0]
            )
        # print(batch_size)
        # print(to_evaluate)
        # print(query_args)
        results = self.stateful_sample(
            batch_size=batch_size,
            to_evaluate=to_evaluate,
            hx=None,
            inp=None,
            no_zs=no_zs,
            temperature=temperature,
            query_args=query_args,
        )
        return results[:-1]

    def stripping_pad(self, sequences: torch.Tensor) -> torch.Tensor:
        assert len(sequences.shape) == 2
        batch_size = sequences.shape[0]
        seq_len = sequences.shape[1]
        to_return = torch.empty_like(sequences)
        to_return.fill_(self.model.__pad__)
        zeros = torch.zeros(batch_size, dtype=torch.long, device=sequences.device)
        ones = torch.ones(batch_size, dtype=torch.long, device=sequences.device)

        indices = torch.zeros_like(zeros, dtype=torch.long, device=sequences.device)
        with torch.no_grad():
            for i in range(seq_len):
                to_return[torch.arange(batch_size), indices] = sequences[:, i]
                zero_indicator: torch.Tensor = sequences[:, i] == 0
                indices = indices + torch.where(zero_indicator, zeros, ones)
                pad_indicator: torch.Tensor = sequences[:, i] == self.model.__pad__
                if torch.all(pad_indicator):
                    break
        return to_return[:, : i + 1].contiguous()

    def stateful_sample(
        self,
        batch_size: int,
        to_evaluate: Optional[torch.Tensor] = None,
        hx: Optional[torch.Tensor] = None,
        inp: Optional[torch.Tensor] = None,
        no_zs: bool = False,
        detach_every: Optional[int] = None,
        to_encode: Optional[torch.Tensor] = None,
        to_encode_mask: Optional[torch.Tensor] = None,
        only_return_final_state: bool = True,
        temperature: float = 1.0,
        query_args: Optional[Dict] = None,
    ):

        model: LeftToRightScorer = self.model
        if model.has_query():
            assert query_args is not None
            assert "to_encode" in query_args.keys()
            assert "to_encode_mask" in query_args.keys()
            model.register_query(**query_args)
        if to_evaluate is not None:
            assert int(to_evaluate.shape[0]) == batch_size
            additional_pad = torch.empty(
                (batch_size, 1), device=to_evaluate.device, dtype=torch.long
            )
            additional_pad.fill_(model.__pad__)
            to_evaluate_padded = torch.cat((to_evaluate, additional_pad), dim=1)

            evaluate_only = True
            return_zs = True
            return_sampled_sequences = False
        else:
            evaluate_only = False
            return_zs = False
            return_sampled_sequences = True

        if no_zs:
            return_zs = False

        new_hx, new_inp, metadata = model.get_init_states(
            batch_size, device=self.model.embeddings.weight.device
        )

        if hx is None:
            hx = new_hx
        if inp is None:
            inp = new_inp

        log_probs = []
        prefixes = []
        zs = []
        intermediate_hx = []

        hard_cut = True
        for timestep in range(
            model.max_length + 1
        ):  # + 1 for the additional padding at the end

            new_hx, masked_out_invalid, updated_metadata = model.left_to_right_score(
                left_h=hx, left_inp=inp, metadata=metadata
            )

            # build distribution over next symbols
            # no need to normalize the probs ourselves -- pytorch does that for us
            masked_out_invalid = masked_out_invalid / temperature
            # assert torch.all(torch.logsumexp(masked_out_invalid, dim=-1) > float('-inf'))

            dist_over_next = Categorical(logits=masked_out_invalid)

            try:
                if evaluate_only:
                    sampled_next_symbol = to_evaluate_padded[:, timestep]
                else:
                    sampled_next_symbol = dist_over_next.sample()
            except Exception as e:
                print(f"{e}\t{timestep}")
                print(prefixes)
                raise e
            metadata = model.metadata_callback(
                new_hx, sampled_next_symbol, updated_metadata
            )

            # also get log prob
            sampled_next_symbol_log_prob = dist_over_next.log_prob(sampled_next_symbol)
            intermediate_hx.append(
                (
                    new_hx
                    * (inp != model.__pad__)[:, np.newaxis]
                    .expand(-1, new_hx.shape[1])
                    .to(new_hx.dtype)
                )[:, np.newaxis, :]
            )

            if return_zs:
                zs.append(torch.logsumexp(masked_out_invalid, dim=1))
            if return_sampled_sequences:
                prefixes.append(sampled_next_symbol)
            log_probs.append(sampled_next_symbol_log_prob)
            if self.all_reached_eos(sampled_next_symbol):
                hard_cut = False
                break

            if detach_every is not None:
                if (timestep + 1) % detach_every == 0:
                    new_hx = new_hx.detach()

            inp = sampled_next_symbol
            hx = new_hx

        if hard_cut:
            raise Exception(
                "ran out of length budget! This should't be possible though."
            )

        # the prefixes will always end with a padding symbol at the end.
        # We are chopping it off to ensure consistency with the evaluate_only case.
        if len(prefixes) > 0:
            prefixes.pop(-1)

        if return_sampled_sequences:
            assert len(intermediate_hx) == len(prefixes) + 1
        if evaluate_only:
            assert len(intermediate_hx) == to_evaluate.shape[1] + 1, "{} / {}".format(
                len(intermediate_hx), to_evaluate.shape[1]
            )

        stacked_log_probs = torch.stack(log_probs, dim=0)
        del log_probs
        prefixes_stacked = torch.stack(prefixes, dim=1)
        del prefixes
        summed_log_probs = stacked_log_probs.sum(dim=0)
        if len(zs) > 0:
            stacked_zs = torch.stack(zs, dim=0).sum(dim=0)
        del zs
        if only_return_final_state:
            hx_to_return = intermediate_hx[-1][:, 0, :]
        else:
            hx_to_return = torch.cat(intermediate_hx, dim=1)
        del intermediate_hx
        if evaluate_only:
            if return_zs:
                return summed_log_probs, stacked_zs, hx_to_return
            else:
                # print('summed_log_probs: {} individuals: {}'.format(summed_log_probs[0], torch.stack(log_probs, dim=0)[:, 0]))
                return summed_log_probs, hx_to_return
        return summed_log_probs, prefixes_stacked, hx_to_return


class EmpiricalMixtureSampler(Sampler):
    def __init__(self, model: LeftToRightScorer, log_p: float):
        super(EmpiricalMixtureSampler, self).__init__(model)

        assert log_p < 0.0
        self.log_p = log_p
        self.log_1mp = np.log1p(-np.exp(log_p))
        self.mixture = None

    def set_mixture(self, mixture: Optional[torch.Tensor]):
        self.mixture = mixture

    def sample(
        self,
        batch_size: int,
        to_evaluate: Optional[torch.Tensor] = None,
        query_args=Optional[Dict],
        no_zs: bool = False,
        temperature: float = 1.0,
    ):
        """

        :param batch_size:
        :param to_evaluate:
        :param query_args:
        :param no_zs:
        :param temperature:
        :return:
        """
        mixture = self.mixture
        sampled = super(EmpiricalMixtureSampler, self).sample(
            batch_size,
            to_evaluate,
            query_args=query_args,
            no_zs=no_zs,
            temperature=temperature,
        )
        if mixture is None:
            return sampled
        log_q = sampled[0]
        samples, mixture = Utils.pad_to_longer(sampled[1], mixture, self.model.__pad__)
        # prob of choosing an entry in mixture is p / mixture.shape[0]
        probs = torch.empty((mixture.shape[0] + 1,), device=mixture.device)
        probs[:-1].fill_(self.log_p)
        probs[:-1] -= np.log(mixture.shape[0])
        probs[-1] = self.log_1mp
        cate = torch.distributions.Categorical(logits=probs)
        indices = cate.sample((samples.shape[0],))
        indices_expanded = indices[:, None].expand(
            -1,
            mixture.shape[1],
        )
        mixture_expanded = torch.cat(
            (
                mixture,
                torch.zeros(
                    (1, mixture.shape[1]), dtype=mixture.dtype, device=mixture.device
                ),
            ),
            dim=0,
        )
        indexed = mixture_expanded[indices]
        replaced = torch.where(indices_expanded != mixture.shape[0], indexed, samples)

        # expand replace
        replaced_expanded = replaced[:, None, :].expand(-1, mixture.shape[0], -1)
        mixture_expanded = mixture[None, :, :].expand(replaced.shape[0], -1, -1)
        diff = torch.abs(replaced_expanded - mixture_expanded).sum(dim=2)
        find_zeros = torch.min(
            diff,
            dim=1,
        )
        log_p_hat_in_support = torch.zeros((replaced.shape[0],), device=replaced.device)
        log_p_hat_in_support -= np.log(mixture.shape[0])
        log_p_hat_out_support = torch.empty(
            (replaced.shape[0],), device=replaced.device
        )
        log_p_hat_out_support.fill_(float("-inf"))
        log_p_hat = torch.where(
            find_zeros.values == 0, log_p_hat_in_support, log_p_hat_out_support
        )
        new_log_q = torch.logsumexp(
            torch.stack((log_p_hat + self.log_p, log_q + self.log_1mp), dim=1), dim=1
        )
        return new_log_q, replaced
