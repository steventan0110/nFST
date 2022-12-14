from typing import Optional, Dict
import torch
import numpy as np
from src.modules.scorers import PaddedScorer, ScorerBase
from src.modules.samplers import Sampler
from torch.functional import F


class Estimators:
    @staticmethod
    def batched_importance_sampling(
        batch_size, k_prime, proposal, query_args, unnormalized_model, step
    ):
        proposal.set_k(k_prime)
        log_q, samples = proposal.sample(
            batch_size=batch_size * k_prime, to_evaluate=None, query_args=query_args
        )
        log_q = log_q.reshape((batch_size, k_prime))
        unnormalized_log_p = unnormalized_model(proposal.stripping_pad(samples))
        samples = samples.reshape((batch_size, k_prime, -1))
        unnormalized_log_p = unnormalized_log_p.reshape((batch_size, k_prime))
        # if step % 150 == 0:
        #     print("log q value from proposal:")
        #     print(log_q[0])
        #     print("log_p value from model output")
        #     print(unnormalized_log_p[0])

        log_w = unnormalized_log_p - log_q.detach()
        log_w = log_w.reshape((batch_size, k_prime))
        return log_q, log_w, samples

    @staticmethod
    def iwae(
        proposal: Sampler,
        unnormalized_model: torch.nn.Module,
        batch_size: int,
        k: int,
        step,
        query_args: Optional[Dict] = None,
    ):
        log_q, log_w, samples = Estimators.batched_importance_sampling(
            batch_size, k, proposal, query_args, unnormalized_model, step
        )
        return torch.logsumexp(log_w, dim=1) - np.log(k), log_q, samples, log_w

    @staticmethod
    def iwae_two_machines(
        proposal: Sampler,
        unnormalized_model: torch.nn.Module,
        k: int,
        transition_1,
        emission_1,
        transition_2,
        emission_2,
        log_p1,
        s1,
    ):
        with torch.no_grad():
            log_q, samples = proposal.two_machine_sample(
                transition_1,
                emission_1,
                transition_2,
                emission_2,
                batch_size=k,
                log_p_s1_p1=log_p1,
                sample_1=s1,
            )
        unnormalized_log_p = unnormalized_model(samples)
        s = unnormalized_log_p - log_q
        return torch.logsumexp(s, dim=0) - np.log(k)

    @staticmethod
    def elbo(proposal: Sampler, unnormalized_model: ScorerBase, batch_size: int):
        log_q, samples = proposal.sample(batch_size=batch_size, to_evaluate=None)
        unnormalized_log_p = unnormalized_model(samples)
        return (unnormalized_log_p - log_q).mean(dim=0)

    @staticmethod
    def normalized_importance_sampling(
        proposal: Sampler,
        unnormalized_model,
        batch_size: int,
        query_args: Optional[Dict] = None,
        k: int = 1,
    ):
        proposal.set_k(k)
        importance_weights, samples, _ = Estimators.get_importance_weights(
            batch_size * k, proposal, unnormalized_model, query_args=query_args
        )

        normalized_weights = torch.log_softmax(importance_weights.reshape(-1, k), dim=1)
        return normalized_weights, samples.reshape(batch_size, k, samples.shape[1])

    @staticmethod
    def chi_upperbound(
        proposal: Sampler,
        unnormalized_model: ScorerBase,
        log_a: torch.Tensor,
        batch_size: int,
        detach_p: bool = False,
        threshold: float = 11.0,
    ):
        importance_weights, _, _ = Estimators.get_importance_weights(
            batch_size, proposal, unnormalized_model, detach_p=detach_p
        )

        log_expectation_squared_tilde_p_over_q = torch.logsumexp(
            importance_weights * 2, dim=0
        ) - np.log(float(batch_size))
        max_to_be_exped = torch.max(log_a + log_expectation_squared_tilde_p_over_q)
        if max_to_be_exped.item() > threshold:
            effective_log_a = log_a - torch.max(
                log_a + log_expectation_squared_tilde_p_over_q
            )
        else:
            effective_log_a = log_a
        to_return = 0.5 * (
            torch.exp(effective_log_a + log_expectation_squared_tilde_p_over_q)
            - effective_log_a
            - 1
        )

        return to_return

    @staticmethod
    def log_z_upperbound(
        proposal: Sampler,
        unnormalized_model: ScorerBase,
        log_a: torch.Tensor,
        batch_size: int,
        detach_p: bool = False,
        threshold: float = 11.0,
    ):
        importance_weights, _, _ = Estimators.get_importance_weights(
            batch_size, proposal, unnormalized_model, detach_p=detach_p
        )

        log_expectation_squared_tilde_p_over_q = torch.logsumexp(
            importance_weights, dim=0
        ) - np.log(float(batch_size))
        max_to_be_exped = torch.max(log_a + log_expectation_squared_tilde_p_over_q)
        if max_to_be_exped.item() > threshold:
            effective_log_a = log_a - torch.max(
                log_a + log_expectation_squared_tilde_p_over_q
            )
        else:
            effective_log_a = log_a
        to_return = (
            torch.exp(effective_log_a + log_expectation_squared_tilde_p_over_q)
            - effective_log_a
            - 1
        )

        return to_return

    @staticmethod
    def get_importance_weights(
        batch_size,
        proposal: Sampler,
        unnormalized_model,
        detach_p: bool = False,
        detach_q: bool = True,
        temperature: float = 1.0,
        query_args: Optional[Dict] = None,
    ):
        log_q, samples = proposal.sample(
            batch_size=batch_size,
            to_evaluate=None,
            temperature=temperature,
            query_args=query_args,
        )
        unnormalized_log_p = unnormalized_model(samples)
        log_q_d = log_q.detach() if detach_q else log_q
        unnormalized_log_p_d = (
            unnormalized_log_p.detach() if detach_p else unnormalized_log_p
        )
        importance_weights = unnormalized_log_p_d - log_q_d
        return importance_weights, samples, log_q

    @staticmethod
    def chi_upperbound_nonlinearized(
        proposal: Sampler, unnormalized_model: ScorerBase, batch_size: int
    ):
        importance_weights, _, _ = Estimators.get_importance_weights(
            batch_size, proposal, unnormalized_model
        )

        log_expectation_squared_tilde_p_over_q = torch.logsumexp(
            importance_weights * 2, dim=0
        ) - np.log(float(batch_size))
        to_return = 0.5 * log_expectation_squared_tilde_p_over_q

        return to_return

    @staticmethod
    def log_z_nonlinearized(
        proposal: Sampler,
        unnormalized_model: ScorerBase,
        batch_size: int,
        temperature: float = 1.0,
        loop=1,
    ):
        to_concat = []
        for _ in range(loop):
            importance_weights, _, _ = Estimators.get_importance_weights(
                batch_size, proposal, unnormalized_model, temperature=temperature
            )
            to_concat.append(importance_weights)
        concatenated = torch.cat(to_concat, dim=0)
        log_expectation_squared_tilde_p_over_q = torch.logsumexp(
            concatenated, dim=0
        ) - np.log(float(batch_size) * loop)
        to_return = log_expectation_squared_tilde_p_over_q

        return to_return

    @staticmethod
    def estimate_log_a_chi(p: ScorerBase, proposal: Sampler, batch_size: int):
        with torch.no_grad():
            chi_loss_nonlinearized = Estimators.chi_upperbound_nonlinearized(
                proposal=proposal, unnormalized_model=p, batch_size=batch_size
            )
            initial_guess = (-2 * chi_loss_nonlinearized).item()
        return initial_guess

    @staticmethod
    def estimate_log_a(p: ScorerBase, proposal: Sampler, batch_size: int):
        with torch.no_grad():
            log_z_nonlinearized = Estimators.log_z_nonlinearized(
                proposal=proposal, unnormalized_model=p, batch_size=batch_size
            )
            initial_guess = -log_z_nonlinearized.item()
        return initial_guess

    @staticmethod
    def estimate_offset_kl_q_p(
        proposal: Sampler,
        unnormalized_model: ScorerBase,
        batch_size: int,
        q_is_hat_p: bool = False,
    ):
        with torch.no_grad():
            if q_is_hat_p:
                log_q, samples = proposal.sample(
                    batch_size=batch_size, to_evaluate=None
                )
                tuple_of_samples = unnormalized_model.to_list_of_tuples(samples)
                unnormalized_log_p = unnormalized_model(samples)
                w = (unnormalized_log_p - log_q).log_softmax(dim=0)

                from collections import defaultdict

                empirical_dist = defaultdict(list)
                empirical_dist_p = dict()
                for w_of_sample, sam, p_of_sample in zip(
                    w.tolist(), tuple_of_samples, unnormalized_log_p.tolist()
                ):
                    empirical_dist[sam].append(w_of_sample)
                    empirical_dist_p[sam] = p_of_sample

                normalized_empirical_dist = dict()
                for k in empirical_dist:
                    normalized_empirical_dist[k] = empirical_dist[k][0] + np.log(
                        float(len(empirical_dist[k]))
                    )

                entropy_term = 0.0
                cross_entropy_term = 0.0

                for k in normalized_empirical_dist:
                    entropy_term += (
                        np.exp(normalized_empirical_dist[k])
                        * normalized_empirical_dist[k]
                    )
                    cross_entropy_term += (
                        np.exp(normalized_empirical_dist[k]) * empirical_dist_p[k]
                    )
            else:
                log_q, samples = proposal.sample(
                    batch_size=batch_size, to_evaluate=None
                )
                unnormalized_log_p = unnormalized_model(samples)
                entropy_term = log_q.mean(dim=0)
                cross_entropy_term = unnormalized_log_p.mean(dim=0)
            return entropy_term - cross_entropy_term


# class Losses:
#     @staticmethod
#     def nce_loss_given_noise(
#         unnormalized_model: ScorerBase,
#         positive_input: torch.Tensor,
#         noise_seqs: torch.Tensor,
#         noise_probs: torch.Tensor,
#         positive_probs: torch.Tensor,
#     ):
#         """
#         we assume each positive input in the minibatch will get the same noise seqs for now
#         :param unnormalized_model:
#         :param positive_input:
#         :param noise_seqs:
#         :param noise_probs:
#         :param positive_probs:
#         :return:
#         """
#         assert len(noise_seqs.shape) == 2
#         k = noise_seqs.shape[0]

#         assert len(noise_probs.shape) == 1
#         assert noise_probs.shape[0] == k

#         s_bar_neg = unnormalized_model(noise_seqs) - noise_probs
#         s_bar_neg_sum = torch.logsumexp(s_bar_neg, dim=0).reshape((1, 1))

#         batch_size = positive_input.shape[0]
#         s_bar_neg_sum_expanded = s_bar_neg_sum.expand(batch_size, -1)

#         assert len(positive_input.shape) == 2
#         s_bar_pos = unnormalized_model(positive_input) - positive_probs

#         classification_prob = s_bar_pos - torch.logsumexp(
#             torch.cat((s_bar_pos[:, np.newaxis], s_bar_neg_sum_expanded), dim=1), dim=1
#         )
#         return -classification_prob.mean()

#     @staticmethod
#     def l2_pretrain(
#         proposal: Sampler,
#         unnormalized_model: ScorerBase,
#         k: int,
#         exp: Optional[bool] = False,
#     ):
#         with torch.no_grad():
#             log_q, x = proposal.sample(k, temperature=1.0)
#         log_tilde_p = unnormalized_model(x)
#         if exp:
#             return torch.logsumexp(
#                 torch.log((log_q - log_tilde_p) ** 2) - log_q, dim=0
#             ) - np.log(k)
#             # return torch.mean(torch.abs(log_q - log_tilde_p), dim=0)
#         else:
#             return torch.mean((log_q - log_tilde_p) ** 2, dim=0)

#     @staticmethod
#     def s_bar(
#         sequences,
#         log_pn,
#         alpha,
#         unnormalized_model: ScorerBase,
#         hx: Optional[torch.Tensor] = None,
#     ):
#         if hx is not None:
#             unnormalized_score = unnormalized_model(sequences, ext=hx)
#         else:
#             unnormalized_score = unnormalized_model(sequences)
#         if alpha == 1:
#             return unnormalized_score, unnormalized_score
#         return (alpha - 1.0) * log_pn + unnormalized_score, unnormalized_score

#     @staticmethod
#     def get_sbar_positive(
#         proposal: Sampler,
#         positive_input: torch.Tensor,
#         unnormalized_model: ScorerBase,
#         temp: float = 1.0,
#         alpha: float = 0.0,
#         use_noise_embeddings: bool = False,
#     ) -> torch.Tensor:
#         with torch.no_grad():
#             if proposal.self_normalized:
#                 pos_log_pn, pos_hx = proposal.model(
#                     positive_input, temp=temp, return_states=True
#                 )
#             else:
#                 pos_log_pn, pos_hx = proposal.stateful_sample(
#                     batch_size=positive_input.shape[0],
#                     to_evaluate=positive_input,
#                     hx=None,
#                     inp=None,
#                     no_zs=True,
#                     temperature=temp,
#                     only_return_final_state=False,
#                 )
#                 from modules.scorers import StaticRNNScorer

#                 if isinstance(proposal.real_model, StaticRNNScorer):
#                     r: StaticRNNScorer = proposal.real_model
#                     pos_hx = r.get_last_layer(pos_hx)
#             if use_noise_embeddings:
#                 s_bar_pos, _ = Losses.s_bar(
#                     positive_input, pos_log_pn, alpha, unnormalized_model, hx=pos_hx
#                 )
#             else:
#                 s_bar_pos, _ = Losses.s_bar(
#                     positive_input, pos_log_pn, alpha, unnormalized_model
#                 )
#         return s_bar_pos, pos_log_pn

#     @staticmethod
#     def get_sbar_noise(
#         proposal: Sampler,
#         unnormalized_model: ScorerBase,
#         k: int = 16,
#         temp: float = 1.0,
#         alpha: float = 0.0,
#         use_noise_embeddings: bool = False,
#         reverse: bool = False,
#     ) -> torch.Tensor:
#         with torch.no_grad():
#             neg_sequences: torch.Tensor
#             neg_log_pn, neg_sequences, neg_hx = proposal.stateful_sample(
#                 k,
#                 to_evaluate=None,
#                 hx=None,
#                 inp=None,
#                 no_zs=True,
#                 temperature=temp,
#                 only_return_final_state=False,
#             )
#             if reverse:
#                 assert not use_noise_embeddings
#                 neg_sequences_np = neg_sequences.cpu().numpy()
#                 num_pads = np.sum(
#                     (neg_sequences_np == proposal.real_model.__pad__),
#                     axis=1,
#                     dtype=np.int,
#                 )
#                 neg_sequences_list = [
#                     list(
#                         reversed(
#                             [y for y in x if y != proposal.real_model.__pad__][:-1]
#                         )
#                     )
#                     + [proposal.real_model.__eos__]
#                     + int(num_pads[idx]) * [proposal.real_model.__pad__]
#                     for idx, x in enumerate(neg_sequences_np)
#                 ]
#                 new_neg_sequences = torch.tensor(
#                     neg_sequences_list,
#                     dtype=neg_sequences.dtype,
#                     device=neg_sequences.device,
#                 )
#                 neg_sequences = new_neg_sequences

#             from modules.scorers import StaticRNNScorer

#             if isinstance(proposal.real_model, StaticRNNScorer):
#                 r: StaticRNNScorer = proposal.real_model
#                 neg_hx = r.get_last_layer(neg_hx)
#             if use_noise_embeddings:
#                 s_bar_neg, _ = Losses.s_bar(
#                     neg_sequences, neg_log_pn, alpha, unnormalized_model, hx=neg_hx
#                 )
#             else:
#                 s_bar_neg, _ = Losses.s_bar(
#                     neg_sequences, neg_log_pn, alpha, unnormalized_model
#                 )
#         return s_bar_neg, neg_sequences

#     @staticmethod
#     def nce_loss(
#         proposal: Sampler,
#         unnormalized_model: ScorerBase,
#         positive_input: torch.Tensor,
#         k: int = 16,
#         temp: float = 1.0,
#         single_noise: bool = False,
#         alpha: float = 0.0,
#         use_noise_embeddings: bool = False,
#         return_mean_abs_s: bool = False,
#     ):
#         batch_size = positive_input.shape[0]

#         if single_noise:
#             num_neg_samples = k
#         else:
#             num_neg_samples = k * batch_size
#         with torch.no_grad():
#             neg_log_pn, neg_sequences, neg_hx = proposal.stateful_sample(
#                 num_neg_samples,
#                 to_evaluate=None,
#                 hx=None,
#                 inp=None,
#                 no_zs=True,
#                 temperature=temp,
#                 only_return_final_state=False,
#             )
#             from modules.scorers import StaticRNNScorer

#             if isinstance(proposal.real_model, StaticRNNScorer):
#                 r: StaticRNNScorer = proposal.real_model
#                 neg_hx = r.get_last_layer(neg_hx)
#         if use_noise_embeddings:
#             s_bar_neg, _ = Losses.s_bar(
#                 neg_sequences, neg_log_pn, alpha, unnormalized_model, hx=neg_hx
#             )
#         else:
#             s_bar_neg, _ = Losses.s_bar(
#                 neg_sequences, neg_log_pn, alpha, unnormalized_model
#             )

#         if single_noise:
#             s_bar_neg_sum = (
#                 torch.logsumexp(s_bar_neg, dim=0).reshape((1, 1)).expand(batch_size, -1)
#             )
#         else:
#             s_bar_neg = s_bar_neg.reshape((batch_size, k))
#             s_bar_neg_sum = torch.logsumexp(s_bar_neg, dim=1, keepdim=True)
#         with torch.no_grad():
#             if proposal.self_normalized:
#                 pos_log_pn, pos_hx = proposal.model(
#                     positive_input, temp=temp, return_states=True
#                 )
#             else:
#                 pos_log_pn, pos_hx = proposal.stateful_sample(
#                     batch_size=batch_size,
#                     to_evaluate=positive_input,
#                     hx=None,
#                     inp=None,
#                     no_zs=True,
#                     temperature=temp,
#                     only_return_final_state=False,
#                 )
#                 from modules.scorers import StaticRNNScorer

#                 if isinstance(proposal.real_model, StaticRNNScorer):
#                     r: StaticRNNScorer = proposal.real_model
#                     pos_hx = r.get_last_layer(pos_hx)
#             assert pos_log_pn.shape[0] == batch_size
#         s_bar_pos: torch.Tensor
#         s: torch.Tensor
#         if use_noise_embeddings:
#             s_bar_pos, s = Losses.s_bar(
#                 positive_input, pos_log_pn, alpha, unnormalized_model, hx=pos_hx
#             )
#         else:
#             s_bar_pos, s = Losses.s_bar(
#                 positive_input, pos_log_pn, alpha, unnormalized_model
#             )

#         s_bar_pos_and_s_bar_neg: torch.Tensor = torch.cat(
#             (s_bar_pos[:, np.newaxis], s_bar_neg_sum), dim=1
#         )
#         assert s_bar_pos_and_s_bar_neg.shape[0] == batch_size
#         assert s_bar_pos_and_s_bar_neg.shape[1] == 2
#         z = torch.logsumexp(s_bar_pos_and_s_bar_neg, dim=1)
#         to_return: torch.Tensor = -(s_bar_pos - z).mean()

#         if return_mean_abs_s is not None:
#             return to_return, torch.mean(torch.abs(s), dim=0)
#         return to_return

#     @staticmethod
#     def get_distribution_over_masked(
#         to_mask: torch.Tensor,
#         masked_index: torch.Tensor,
#         bert_scorer: MaskedLM,
#         mask_symbol,
#     ):
#         """
#         returns distributions over masked words (roberta bpe'd) at positions masked_index
#         :param to_mask: batch_size x max_length
#         :param masked_index: batch_size
#         :param bert_scorer
#         :param mask_symbol
#         :return: batch_size x vocab_size
#         """
#         batch_size = to_mask.shape[0]
#         one_hot_mask = torch.zeros_like(
#             to_mask, dtype=torch.bool, device=to_mask.device
#         )
#         one_hot_mask[
#             torch.arange(batch_size, device=to_mask.device), masked_index
#         ] = True
#         masked = to_mask.masked_fill(one_hot_mask, mask_symbol)

#         features = bert_scorer(masked)
#         logits = features[torch.arange(batch_size, device=to_mask.device), masked_index]
#         # no need to normalize: will do so when we sample
#         return logits

#     @staticmethod
#     def pretrain_mlm_loss(inp: torch.Tensor, masked_lm: PMaskedLM, mask_symbol):
#         """

#         :param inp:
#         :param masked_lm:
#         :param mask_symbol:
#         :return:
#         """
#         from torch.nn.parallel import DistributedDataParallel

#         if isinstance(masked_lm, DistributedDataParallel):
#             real_model: PaddedScorer = masked_lm.module.model
#         else:
#             real_model: PaddedScorer = masked_lm.model

#         batch_size = inp.shape[0]

#         with torch.no_grad():
#             positions = real_model.sample_positions(inp)  # (batch_size,)
#         dist_over_masked = Losses.get_distribution_over_masked(
#             inp, positions, bert_scorer=masked_lm, mask_symbol=mask_symbol
#         )
#         noise_dist = Categorical(logits=dist_over_masked)

#         true_answer = inp[torch.arange(batch_size), positions].reshape(1, batch_size)
#         prob = noise_dist.log_prob(true_answer).flatten()
#         return -prob.mean(dim=0)

#     @staticmethod
#     def get_nce_mple_prob(
#         inp: torch.Tensor,
#         model: PaddedScorer,
#         bert: MaskedLM,
#         mask_symbol,
#         k,
#         temp,
#         alpha: float,
#         noise: Optional[Sampler],
#         flm_temp: float,
#         use_noise_embeddings: bool,
#     ):
#         assert isinstance(noise, Sampler)
#         # TODO fall back gracefully when there's no noise, or print readable error messages

#         real_model: PaddedScorer = get_model(model)
#         assert bert is not None, "unigram noise not supported yet!"
#         batch_size = inp.shape[0]

#         # FIXME should refactor the following into a method of either Sampler or PaddedScorer
#         with torch.no_grad():
#             if alpha > 0.0:
#                 flm_noise_pos, pos_hx = noise.stateful_sample(
#                     batch_size=batch_size,
#                     to_evaluate=inp,
#                     no_zs=True,
#                     temperature=flm_temp,
#                     only_return_final_state=False,
#                 )
#                 from modules.scorers import StaticRNNScorer

#                 if isinstance(noise.real_model, StaticRNNScorer):
#                     r: StaticRNNScorer = noise.real_model
#                     pos_hx = r.get_last_layer(pos_hx)
#                 alpha_flm_noise_pos = alpha * flm_noise_pos
#             else:
#                 alpha_flm_noise_pos = 0.0
#                 pos_hx = None
#         if use_noise_embeddings:
#             truth_log_probs = (model(inp, ext=pos_hx) + alpha_flm_noise_pos).reshape(
#                 (batch_size, 1)
#             )
#         else:
#             truth_log_probs = (model(inp) + alpha_flm_noise_pos).reshape(
#                 (batch_size, 1)
#             )
#         with torch.no_grad():
#             positions = real_model.sample_positions(inp)  # (batch_size,)
#             dist_over_masked = Losses.get_distribution_over_masked(
#                 inp, positions, bert_scorer=bert, mask_symbol=mask_symbol
#             )
#             dist_over_masked = dist_over_masked / temp
#             noise_dist = Categorical(logits=dist_over_masked)
#             sampled_noises_original = noise_dist.sample((k,))  # (k, batch_size)
#             sampled_noises = torch.t(sampled_noises_original)  # (batch_size, k)

#         pn_true: torch.Tensor = noise_dist.log_prob(
#             inp[torch.arange(batch_size), positions].reshape(1, batch_size)
#         )  # (1, batch_size)
#         pn_noise: torch.Tensor = noise_dist.log_prob(
#             sampled_noises_original
#         )  # (k, batch_size)

#         # permute pn_true and pn_noise back to (batch_size, k)
#         pn_true = pn_true.t()
#         pn_noise = pn_noise.t()

#         # now construct full noised-up sequences
#         # TODO had to clone for assignment via adv-indexing to work. no clue why...
#         inp_expanded = (
#             inp[:, :, np.newaxis].expand(-1, -1, k).clone()
#         )  # (batch_size, max_length, k)
#         inp_expanded[
#             torch.arange(batch_size, device=positions.device), positions
#         ] = sampled_noises
#         inp_expanded_swapped = inp_expanded.permute(
#             0, 2, 1
#         )  # batch_size, k, max_length
#         # TODO hook up actual model here

#         noise_seqs = inp_expanded_swapped.reshape((-1, inp_expanded_swapped.shape[-1]))

#         with torch.no_grad():
#             # don't multiply in alpha if alpha is zero (this is to avoid nan problems)
#             if alpha > 0.0:
#                 flm_noise_neg, neg_hx = noise.stateful_sample(
#                     batch_size=noise_seqs.shape[0],
#                     to_evaluate=noise_seqs,
#                     no_zs=True,
#                     temperature=flm_temp,
#                     only_return_final_state=False,
#                 )
#                 from modules.scorers import StaticRNNScorer

#                 if isinstance(noise.real_model, StaticRNNScorer):
#                     r: StaticRNNScorer = noise.real_model
#                     neg_hx = r.get_last_layer(neg_hx)
#                 alpha_flm_noise_neg = alpha * flm_noise_neg
#             else:
#                 alpha_flm_noise_neg = 0.0
#                 neg_hx = None
#         if use_noise_embeddings:
#             noise_log_probs = (
#                 model(noise_seqs, ext=neg_hx) + alpha_flm_noise_neg
#             ).reshape((batch_size, k))
#         else:
#             noise_log_probs = (
#                 model(noise_seqs, ext=neg_hx) + alpha_flm_noise_neg
#             ).reshape((batch_size, k))

#         # (batch_size, k)

#         s_bar_x = truth_log_probs - pn_true  # (batch_size, 1)
#         s_bar_x_1_to_k = noise_log_probs - pn_noise  # (batch_size, k)

#         denominator = torch.logsumexp(
#             torch.cat((s_bar_x, s_bar_x_1_to_k), dim=1), dim=1
#         )
#         reranked_prob = s_bar_x.reshape((batch_size,)) - denominator
#         return reranked_prob

#     @staticmethod
#     def nce_pseudolikelihood_loss(
#         unnormalized_model: ScorerBase,
#         bert: MaskedLM,
#         vocab: Vocabulary,
#         positive_input: torch.Tensor,
#         k: int = 16,
#         temp: float = 1.0,
#         alpha: float = 0.0,
#         flm_temp: float = 1.0,
#         noise: Optional[Sampler] = None,
#         use_noise_embeddings: bool = False,
#     ):
#         """

#         :param unnormalized_model:
#         :param bert:
#         :param vocab:
#         :param positive_input:
#         :param k:
#         :param temp:
#         :param alpha:
#         :param flm_temp:
#         :param noise:
#         :param use_noise_embeddings:
#         :return:
#         """
#         mask_idx = vocab.get_token_index("<mask>", namespace="tags")
#         return -Losses.get_nce_mple_prob(
#             positive_input,
#             model=unnormalized_model,
#             bert=bert,
#             mask_symbol=mask_idx,
#             k=k,
#             temp=temp,
#             alpha=alpha,
#             flm_temp=flm_temp,
#             noise=noise,
#             use_noise_embeddings=use_noise_embeddings,
#         ).mean()

#     @staticmethod
#     def cross_entropy_hat_p_q(
#         proposal: Sampler, unnormalized_model: ScorerBase, batch_size: int
#     ):
#         w, x = Estimators.normalized_importance_sampling(
#             proposal, unnormalized_model, batch_size
#         )
#         log_q_x, _ = proposal.sample(batch_size=batch_size, to_evaluate=x)
#         return (-torch.exp(w) * log_q_x).sum()

#     @staticmethod
#     def cross_entropy_q_tilde_p(
#         proposal: Sampler,
#         unnormalized_model: ScorerBase,
#         batch_size: int,
#         baseline: float = 0.0,
#     ):
#         log_q, samples = proposal.sample(batch_size=batch_size, to_evaluate=None)
#         with torch.no_grad():
#             unnormalized_log_p = unnormalized_model(samples)
#         importance_weights = unnormalized_log_p - log_q
#         return ((importance_weights * importance_weights) - baseline).sum(), (
#             importance_weights * importance_weights
#         ).sum()

#     @staticmethod
#     def proposal_policy_gradient_loss(
#         proposal: Sampler,
#         baseline: torch.Tensor,
#         sequence: torch.Tensor,
#         reward: torch.Tensor,
#     ):
#         """
#         sequence has to come from the distribution of proposal to make this correct!

#         :param proposal:
#         :param baseline:
#         :param sequence:
#         :param reward:
#         :return:
#         """
#         log_q, zs = proposal.sample(batch_size=sequence.shape[0], to_evaluate=sequence)
#         return -((reward - baseline) * log_q)

#     @staticmethod
#     def sandwich_loss(
#         proposal: Sampler,
#         unnormalized_model: ScorerBase,
#         log_a: torch.Tensor,
#         batch_size: int,
#         threshold: float = 11.0,
#     ):
#         importance_weights, _, log_q = Estimators.get_importance_weights(
#             batch_size, proposal, unnormalized_model, detach_p=True, detach_q=True
#         )
#         # per_sample_p_over_q_squared = importance_weights * 2
#         # max_to_be_exped = torch.max(log_a + per_sample_p_over_q_squared)
#         # if max_to_be_exped.item() > threshold:
#         #     effective_log_a = log_a - max_to_be_exped
#         # else:
#         #     effective_log_a = log_a
#         # upper_bound = 0.5 * ( torch.exp(effective_log_a + per_sample_p_over_q_squared) - effective_log_a - 1 )
#         lower_bound = importance_weights
#         upper_bound = 0.0
#         bound_diff = upper_bound - lower_bound
#         # bound_diff = - lower_bound
#         return bound_diff, log_q
