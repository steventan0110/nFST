from abc import ABCMeta, abstractmethod
import torch
from typing import Optional, Tuple, Dict, AnyStr, Any, Type, List
import numpy as np
import torch
from torch import nn
from torch.nn import Embedding, GRUCell, Linear, Dropout, GRU, LSTM, RNNBase
from src.util.preprocess_util import Vocab
import src.modules.model as model
from src.modules.queries import EncodedQuery, QueryBase
from src.modules.nros import NRO
from src.util.preprocess_util import Utils


# TODO: refactor this file


class TiedScorer(nn.Module):
    def __init__(self, embeddings, vocab_size):
        super(TiedScorer, self).__init__()
        self.embeddings: nn.Embedding = embeddings
        self.vocab_size = vocab_size
        b: torch.Tensor = torch.zeros(
            vocab_size,
        )
        b.data.uniform_(-0.1, 0.1)
        self.bias = nn.Parameter(b)

    def forward(self, to_score: torch.Tensor):
        return torch.matmul(to_score, self.embeddings.weight.t()) + self.bias


class ScorerBase(torch.nn.Module, metaclass=ABCMeta):
    def forward(self, sequence: torch.Tensor, **kwargs):
        """
        implementation is in evaluate_seq
        :param sequence:
        :return:
        """
        return self.evaluate_seq(sequence, **kwargs)

    @abstractmethod
    def evaluate_seq(self, sequence: torch.Tensor, **kwargs):
        """

        :param sequence: of shape (batch_size x max_length). *not* bos_padded but each sequence is eos_padded.
        :return:
        """
        pass

    def __init__(self, pad, bos, eos):
        super(ScorerBase, self).__init__()
        self.__pad__ = pad
        self.__bos__ = bos
        self.__eos__ = eos


class PaddedScorer(ScorerBase):
    def mask_invalid_helper(
        self, has_to_end, inp, bos_mask, eos_mask, pad_mask, no_pad_mask
    ):
        vocab_size: int = self.vocab_size
        batch_size: int = inp.shape[0]
        final_mask = bos_mask[np.newaxis, :].expand((batch_size, -1))
        # is the previous emission EOS or PAD?
        eos_or_pad = inp == self.__eos__
        eos_or_pad += inp == self.__pad__
        # expanding pad mask to batch_size x vocab_size
        expanded_pad_mask = pad_mask[np.newaxis, :].expand((batch_size, -1))
        expanded_no_pad_mask = no_pad_mask[np.newaxis, :].expand((batch_size, -1))
        # if the previous emission is eos or pad, the only q_valid choice is pad
        is_eos_or_pad = eos_or_pad[:, np.newaxis].expand((-1, vocab_size))
        zero_scores = torch.zeros((batch_size, vocab_size), device=inp.device)
        must_be_pad_mask = torch.where(
            is_eos_or_pad, expanded_pad_mask, expanded_no_pad_mask
        )
        final_mask = must_be_pad_mask + final_mask
        if has_to_end:
            expanded_eos_mask = eos_mask[np.newaxis, :].expand((batch_size, -1))
            final_mask = (
                torch.where(is_eos_or_pad, zero_scores, expanded_eos_mask) + final_mask
            )
        return final_mask

    @property
    def self_normalized(self):
        return False

    def slow_but_correct_batched_mask_out_invalid(
        self, inp: torch.Tensor, max_length: Optional[int] = None
    ):
        """

        :param inp: batch_size x max_inp_length
        :return: batch_size x max_inp_length x vocab_size
        """
        assert len(inp.shape) == 2
        batch_size = inp.shape[0]
        max_inp_length = inp.shape[1]
        device = inp.device
        first = torch.zeros(
            (batch_size, 1, self.vocab_size),
            dtype=torch.get_default_dtype(),
            device=device,
        )
        first[:, 0, self.__bos__] = float("-inf")
        first[:, 0, self.__pad__] = float("-inf")

        bos_mask = self.bos_mask.to(device)
        eos_mask = self.eos_mask.to(device)
        pad_mask = self.pad_mask.to(device)
        no_pad_mask = self.no_pad_mask.to(device)

        to_return = [first]
        # note that the masks are generated *before* reading in the inputs
        # therefore we won't be generating a mask after we read in the last input
        # this ensures our mask to have shape batch_size, max_inp_length, vocab_size
        for idx in range(max_inp_length - 1):
            if max_length is None:
                flag = False
            else:
                flag = idx + 1 > max_length
            sliced = inp[:, idx]
            new_mask = self.mask_invalid_helper(
                flag,
                sliced,
                bos_mask=bos_mask,
                eos_mask=eos_mask,
                pad_mask=pad_mask,
                no_pad_mask=no_pad_mask,
            )
            to_return.append(new_mask[:, np.newaxis, :])

        return torch.cat(to_return, dim=1)

    def sample_positions(self, to_sample: torch.Tensor):
        """

        :param to_sample: batch_size x max_length, we assume the input is shifted (doesn't start with BOS)
        :return: batch_size
        """
        from torch.distributions import Categorical

        assert len(to_sample.shape) == 2
        length_mask = to_sample != self.__pad__
        length_mask = length_mask & (to_sample != self.__eos__)
        probs = torch.ones_like(length_mask, dtype=torch.float) * length_mask.to(
            torch.get_default_dtype()
        )
        cat = Categorical(probs=probs)
        sampled = cat.sample()
        return sampled

    @property
    def has_seq(self):
        return False

    def get_seqs(self, sequence: torch.Tensor, **kwargs):
        """

        :param sequence: batch_size, seq_length, vocab_size
        :param kwargs:
        :return: batch_size, seq_length, vocab_size
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_seq(
        self, sequence: torch.Tensor, return_states: bool = False, **kwargs
    ):
        pass

    @abstractmethod
    def attune(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        :param hidden_states: batch_size x hid_dim
        :return: batch_size
        """
        # FIXME should make sure whether this method makes sense for undirectional sequence models
        pass

    def pad_masking(self, scores):
        from numpy import newaxis

        return scores * self.pad_zero_mask[newaxis, :].to(scores.device).expand(
            scores.shape[0], -1
        )

    def pad_masking_3d(self, scores):
        return scores * self.pad_zero_mask.view(1, 1, -1).to(scores.device).expand(
            scores.shape[0], scores.shape[1], -1
        )

    def hidden_state_masking(self, h, inp, old_h):
        from numpy import newaxis

        inp_is_pad = (inp == self.__pad__).to(h.device)
        return torch.where(inp_is_pad[:, newaxis].expand(-1, h.shape[1]), old_h, h)

    def get_init_states(self, batch_size, device):
        metadata = {"length": 1}
        return (
            self.h0.to(device).expand(batch_size, -1),
            self.inp0.to(device).expand(batch_size),
            metadata,
        )

    def activation(self, scores):
        if self.__activation__ is not None:
            return self.__activation__(scores)
        return scores

    def get_seq_mask(self, sequences: torch.Tensor):
        mask = (sequences != self.__pad__).to(sequences.device)
        return mask

    def __init__(
        self, hid_dim, vocab_size, activation=None, pad=None, bos=None, eos=None
    ):
        assert pad is not None and bos is not None and eos is not None
        super(PaddedScorer, self).__init__(pad=pad, bos=bos, eos=eos)

        assert vocab_size > 3
        self.hid_dim = hid_dim
        self.vocab_size = vocab_size

        self.pad_zero_mask = torch.ones((vocab_size,))
        self.pad_zero_mask[self.__pad__] = 0

        self.inp0 = torch.empty((1,), dtype=torch.long)
        self.inp0.fill_(self.__bos__)
        self.h0 = torch.zeros((1, hid_dim))

        self.bos_mask = torch.zeros((self.vocab_size,))
        self.bos_mask[self.__bos__] = float("-inf")

        self.pad_mask = torch.empty(
            (self.vocab_size,),
        )
        self.pad_mask.fill_(float("-inf"))
        self.pad_mask[self.__pad__] = 0

        self.no_pad_mask = torch.zeros(
            (self.vocab_size,),
        )
        self.no_pad_mask[self.__pad__] = float("-inf")

        self.eos_mask = torch.empty((self.vocab_size,))
        self.eos_mask.fill_(float("-inf"))
        self.eos_mask[self.__eos__] = 0

        self.__activation__ = activation

        return

    def get_output_dim(self):
        return self.hid_dim


class ContextLayer(torch.nn.Module):
    def __init__(self, hid_dim: int, context_dim: int, drop: float = 0.5) -> None:
        super().__init__()
        self.hid_dim = hid_dim
        self.context_dim = context_dim
        self.double_layer = torch.nn.Linear(
            self.context_dim + self.hid_dim, self.hid_dim
        )
        self.drop = Dropout(p=drop)

    def forward(self, queried, h):
        return torch.tanh(self.drop(self.double_layer(torch.cat((queried, h), dim=1))))


class LeftToRightScorer(PaddedScorer):
    def batched_mask_out_invalid(self, inp: torch.Tensor):
        """

        :param inp: batch_size, max_length
        :return: batch_size, max_length, vocab_size
        """
        assert len(inp.shape) == 2
        batch_size = inp.shape[0]
        max_inp_length = inp.shape[1]
        to_return = torch.zeros(
            (batch_size, max_inp_length, self.vocab_size), device=inp.device
        )
        # BOS is always disallowed
        to_return[:, :, self.__bos__] = float("-inf")

        # by default we disallow PAD too
        to_return[:, :, self.__pad__] = float("-inf")

        # when we see pads that means only pads are allowed at that position
        pad_only = (inp == self.__pad__)[:, :, np.newaxis].expand(
            -1, -1, self.vocab_size
        )
        pad_only_effect = (
            self.pad_mask.to(inp.device)
            .reshape(1, 1, -1)
            .expand(batch_size, max_inp_length, -1)
        )
        to_return.masked_scatter_(pad_only, pad_only_effect)

        # beyond the max length, we can only generate either PAD or EOS
        if max_inp_length > self.max_length:
            beyond_max_length_effect = (
                self.eos_or_pad_mask.to(to_return.device)
                .reshape(1, 1, -1)
                .expand(batch_size, max_inp_length - self.max_length, -1)
            )
            to_return[:, self.max_length :, :] += beyond_max_length_effect
        return to_return

    def mask_out_invalid(
        self, inp: torch.Tensor, metadata: Dict[AnyStr, Any]
    ) -> torch.Tensor:
        """

        :param inp:
        :param metadata:
        :return:
        """
        device = inp.device

        # override for more sophisticated state tracking
        assert "length" in metadata
        has_to_end: bool = metadata["length"] > self.max_length

        bos_mask = self.bos_mask.to(device)
        eos_mask = self.eos_mask.to(device)
        pad_mask = self.pad_mask.to(device)
        no_pad_mask = self.no_pad_mask.to(device)

        final_mask = self.mask_invalid_helper(
            has_to_end, inp, bos_mask, eos_mask, pad_mask, no_pad_mask
        )

        return final_mask

    def left_to_right_score(
        self,
        left_h,
        left_inp,
        metadata: Dict[AnyStr, Any],
        locally_normalize: bool = False,
        beta=None,
    ):
        assert metadata["length"] > 0
        assert "not_cleared" not in metadata
        next_h, scores, updated_metadata = self.actual_left_to_right_score(
            left_h, left_inp, metadata, beta
        )

        updated_metadata["length"] = metadata["length"] + 1
        updated_metadata["not_cleared"] = True
        scores_mask = self.mask_out_invalid(left_inp, metadata=metadata)
        through_activation = self.activation(self.pad_masking(scores) + scores_mask)
        if locally_normalize:
            final_score = torch.log_softmax(through_activation, dim=1)
        else:
            final_score = through_activation
        return (
            self.hidden_state_masking(next_h, left_inp, left_h),
            final_score,
            updated_metadata,
        )

    def metadata_callback(self, h, just_sampled_inp, metadata):
        metadata.pop("not_cleared")
        return metadata

    @abstractmethod
    def actual_left_to_right_score(
        self,
        left_h: torch.Tensor,
        left_inp: torch.Tensor,
        metadata: Dict[AnyStr, Any],
        beta=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[AnyStr, Any]]:
        pass

    def get_all_permutations(self):
        permutations = []
        indices = [
            _
            for _ in range(self.vocab_size)
            if _ != self.__bos__ and _ != self.__eos__ and _ != self.__pad__
        ]
        effective_max_length = self.max_length - 1

        from collections import deque

        prefixes = deque()
        prefixes.append([])

        while len(prefixes) > 0:
            to_process = prefixes.popleft()
            permutations.append(
                to_process
                + [self.__eos__]
                + (effective_max_length - len(to_process)) * [self.__pad__]
            )

            if len(to_process) < effective_max_length:
                for i in indices:
                    to_put_back = list(to_process) + [i]
                    prefixes.append(to_put_back)

        to_return = torch.tensor(permutations, dtype=torch.long)
        return to_return

    def to_list_of_tuples(self, sequences: torch.Tensor):
        sequences_to_list = sequences.tolist()
        to_return = []
        for seq in sequences_to_list:
            to_return.append(tuple([_ for _ in seq if _ != self.__pad__]))

        return to_return

    def evaluate_seq(self, sequence: torch.Tensor, **kwargs):
        """
        please override this method if your scorer can somehow evaluate an entire seq fast
        :param sequence:
        :return:
        """

        dev = sequence.device
        h, inp, metadata = self.get_init_states(sequence.shape[0], dev)

        to_return = []
        for _ in range(sequence.shape[1]):
            h, score, updated_metadata = self.left_to_right_score(
                left_h=h, left_inp=inp, metadata=metadata
            )
            next_up = sequence[:, _]
            metadata = self.metadata_callback(h, next_up, updated_metadata)
            seq_arange = torch.arange(sequence.shape[0], device=dev)
            try:
                selected = score[seq_arange, next_up]
            except Exception as e:
                raise e
            to_return.append(selected)
            inp = next_up
        sum_to_return = torch.stack(to_return, dim=0).sum(dim=0)
        return sum_to_return

    def has_query(self):
        return self.query is not None

    def register_query(
        self, to_encode: torch.Tensor, to_encode_mask: torch.Tensor, **kwargs
    ):
        if not self.has_query():
            assert to_encode_mask is None
            assert to_encode is None
            return
        query: EncodedQuery = self.query
        query.undo_encode()
        query.encode(to_encode, to_encode_mask, **kwargs)

    def attune(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.has_query():
            looked_up_context = self.query(hidden_states, **kwargs)
            attuned = self.context(h=hidden_states, queried=looked_up_context)
        else:
            attuned = hidden_states
        return attuned

    def __init__(
        self,
        hid_dim,
        vocab_size,
        activation=None,
        pad=0,
        bos=1,
        eos=2,
        query: Optional[EncodedQuery] = None,
        context_state_dim: Optional[int] = None,
        max_length: int = 400,
        embeddings: Optional[torch.nn.modules.Embedding] = None,
    ):
        super(LeftToRightScorer, self).__init__(
            hid_dim=hid_dim,
            vocab_size=vocab_size,
            activation=activation,
            pad=pad,
            bos=bos,
            eos=eos,
        )
        self.scorer = None  # torch.nn.Linear(hid_dim, vocab_size)
        if embeddings is not None:
            self.add_module("embeddings", embeddings)
        else:
            self.embeddings = Embedding(vocab_size, hid_dim)
        if context_state_dim is None:
            context_state_dim = hid_dim
        self.query: Optional[QueryBase] = None
        self.max_length = max_length
        if query is not None:
            self.query = query
            self.add_module("query", query)
            self.context = ContextLayer(
                context_dim=query.get_output_dim(), hid_dim=context_state_dim
            )

        self.eos_or_pad_mask = torch.empty(
            (self.vocab_size,),
        )
        self.eos_or_pad_mask.fill_(float("-inf"))
        self.eos_or_pad_mask[self.__eos__] = 0
        self.eos_or_pad_mask[self.__pad__] = 0


class GRUScorer(LeftToRightScorer):
    def __init__(
        self,
        hid_dim,
        vocab_size,
        activation=None,
        pad=0,
        bos=1,
        eos=2,
        dropout=0.5,
        query=None,
        max_length=400,
        embeddings=None,
        tied_embeddings: bool = False,
    ):
        super(GRUScorer, self).__init__(
            hid_dim,
            vocab_size,
            activation=activation,
            pad=pad,
            bos=bos,
            eos=eos,
            query=query,
            max_length=max_length,
            embeddings=embeddings,
        )
        self.unit = GRUCell(hid_dim, hid_dim)
        self.drop = Dropout(dropout)
        self.beta_scorer = torch.nn.Sequential(
            torch.nn.Linear(
                hid_dim,
                hid_dim,
            ),
            torch.nn.GELU(),
            torch.nn.Linear(
                hid_dim,
                self.vocab_size,
            ),
        )
        if tied_embeddings:
            self.scorer = torch.nn.Sequential(
                torch.nn.Linear(
                    2 * hid_dim,
                    hid_dim,
                ),
                torch.nn.GELU(),
                TiedScorer(embeddings, vocab_size),
            )
        else:
            self.scorer = torch.nn.Sequential(
                torch.nn.Linear(
                    2 * hid_dim,
                    hid_dim,
                ),
                torch.nn.GELU(),
                torch.nn.Linear(
                    hid_dim,
                    self.vocab_size,
                ),
            )
        return

    def actual_left_to_right_score(self, left_h, left_inp, metadata, beta, **kwargs):
        looked_up = self.drop(self.embeddings(left_inp))
        rnn_updated = self.drop(self.unit(looked_up, left_h))

        if beta is not None:
            # bz x vocab_size (emission size)
            prefix_score = self.beta_scorer(rnn_updated)
            current_state = metadata["state"]  # bz
            # bz x vocab_size
            transition = self.transition_k[
                torch.arange(current_state.shape[0], device=current_state.device),
                current_state,
            ]
            beta_logits = torch.gather(beta, 1, transition)
            # TODO: drastically different scale?
            final_score = beta_logits + prefix_score
            return (rnn_updated, final_score, metadata)
        else:
            # use raw x,y encoding to update the hidden state
            rnn_updated_again = self.attune(rnn_updated, **kwargs)
            return (
                rnn_updated,
                self.scorer(torch.cat([rnn_updated, rnn_updated_again], dim=1)),
                metadata,
            )


class FSAGRUScorer(GRUScorer):
    def get_seqs(self, sequence: torch.Tensor, **kwargs):
        pass

    def get_init_states(self, batch_size, device):
        assert batch_size % self.k == 0
        h0, x0, metadata = super(FSAGRUScorer, self).get_init_states(
            batch_size=batch_size, device=device
        )
        metadata["state"] = torch.zeros((batch_size,), dtype=torch.long, device=device)
        metadata["accumulated"] = torch.zeros(
            (batch_size,), dtype=torch.long, device=device
        )
        metadata["next_is_output"] = torch.zeros(
            (batch_size,), dtype=torch.bool, device=device
        )
        metadata["next_is_input"] = torch.zeros(
            (batch_size,), dtype=torch.bool, device=device
        )
        metadata["vocab_use"] = torch.zeros(
            (batch_size, self.vocab_size),
            dtype=torch.get_default_dtype(),
            device=device,
        )
        return h0, x0, metadata

    def actual_left_to_right_score(
        self, left_h, left_inp, metadata, beta=None, **kwargs
    ):
        next_output = torch.empty_like(left_inp)
        next_output.fill_(self.__pad__)
        next_output.masked_scatter(metadata["next_is_output"], left_inp)

        next_input = torch.empty_like(left_inp)
        next_input.fill_(self.__pad__)
        next_input.masked_scatter(metadata["next_is_input"], left_inp)

        rnn_updated, scored, metadata = super(
            FSAGRUScorer, self
        ).actual_left_to_right_score(
            left_h,
            left_inp,
            metadata,
            beta,
            update_output=next_output,
            update_input=next_input,
        )
        metadata["next_is_output"] = left_inp == self.output_mark
        metadata["next_is_input"] = left_inp == self.input_mark

        metadata["accumulated"] = torch.where(
            left_inp == self.insertion_mark,
            metadata["accumulated"] + 1,
            metadata["accumulated"],
        )
        metadata["vocab_use"][
            torch.arange(left_inp.shape[0], device=left_inp.device), left_inp
        ] += 1

        if self.insert_threshold > 0:
            scored[:, self.insertion_mark] = torch.where(
                metadata["accumulated"] > self.insert_threshold,
                scored[:, self.insertion_mark]
                - self.insert_penalty * (metadata["length"] - self.insert_threshold),
                scored[:, self.insertion_mark],
            )

        if 0 < self.length_threshold < metadata["length"]:
            """
            zero_mask = torch.zeros_like(scored)
            zero_mask[:, self.__eos__] = self.length_penalty * (metadata['length'] - self.length_threshold)
            scored = scored + zero_mask
            """
            scored = scored - metadata["vocab_use"] * self.length_penalty

        metadata["state"] = self.update_fsa_state(left_inp, metadata["state"])

        return rnn_updated, scored, metadata

    def update_fsa_state(self, updated: torch.Tensor, prev_states: torch.Tensor):
        assert self.transition_k is not None
        transitions = self.transition_k[
            torch.arange(prev_states.shape[0], device=prev_states.device), prev_states
        ]
        return transitions[
            torch.arange(prev_states.shape[0], device=prev_states.device), updated
        ]

    def compute_beta_per_sample(self, transition):
        # TODO: parallelize the computation to make it faster
        n_states = transition.shape[0]
        parents = [[] for _ in range(n_states)]
        message = [[] for _ in range(n_states)]
        edge_list = [[] for _ in range(n_states)]
        # store beta score for each state
        beta = torch.zeros(n_states).to(self.W.device)
        # store embedding of each state
        beta_hat = torch.zeros(n_states, self.hid_dim).to(self.W.device)

        # construct parent nodes and child nodes for each state
        prev_states = []
        for i in range(n_states):
            for j in range(transition.shape[1]):
                to_state = transition[i, j].item()
                # omit self recursion for padding
                # if to_state == i and to_state != 0:
                #     print(f"recursive state {i} with arc f{Vocab.r_lookup(j)}")
                if to_state != 0 and to_state != i:
                    # state i can transition to state j
                    parents[i].append(to_state)
                    edge_list[to_state].append((i, j))
            if len(parents[i]) == 0:
                prev_states.append(i)

        # there should be only one start state, that is the end state
        assert len(prev_states) == 1
        beta[prev_states[0]] = 1  # init beta=1 for the end state

        # itr = 0
        while prev_states:
            # itr += 1
            new_states = []
            for top_node in prev_states:
                for (child, mark) in edge_list[top_node]:
                    # take out beta and beta hat of next state (parent)
                    parent_beta = beta[top_node]
                    parent_beta_hat = beta_hat[top_node]
                    # compute message passed
                    mark_emb = self.embeddings(torch.tensor(mark).to(self.Wx.device))
                    beta_hat_tran = self.beta_activation(
                        self.Wx @ mark_emb + self.Wh @ parent_beta_hat + self.beta_bias
                    )
                    beta_tran = (
                        torch.exp(self.W @ beta_hat_tran) * parent_beta
                    ).squeeze()
                    # store incoming messages
                    message[child].append((top_node, beta_tran, beta_hat_tran))
                    if len(message[child]) == len(parents[child]):
                        # message aggregation
                        for (_, sender_tran, _) in message[child]:
                            beta[child] += sender_tran
                        for (sender, sender_tran, sender_hat_tran) in message[child]:
                            tran_prob = sender_tran / beta[child]
                            beta_hat[child] += tran_prob * sender_hat_tran
                        new_states.append(child)
            prev_states = new_states
        # beta[0] = -1e9  # always disable the first state
        return beta

    def compute_beta_parallel(self):
        """compute beta prob using matrix parallelization"""
        bzk, n_states, vocab_size = self.transition_k.shape
        bz = bzk // self.k
        # track number of parents for each state
        parents = self.transition_k.new_full((bz, n_states), 0)
        visited = self.transition_k.new_full((bz, n_states), False).bool()
        # track the transition arc
        edges = self.transition_k.new_full((bz, n_states, n_states), 0).int()

        # TODO: parallelize the graph construction
        for b in range(bz):
            transition = self.transition_k[b * self.k]
            # build graph for current transition
            for i in range(n_states):
                for j in range(transition.shape[1]):
                    to_state = transition[i, j].item()
                    # omit self recursion for padding
                    # if to_state == i and to_state != 0:
                    #     print(f"recursive state {i} with arc f{Vocab.r_lookup(j)}")
                    if to_state != 0 and to_state != i:
                        # state i can transition to state j
                        parents[b][i] += 1
                        edges[b][to_state][i] = j
        # print(parents)
        # store beta score for each state
        beta = torch.zeros(bz, n_states).to(self.W.device)
        # store embedding of each state
        beta_hat = torch.zeros(bz, n_states, self.hid_dim).to(self.W.device)
        # track beta(parent->current)
        parent_msg = torch.zeros(bz, n_states, n_states).to(self.W.device)
        # track beta_hat(parent->current)
        parent_msg_emb = torch.zeros(bz, n_states, n_states, self.hid_dim).to(
            self.W.device
        )

        # Repeat until all sample finished:
        init = True
        # bz x #state x #state x hid (word emb dim)
        arc_emb = self.embeddings(edges)
        while True:
            # step 0: comptute current outgoing state (with 0 parents needs to be seen)
            no_parent = parents == 0
            cur_states = torch.logical_and(~visited, no_parent)  # bz x #states
            if torch.all(~cur_states):
                # no more outgoing states, break from the update
                break
            # step 1: aggregate info to compute beta and beta_hat, if at start, init it

            if init:
                # no need to init beta hat as they are fine starting 0s
                beta[cur_states] = 1
                init = False
            else:
                # compute transition prob q
                # compute beta and beta hat
                reverse_parent_msg = parent_msg.transpose(1, 2)
                beta[cur_states] = torch.sum(reverse_parent_msg[cur_states], dim=-1)
                reverse_parent_msg_emb = parent_msg_emb.transpose(1, 2)
                # bz x S x S x 1
                q = (
                    reverse_parent_msg[cur_states] / beta[cur_states].unsqueeze(-1)
                ).unsqueeze(-1)
                # sum(N x S x H * N x S x 1) -> N x H
                beta_hat[cur_states] = torch.sum(
                    reverse_parent_msg_emb[cur_states] * q, dim=1
                )

            # step 2: compute msg to their outgoing children
            # bz x #state x 1 x hid
            beta_hat_h = torch.einsum("abcd,ed->abce", beta_hat.unsqueeze(2), self.Wh)
            # bz x #states x #state x hid
            beta_hat_x = torch.einsum("abcd,ed->abce", arc_emb, self.Wx)
            # bz x #states x #state x hid
            beta_msg = self.beta_activation(beta_hat_h + beta_hat_x + self.beta_bias)
            msg_passing_mask = edges[cur_states] != 0
            # print(msg_passing_mask.shape)
            # TODO: currently require the intermediate tmp to pass the value
            # There should be more efficient approach to update the matrices
            tmp = torch.zeros(beta_msg[cur_states].shape).to(self.W.device)
            tmp[msg_passing_mask] = beta_msg[cur_states][msg_passing_mask]
            parent_msg_emb[cur_states] = tmp

            # bz x #state x #state x 1
            compatibility = torch.einsum("abcd,ed->abce", beta_msg, self.W).squeeze(-1)
            compatibility = torch.exp(compatibility)
            # make beta (bz x S x 1) to perform elementwise multiplication
            beta_transition = compatibility * beta.unsqueeze(-1)
            tran_tmp = torch.zeros(beta_transition[cur_states].shape).to(self.W.device)
            tran_tmp[msg_passing_mask] = beta_transition[cur_states][msg_passing_mask]
            parent_msg[cur_states] = tran_tmp

            # update visited and parent for next iteration
            visited[cur_states] = True
            parent_update_matrix = edges.new_full(edges.shape, 0).float()
            parent_update_tmp = torch.zeros(edges[cur_states].shape).to(self.W.device)
            parent_update_tmp[msg_passing_mask] = 1
            parent_update_matrix[cur_states] = parent_update_tmp
            num_visits = torch.sum(parent_update_matrix, dim=1)
            parents = parents - num_visits

        return beta.repeat_interleave(self.k, dim=0), parent_msg.repeat_interleave(
            self.k, dim=0
        )

    def compute_beta(self):
        beta, beta_hat = self.compute_beta_parallel()
        # """compute beta using backward algorithm along with Tree-LSTM to encode the weight"""
        # bz = self.transition_k.shape[0] // self.k
        # beta_all = self.transition_k.new_full((bz, self.transition_k.shape[1]), 0)

        # # print(f"batch size: {bz} and k: {self.k}")
        # for i in range(bz):
        #     transition = self.transition_k[i * self.k].int()
        #     beta = self.compute_beta_per_sample(transition)
        #     beta_all[i] = beta
        #     # emission = self.emission_k[i]
        #     # for j in range(emission.shape[0]):
        #     #     if emission[j, :].sum() == 1 and emission[j, self.__pad__] == 1:
        #     #         print("end state:", j)
        # print("beta computed for one batch")
        # beta_all = beta_all.repeat_interleave(self.k, dim=0)
        return beta

    def set_masks(self, emission: torch.Tensor, transition: torch.Tensor):
        assert len(emission.shape) == 3
        assert len(transition.shape) == 3
        with torch.no_grad():
            self.emission, self.transition = None, None
            self.emission, self.transition = (
                emission.contiguous(),
                transition.contiguous(),
            )

    def set_k(self, k: int):
        try:
            with torch.no_grad():
                self.k = k
                self.emission_k = None
                self.transition_k = None
                self.state_count_k = None
                self.emission_k = (
                    self.emission.view(
                        self.emission.shape[0],
                        1,
                        self.emission.shape[1],
                        self.emission.shape[2],
                    )
                    .expand(-1, self.k, -1, -1)
                    .reshape(-1, self.emission.shape[1], self.emission.shape[2])
                )

                self.transition_k = (
                    self.transition.view(
                        self.emission.shape[0],
                        1,
                        self.emission.shape[1],
                        self.emission.shape[2],
                    )
                    .expand(-1, self.k, -1, -1)
                    .reshape(-1, self.transition.shape[1], self.transition.shape[2])
                )

        except Exception as e:
            print(f"k={k} batch_size={self.emission.shape[0]}")
            raise e

    def __init__(
        self,
        hid_dim,
        vocab_size,
        activation=None,
        pad=0,
        bos=1,
        eos=2,
        dropout=0.5,
        max_length=400,
        insert_threshold: int = 2,
        insert_penalty: float = 1000.0,
        length_threshold: int = 0,
        length_penalty: float = 1000.0,
        k: int = 1,
        query=None,
        embeddings=None,
        tied_embeddings: bool = False,
        use_beta: bool = False,
    ):
        super(FSAGRUScorer, self).__init__(
            hid_dim=hid_dim,
            vocab_size=vocab_size,
            activation=activation,
            pad=pad,
            bos=bos,
            eos=eos,
            query=query,
            max_length=max_length,
            dropout=dropout,
            embeddings=embeddings,
            tied_embeddings=tied_embeddings,
        )

        self.use_beta = use_beta
        if self.use_beta:
            # init parameters for beta
            # State transition matrix for beta'(S')
            self.Wh = torch.nn.Parameter(torch.randn(hid_dim, hid_dim))
            torch.nn.init.xavier_uniform(self.Wh)
            self.Wh.requires_grad = True
            # Input embedding transformation matrix
            self.Wx = torch.nn.Parameter(torch.randn(hid_dim, hid_dim))
            torch.nn.init.xavier_uniform(self.Wx)
            self.Wx.requires_grad = True
            # compatibility function for computing beta(S->S')
            self.W = torch.nn.Parameter(torch.randn(1, hid_dim))
            torch.nn.init.xavier_uniform(self.W)
            self.W.requires_grad = True
            self.beta_activation = torch.nn.Tanh()
            # self.h0 = torch.zeros((1, hid_dim))
            self.beta_bias = torch.nn.Parameter(torch.zeros(hid_dim))
            self.beta_bias.requires_grad = True

        input_mark = Vocab.lookup("input-mark")
        output_mark = Vocab.lookup("output-mark")
        insertion_mark = Vocab.lookup("insertion-mark")
        neg_inf_vocab = torch.empty((self.vocab_size,))
        neg_inf_vocab.fill_(float("-inf"))
        zero_vocab = torch.zeros_like(neg_inf_vocab)
        self.register_buffer("neg_inf_vocab", neg_inf_vocab)
        self.register_buffer("zero_vocab", zero_vocab)
        self.emission, self.transition, = (
            None,
            None,
        )
        self.insertion_mark = insertion_mark
        self.insert_threshold = insert_threshold
        self.insert_penalty = insert_penalty
        self.length_threshold = length_threshold
        self.length_penalty = length_penalty
        self.k = k
        self.input_mark = input_mark
        self.output_mark = output_mark

    @staticmethod
    def get_state_mask_pynini(
        machine, vocab_size, pad, to_numpy: bool = False, weighted: bool = False
    ):
        from pynini import Fst, Weight

        machine: Fst
        zero = Weight.zero(machine.weight_type())
        import numpy as np

        assert machine.start() == 0
        emission = np.zeros(
            (machine.num_states() + 1, vocab_size),
            dtype=np.bool if not weighted else np.float,
        )
        transition = np.zeros((machine.num_states() + 1, vocab_size), dtype=np.long)
        if weighted:
            emission = np.log(emission)
            emission[machine.num_states(), pad] = 0
        else:
            emission[machine.num_states(), pad] = 1
        transition[machine.num_states(), pad] = machine.num_states()
        for state in range(machine.num_states()):
            arcs = machine.arcs(state)
            ilabel_set = set()
            for arc in arcs:
                arc: Weight

                nextstate = arc.nextstate
                if machine.final(nextstate) != zero:
                    nextstate = machine.num_states()
                if weighted:
                    emission[state, arc.ilabel] = -float(arc.weight)
                else:
                    emission[state, arc.ilabel] = 1
                assert arc.ilabel not in ilabel_set
                ilabel_set.add(arc.ilabel)
                transition[state, arc.ilabel] = nextstate
        if to_numpy:
            return emission, transition
        return torch.from_numpy(emission), torch.from_numpy(transition)

    def mask_out_invalid(self, inp: torch.Tensor, metadata) -> torch.Tensor:
        assert self.emission is not None
        to_return = super(FSAGRUScorer, self).mask_out_invalid(inp, metadata)
        current_fsa_states = metadata["state"]
        emission_k: torch.Tensor = self.emission_k
        if emission_k.dtype == torch.bool:
            state_emission_masks = torch.where(
                emission_k[
                    torch.arange(current_fsa_states.shape[0]), current_fsa_states
                ],
                self.zero_vocab,
                self.neg_inf_vocab,
            )
        else:
            state_emission_masks = emission_k[
                torch.arange(current_fsa_states.shape[0]), current_fsa_states
            ]
        return to_return + state_emission_masks.to(to_return.device)


class FSAMaskScorer(FSAGRUScorer):
    def actual_left_to_right_score(
        self, left_h, left_inp, metadata, beta=None, **kwargs
    ):
        next_output = torch.empty_like(left_inp)
        next_output.fill_(self.__pad__)
        next_output.masked_scatter(metadata["next_is_output"], left_inp)

        next_input = torch.empty_like(left_inp)
        next_input.fill_(self.__pad__)
        next_input.masked_scatter(metadata["next_is_input"], left_inp)

        scored = torch.zeros(
            (left_inp.shape[0], self.vocab_size),
            dtype=torch.get_default_dtype(),
            device=left_inp.device,
        )
        rnn_updated = left_h
        metadata["next_is_output"] = left_inp == self.output_mark
        metadata["next_is_input"] = left_inp == self.input_mark

        metadata["accumulated"] = torch.where(
            left_inp == self.insertion_mark,
            metadata["accumulated"] + 1,
            metadata["accumulated"],
        )
        metadata["vocab_use"][
            torch.arange(left_inp.shape[0], device=left_inp.device), left_inp
        ] += 1
        metadata["state"] = self.update_fsa_state(left_inp, metadata["state"])
        return rnn_updated, scored, metadata


class CompositeScorer(LeftToRightScorer):
    def __init__(
        self,
        hid_dim,
        vocab_size,
        activation=None,
        pad=0,
        bos=1,
        eos=2,
        num_hidden_states=2,
        query: Optional[EncodedQuery] = None,
        context_state_dim: Optional[int] = None,
        max_length: int = 400,
        embeddings: Optional[torch.nn.modules.Embedding] = None,
    ):
        if context_state_dim is None:
            context_state_dim = hid_dim * num_hidden_states
        super(CompositeScorer, self).__init__(
            hid_dim=hid_dim,
            vocab_size=vocab_size,
            activation=activation,
            pad=pad,
            bos=bos,
            eos=eos,
            query=query,
            context_state_dim=context_state_dim,
            max_length=max_length,
            embeddings=embeddings,
        )
        self.num_hidden_states = num_hidden_states
        self.scorer = torch.nn.Linear(self.get_output_dim(), vocab_size)

    @abstractmethod
    def get_output_dim(self):
        pass

    def get_init_states(self, batch_size, device):
        hs = []
        inp0 = None
        for _ in range(self.num_hidden_states):
            h0, inp0, _ = super(CompositeScorer, self).get_init_states(
                batch_size, device
            )  # ignore per layer metadata
            hs.append(h0)
        to_return = self.combine_h(*hs)
        assert inp0 is not None
        return to_return, inp0, {"length": 1}

    def combine_h(self, *args):
        """

        :param args: a bunch of hidden states, each of shape batch_size x hid_dim
        :return:
        """
        batch_size = args[0].shape[0]
        return torch.stack(args, dim=2).reshape((batch_size, -1))

    def split_h(self, h: torch.Tensor):
        """

        :param h:
        :return: a bunch of hidden states
        """
        assert len(h.shape) == 2
        h_reshaped = h.reshape((h.shape[0], self.hid_dim, self.num_hidden_states))
        return torch.unbind(h_reshaped, 2)


class StaticRNNScorer(CompositeScorer):
    def get_last_layer(self, hx: torch.Tensor):
        batch_size = hx.shape[0]
        return hx.reshape((batch_size, -1, self.hid_dim, self.num_hidden_states))[
            :, :, :, -1
        ]

    def get_seqs(
        self,
        sequence: torch.Tensor,
        ext: Optional[torch.Tensor] = None,
        bptt: bool = False,
        bptt_window_size: int = 35,
        **kwargs,
    ):
        """

        :param sequence:
        :param ext:
        :param kwargs:
        :return:
        """

        if self.external_input:
            assert ext is not None

        batch_size = sequence.shape[0]
        seq_length = sequence.shape[1]
        dev = sequence.device
        init_h, init_inp, metadata = self.get_init_states(batch_size, dev)

        # TODO note that prepended_seq is different from that of TransformerScorer!
        prepended_seq = torch.cat((init_inp[:, np.newaxis], sequence), dim=1)
        assert prepended_seq.shape[0] == batch_size
        assert prepended_seq.shape[1] == seq_length + 1
        mask = self.get_seq_mask(prepended_seq)
        looked_up = self.drop(self.embeddings(prepended_seq))
        if self.external_input:
            ext_prepended = ext
            looked_up = torch.cat((looked_up, self.drop(ext_prepended)), dim=2)

        if self.bidirectional:
            assert not self.self_normalized
            if self.right_to_left:
                output = self.wrapped(
                    inputs=looked_up.flip(dims=(1,)),
                    mask=self.get_seq_mask(prepended_seq.flip(dims=(1,))),
                ).flip(dims=(1,))
            else:
                # FIXME this is a hack. it is not clear how we should initialize the right-to-left h_0 so we pass in blank
                output = self.wrapped(
                    inputs=looked_up,
                    mask=mask,
                )
        else:

            if self.rnn_type == "LSTM":
                # FIXME this is a hack. we ignore h0 for LSTMs but we should learn how to properly initialize it...
                pass
            else:
                pass
                # permuted_h = init_h.reshape((batch_size, self.num_hidden_states, self.hid_dim)).permute(1, 0, 2).contiguous()
                ## permuted_h = init_h.reshape((batch_size, self.num_hidden_states, self.hid_dim))

            if bptt:
                all_output = []

                num_windows = int(np.ceil(looked_up.shape[1] / bptt_window_size))
                for window in range(num_windows):
                    windowed_looked_up = looked_up[
                        :, window * (bptt_window_size) : (window + 1) * bptt_window_size
                    ]
                    local_output = self.wrapped(
                        inputs=windowed_looked_up,
                        mask=mask[
                            :,
                            window
                            * (bptt_window_size) : (window + 1)
                            * bptt_window_size,
                        ],
                    )
                    # (batch_size, len, hid_dim), (num_hidden_states, batch_size, hid_dim)
                    all_output.append(local_output)
                output = torch.cat(all_output, dim=1).contiguous()
            else:

                output = self.wrapped(inputs=looked_up, mask=mask)
        assert output.shape[0] == batch_size
        assert output.shape[1] == seq_length + 1

        d = self.drop(output)
        attuned = self.attune(d)
        if not self.right_to_left:
            # TODO not sure about this one
            attuned = attuned * mask[:, :, np.newaxis].expand(
                -1, -1, attuned.shape[2]
            ).to(torch.get_default_dtype())

        if self.simple_sum:
            return self.scorer(attuned)[:, :-1, 0].sum(dim=1), attuned
        # FIXME below is likely incorrect
        return self.scorer(attuned)[:, :-1, :], attuned

    def has_seq(self):
        return True

    def get_output_dim(self):
        if self.bidirectional:
            return self.hid_dim * 2
        else:
            return self.hid_dim

    def load_pytorch_example_states(self, model_fname: str):
        loaded: model.RNNModel = torch.load(model_fname)
        self.embeddings.load_state_dict(loaded.encoder.state_dict())
        self.unit.load_state_dict(loaded.rnn.state_dict())
        self.unit.flatten_parameters()
        self.scorer.load_state_dict(loaded.decoder.state_dict())
        self.drop.p = loaded.drop.p

    def __init__(
        self,
        hid_dim,
        vocab_size,
        activation=None,
        pad=0,
        bos=1,
        eos=2,
        dropout=0.5,
        query: Optional[EncodedQuery] = None,
        max_length=400,
        num_hidden_states: int = 2,
        locally_normalized: bool = True,
        context_state_dim: Optional[int] = None,
        bidirectional: bool = False,
        external_input: bool = False,
        external_hid_dim: int = 500,
        after_sum_activation: Optional[str] = None,
        after_sum_activation_offset: float = 0.0,
        attention: bool = False,
        simple_sum: bool = False,
        right_to_left: bool = False,
        embeddings: Optional[torch.nn.Embedding] = None,
        rnn_type: str = "GRU",
        tied_embeddings: bool = False,
        label_smoothing: float = 0.0,
        two_level_marks: bool = False,
        **kwargs,
    ):
        from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper

        self.bidirectional = bidirectional
        super(StaticRNNScorer, self).__init__(
            hid_dim,
            vocab_size,
            activation=activation,
            pad=pad,
            bos=bos,
            eos=eos,
            query=query,
            max_length=max_length,
            context_state_dim=context_state_dim,
            num_hidden_states=num_hidden_states,
            embeddings=embeddings,
        )
        self.external_input = external_input
        self.external_hid_dim = external_hid_dim

        if self.external_input:
            inp_hid_dim = self.external_hid_dim + hid_dim
        else:
            inp_hid_dim = hid_dim

        self.attention = attention
        if self.attention:
            # FIXME this really doesn't belong in this module!!
            assert bidirectional
            from allennlp.modules.attention.linear_attention import LinearAttention

            self.attention_module = LinearAttention(
                self.hid_dim * 2, self.hid_dim, normalize=True
            )

        self.simple_sum = simple_sum
        if self.simple_sum:
            assert bidirectional

        self.rnn_type = rnn_type
        model = None
        if self.rnn_type == "GRU":
            model: Type[RNNBase] = GRU
        elif self.rnn_type == "LSTM":
            model: Type[RNNBase] = LSTM
        else:
            assert NotImplementedError

        self.right_to_left = right_to_left
        if bidirectional:
            if right_to_left:
                self.unit = model(
                    inp_hid_dim,
                    hid_dim,
                    num_layers=num_hidden_states,
                    dropout=dropout,
                    bidirectional=False,
                    batch_first=True,
                )
                self.scorer = torch.nn.Linear(self.hid_dim, self.vocab_size)
            else:
                self.unit = model(
                    inp_hid_dim,
                    hid_dim,
                    num_layers=num_hidden_states,
                    dropout=dropout,
                    bidirectional=True,
                    batch_first=True,
                )
        else:
            self.unit = model(
                input_size=inp_hid_dim,
                hidden_size=hid_dim,
                num_layers=num_hidden_states,
                dropout=dropout,
                bidirectional=False,
                batch_first=True,
            )
        self.wrapped = PytorchSeq2SeqWrapper(self.unit, stateful=True)
        if embeddings is None:
            self.embeddings.weight.data.uniform_(-0.1, 0.1)

        if tied_embeddings:
            self.scorer = TiedScorer(self.embeddings, vocab_size)
        else:
            self.scorer.bias.data.zero_()
            self.scorer.weight.data.uniform_(-0.1, 0.1)

        self.drop = Dropout(dropout)
        self.locally_normalized = locally_normalized
        self.label_smoothing = label_smoothing

        self.after_sum_activation = after_sum_activation
        self.after_sum_activation_offset = after_sum_activation_offset

        self.two_level_marks = two_level_marks
        # FIXME these are hacks that hopefully patch mem leaking
        self.first_mask = None
        self.first_marks = None
        self.working_buffer = None
        self.first_scores = None
        self.second_scores = None

        # FIXME debugging flag
        self.debugging = False
        self.bptt_warning = False
        return

    @property
    def self_normalized(self):
        return self.locally_normalized

    def actual_left_to_right_score(
        self,
        left_h: torch.Tensor,
        left_inp: torch.Tensor,
        metadata: Dict[AnyStr, Any],
        beta=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[AnyStr, Any]]:
        """

        :param left_h: batch_size x hid_dim x num_hidden_states
        :param left_inp: batch_size
        :param metadata: broad-coverage challenge corpus forsentence understanding through inference
        :return:
        """
        assert not self.two_level_marks
        assert not self.bidirectional
        assert not self.external_input
        batch_size = left_h.shape[0]
        permuted_h = (
            left_h.reshape((batch_size, self.hid_dim, self.num_hidden_states))
            .permute(2, 0, 1)
            .contiguous()
        )
        looked_up = self.drop(self.embeddings(left_inp))[:, np.newaxis, :]
        looked_up = looked_up.permute((1, 0, 2))
        output, hn = self.unit(
            input=looked_up, hx=permuted_h
        )  # (1, batch_size, hid_dim), (num_hidden_states, batch_size, hid_dim)
        output = output.permute((1, 0, 2))
        permuted_hn = hn.permute(1, 2, 0).reshape((batch_size, -1))
        to_score = output.reshape((batch_size, self.hid_dim))

        # FIXME the semantics of attune is different from GRUSCorer
        scored = self.scorer(self.attune(to_score))
        if self.self_normalized:
            scored = torch.log_softmax(scored, dim=1)
        return permuted_hn, scored, metadata

    def evaluate_seq(
        self,
        sequence: torch.Tensor,
        temp: float = 1.0,
        return_states: bool = False,
        **kwargs,
    ):
        if self.debugging:
            assert not self.two_level_marks
            assert "output_mask" not in kwargs
            to_return = super(StaticRNNScorer, self).evaluate_seq(sequence)
        else:
            self.wrapped.reset_states()
            if self.two_level_marks:
                assert return_states is False
                with torch.no_grad():
                    self.first_mask = Utils.get_level_zero_mask(sequence, self.__eos__)
                    self.first_marks, self.working_buffer = Utils.padded_masked_select(
                        sequence, self.first_mask, self.__pad__
                    )
                self.first_scores = self.evaluate_seq_with_temp(
                    self.first_marks, temp, return_states=False, **kwargs
                )
                self.second_scores = self.evaluate_seq_with_temp(
                    sequence, temp, return_states=False, **kwargs
                )
                to_return = self.first_scores + self.second_scores
                self.wrapped.reset_states()

            else:
                to_return = self.evaluate_seq_with_temp(
                    sequence, temp, return_states=return_states, **kwargs
                )
        if self.after_sum_activation is not None:
            if self.after_sum_activation == "tanh":
                to_return = -2 * torch.tanh(
                    to_return + self.after_sum_activation_offset
                )
            elif self.after_sum_activation == "softplus":
                to_return = -torch.nn.functional.softplus(
                    to_return + self.after_sum_activation_offset
                )
            else:
                raise NotImplementedError
        return to_return

    def smooth_one_hot(
        self,
        true_labels: torch.Tensor,
        classes: int,
        scores: torch.Tensor,
        smoothing=0.0,
    ):
        """
        if smoothing == 0, it's one-hot method
        if 0 < smoothing < 1, it's smooth method

        """
        assert 0 <= smoothing < 1
        confidence = 1.0 - smoothing
        label_shape = torch.Size((true_labels.size(0), classes))
        with torch.no_grad():
            # true_dist = torch.empty(size=label_shape, device=true_labels.device)
            greater_than_inf = scores > float("-inf")
            classes_per_datapoint = (
                greater_than_inf.to(torch.get_default_dtype())
                .sum(dim=1)[:, None]
                .expand(-1, classes)
            )
            true_dist = torch.clone(smoothing / (classes_per_datapoint - 1))
            true_dist = torch.masked_fill(true_dist, ~greater_than_inf, 0)
            true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
        return true_dist

    def evaluate_seq_with_temp(
        self,
        sequence: torch.Tensor,
        temp: float = 1.0,
        return_states: bool = False,
        bptt: bool = False,
        bptt_window_size: int = 35,
        output_mask: Optional = None,
        **kwargs,
    ):
        """

        :param sequence:
        :param temp:
        :param return_states:
        :param kwargs:
        :return:
        """
        if not self.bptt_warning:
            self.bptt_warning = True
            print(f"{bptt}\t{bptt_window_size}")

        batch_size = sequence.shape[0]
        seq_length = sequence.shape[1]

        scores, hidden_states = self.get_seqs(
            sequence, bptt=bptt, bptt_window_size=bptt_window_size, **kwargs
        )
        if self.simple_sum:
            if return_states:
                return scores, hidden_states
            return scores
        assert scores.shape[0] == batch_size
        assert scores.shape[1] == seq_length
        # scores_mask = self.batched_mask_out_invalid(sequence)
        scores_mask = self.slow_but_correct_batched_mask_out_invalid(
            sequence, self.max_length
        )
        # FIXME check condition below
        if False and self.training and self.self_normalized:
            through_activation = self.activation(scores)
        else:
            through_activation = (
                self.activation(self.pad_masking_3d(scores) + scores_mask) / temp
            )
            # through_activation = self.activation(scores + scores_mask) # /temp

        if self.self_normalized or self.right_to_left:
            final_score = torch.log_softmax(through_activation + scores_mask, dim=2)
        else:
            final_score = through_activation + scores_mask
        true_labels = sequence.flatten()
        reshaped_for_selection = final_score.reshape(-1, self.vocab_size)

        if self.training:
            smoothed_labels = self.smooth_one_hot(
                true_labels,
                self.vocab_size,
                scores=reshaped_for_selection,
                smoothing=self.label_smoothing if self.training else 0.0,
            )
            reshaped_for_selection = reshaped_for_selection.clamp(min=-10e8, max=10e8)
            selected_flat = (smoothed_labels * reshaped_for_selection).sum(dim=1)
        else:
            selected_flat = reshaped_for_selection[
                torch.arange(
                    reshaped_for_selection.shape[0],
                    device=reshaped_for_selection.device,
                ),
                true_labels,
            ]

        mask_seq = (sequence != self.__pad__).to(torch.get_default_dtype())
        # FIXME the first condition below seems fishy
        if False and self.training and self.self_normalized:
            selected = selected_flat.reshape(batch_size, seq_length)
            sum_of_scores = selected.sum()
        else:
            selected = selected_flat.reshape(batch_size, seq_length) * mask_seq
            if output_mask is not None:
                selected *= output_mask.to(torch.get_default_dtype())
            sum_of_scores = selected.sum(dim=1)
        if return_states:
            return sum_of_scores, hidden_states
        return sum_of_scores


class GlobalCompositionNROScorer(torch.nn.Module):
    def __init__(self, *args, forced_type=None, mask=False, **kwargs):
        super(GlobalCompositionNROScorer, self).__init__()
        self._forced_type = forced_type
        self.p_scorer = StaticRNNScorer(*args, **kwargs)
        self.full_scorer = StaticRNNScorer(*args, **kwargs)
        self.mask = mask

        self.__pad__ = self.full_scorer.__pad__
        self.__eos__ = self.full_scorer.__eos__
        self.__bos__ = self.full_scorer.__bos__

    def forward(self, sequence: torch.Tensor, **kwargs):
        tmp_mask = torch.zeros_like(sequence, device=sequence.device)

        if self._forced_type == "full":
            return self.full_scorer(sequence, output_mask=-tmp_mask + 1, **kwargs)
        elif self._forced_type == "p":
            return self.p_scorer(sequence, output_mask=-tmp_mask + 1, **kwargs)
        else:
            first, second, first_mask, second_mask = NRO.extract_first_second(
                sequence,
                Vocab.lookup("cpstart"),
                Vocab.lookup("cpend"),
                Vocab.lookup("cp1"),
                Vocab.lookup("cp2"),
                self.__eos__,
                self.__pad__,
            )
            first_score = self.full_scorer(first, **kwargs)
            # note that the full scorer takes into account anything that is not scored by p_scorer
            # therefore, we do not use first_mask as output_mask. but rather, we use the complement of second_mask below
            if self.mask:
                second_score = self.full_scorer(
                    sequence,
                    presequence=second,
                    output_mask=second_mask.new_full(second_mask.size(), 1),
                    **kwargs,
                )
            else:
                second_score = self.full_scorer(
                    sequence, presequence=first, output_mask=~first_mask, **kwargs
                )
            return second_score


class WFSTScorer(ScorerBase):
    def __init__(self, pad, bos, eos, scorer):
        super(WFSTScorer, self).__init__(pad, bos, eos)
        self.scorer = scorer

    def evaluate_seq(self, sequence: torch.Tensor, **kwargs):
        return self.wfst_score(sequence)

    def wfst_scorer_masked(self, t: torch.Tensor):
        # note: we parametrize an arc's weight as *solely decided by the mark on it*
        # this allows us to score a path in an WFST just by looking at its marks
        # and the marks can be read independent of each other
        #

        scored = self.scorer(t).reshape(t.shape)
        masked = t == self.__pad__
        return torch.where(masked, torch.zeros_like(scored), scored)

    def wfst_dist_over_next(self, vocab_size):
        t = torch.arange(vocab_size)
        return self.wfst_scorer_masked(t)

    def wfst_score(self, t: torch.Tensor):
        assert len(t.shape) == 2
        return self.wfst_scorer_masked(t).sum(dim=1)
