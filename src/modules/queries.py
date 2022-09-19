from abc import ABCMeta, abstractmethod
from typing import Optional

import torch
from torch.nn.modules import Embedding


class QueryBase(torch.nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def query(self, state: torch.Tensor, **kwargs):
        pass

    @abstractmethod
    def get_output_dim(self) -> int:
        pass

    def forward(self, state: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.query(state, **kwargs)

    @abstractmethod
    def encode(self, **kwargs):
        pass


class EncodedQuery(QueryBase):
    def __init__(self, previously_encoded: bool = False):
        super(EncodedQuery, self).__init__()
        self.__encoded = None
        self.__mask = None
        self.previously_encoded = previously_encoded

    def encoded(self):
        return self.__encoded is not None

    def encode(self, to_encode: torch.Tensor, mask: torch.Tensor, **kwargs):
        assert not self.encoded()
        self.__encoded = self.__actually_encode__(to_encode, mask, **kwargs)
        self.__mask = mask

    def undo_encode(self):
        self.__encoded = None

    @abstractmethod
    def __actually_encode__(self, to_encode: torch.Tensor, mask: torch.Tensor, **kwargs):
        pass

    def query(self, state: torch.Tensor, **kwargs):
        assert self.encoded()

        # deal with batching
        state_shape = state.shape
        encoded_shape = self.__encoded.shape
        assert len(encoded_shape) == len(state_shape) + 1
        if encoded_shape[0] == state_shape[0]:
            # in this case, encoded has shape batch_size x seq_length x hid_dim_1
            # and state has shape batch_size x hid_dim_2
            return self.__actually_query__(state, self.__encoded, self.__mask, **kwargs)
        else:
            # in this case, encoded must have shape 1 x seq_length x hid_dim_1,
            # and we must expand the 0-th dimension into batch_size
            assert encoded_shape[0] == 1, 'shapes: encoded_shape {} state_shape {}'.format(encoded_shape, state_shape)
            # add an additional dimension at the beginning and expand it
            expanded_encoded = self.__encoded.expand(state_shape[0], -1, -1)
            expanded_mask = self.__mask.expand(state_shape[0], -1)
            return self.__actually_query__(state, expanded_encoded, expanded_mask, **kwargs)

    @abstractmethod
    def __actually_query__(self, state: torch.Tensor, encoded: torch.Tensor, mask: torch.Tensor, **kwargs):
        pass


class BiGRUEncodedQuery(EncodedQuery):
    def __init__(self, hid_dim: int, num_layers: int, state_hid_dim,
                 embedder: Optional[Embedding] = None, drop: float = 0.5,
                 vocab_size: Optional[int] = None,
                 previously_encoded_hid_dim: Optional[int] = None,
                 previously_encoded: bool = False,
                 activation=torch.relu):
        from allennlp.modules.attention.bilinear_attention import BilinearAttention
        from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
        super(BiGRUEncodedQuery, self).__init__(previously_encoded=previously_encoded)
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.previously_encoded = previously_encoded
        if self.previously_encoded:
            assert previously_encoded_hid_dim is not None
            self.embedder = torch.nn.Linear(previously_encoded_hid_dim, hid_dim)
        else:
            if embedder is not None:
                self.embedder = embedder
                self.add_module('embedder', embedder)
            else:
                self.embedder = Embedding(vocab_size, hid_dim)
        unit = torch.nn.GRU(input_size=hid_dim, hidden_size=hid_dim, num_layers=num_layers,
                            batch_first=True, dropout=drop, bidirectional=True)
        self.unit = PytorchSeq2SeqWrapper(unit, stateful=False)
        self.attn = BilinearAttention(vector_dim=state_hid_dim, matrix_dim=2 * hid_dim, normalize=True)
        self.state_hid_dim = state_hid_dim
        self.activation=activation

    def __actually_encode__(self, to_encode: torch.Tensor, mask: torch.Tensor, **kwargs):
        """

        :param to_encode: batch_size x sequence_length OR batch_size x sequence_length x hid_dim
        :param mask: batch_size x sequence_length
        :return: batch_size x sequence_length x hid_dim
        """
        if self.previously_encoded:
            embedded = self.activation(self.embedder(to_encode)) # FIXME make this more versatile
        else:
            embedded = self.embedder(to_encode)
        out = self.unit(inputs=embedded, mask=mask)
        return out

    def __actually_query__(self, state: torch.Tensor, encoded: torch.Tensor, mask: torch.Tensor, **kwargs):
        weights = self.attn(vector=state, matrix=encoded, matrix_mask=mask)
        from allennlp.nn.util import weighted_sum
        return weighted_sum(encoded, weights)

    def get_output_dim(self) -> int:
        return 2 * self.hid_dim

class QueryPi(EncodedQuery):
    def __actually_encode__(self, to_encode: torch.Tensor, mask: torch.Tensor, **kwargs):
        pass

    def __actually_query__(self, state: torch.Tensor, encoded: torch.Tensor, mask: torch.Tensor, **kwargs):
        pass

    def update_buffer_indices(self, to_update: torch.Tensor, buffer: torch.Tensor, indices: torch.Tensor):
        try:
            buffer[torch.arange(to_update.shape[0]), indices].masked_scatter((to_update!=self.pad), to_update)
            indices = self.update_indices(indices, to_update)
            buffer[torch.arange(to_update.shape[0]), indices].fill_(self.eos)
        except Exception as e:
            raise e
        return buffer, indices

    def update_indices(self, indices, to_update):
        offset = torch.where((to_update != self.pad), torch.ones_like(to_update, dtype=torch.int),
                             torch.zeros_like(to_update, dtype=torch.int))
        indices = indices + offset
        return indices

    def query_buffer(self, buffer, state, machine, attention):
        mask = (buffer != self.pad)
        prefix_queried = machine(inputs=self.embedder(buffer), mask=mask)
        weighted = self.get_weighted(attention, mask, prefix_queried, state)
        return weighted

    def get_weighted(self, attention, mask, prefix_queried, state):
        from allennlp.nn.util import weighted_sum

        weights = attention(vector=state, matrix=prefix_queried, matrix_mask=mask)
        weighted = weighted_sum(prefix_queried, weights)
        return weighted

    def query(self, state: torch.Tensor = None, update_output: Optional[torch.Tensor] = None,
              update_input: Optional[torch.Tensor] = None):
        if not self.encoded():
            batch_size = state.shape[0]
            indices, output_buffer = self.init_buffer_indices(batch_size, state)
            self.output_buffer = output_buffer
            self.output_indices = indices

            input_indices, input_buffer = self.init_buffer_indices(batch_size, state)
            self.input_buffer = input_buffer
            self.input_indices = input_indices
        self.input_buffer, self.input_indices = self.update_buffer_indices(update_input, self.input_buffer, self.input_indices)
        self.output_buffer, self.output_indices = self.update_buffer_indices(update_output, self.output_buffer, self.output_indices)
        input_weighted = self.query_buffer(self.input_buffer, state, self.input_gru, self.input_attn)
        output_weighted = self.query_buffer(self.output_buffer, state, self.output_gru, self.output_attn)
        return torch.cat([input_weighted, output_weighted], dim=1)

    def get_output_dim(self) -> int:
        return 4 * self.hid_dim

    def encode(self, to_encode, mask, **kwargs):
        return

    def undo_encode(self):
        self.input_buffer = None
        self.input_indices = None
        self.output_buffer = None
        self.output_indices = None

    def encoded(self):
        return isinstance(self.input_buffer, torch.Tensor)

    def init_buffer_indices(self, batch_size, to_encode):
        buffer = torch.empty(batch_size, self.max_length, dtype=torch.long, device=to_encode.device)
        buffer.fill_(self.pad)
        buffer[:, 0].fill_(self.bos)
        buffer[:, 1].fill_(self.eos)
        indices = torch.ones((batch_size,), dtype=torch.long, device=to_encode.device)
        return indices, buffer

    def __init__(self, hid_dim: int, num_layers: int, state_hid_dim,
                 embedder: Optional[Embedding] = None, drop: float = 0.5,
                 vocab_size: Optional[int] = None,
                 previously_encoded_hid_dim: Optional[int] = None,
                 previously_encoded: bool = False,
                 activation=torch.relu,
                 max_length: int = 100, pad: int = 0, bos: int = 0, eos: int = 0):
        from allennlp.modules.attention.bilinear_attention import BilinearAttention
        from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
        super(QueryPi, self).__init__()
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.previously_encoded = previously_encoded
        if self.previously_encoded:
            assert previously_encoded_hid_dim is not None
            self.embedder = torch.nn.Linear(previously_encoded_hid_dim, hid_dim)
        else:
            if embedder is not None:
                self.embedder = embedder
                self.add_module('embedder', embedder)
            else:
                self.embedder = Embedding(vocab_size, hid_dim)
        self.input_gru = PytorchSeq2SeqWrapper(torch.nn.GRU(input_size=hid_dim, hidden_size=hid_dim,
                                                            num_layers=num_layers,
                                                            batch_first=True, dropout=drop,
                                                            bidirectional=True), stateful=False)
        self.output_gru = PytorchSeq2SeqWrapper(torch.nn.GRU(input_size=hid_dim, hidden_size=hid_dim,
                                                             num_layers=num_layers,
                                                             batch_first=True, dropout=drop,
                                                             bidirectional=True), stateful=False)
        self.input_attn = BilinearAttention(vector_dim=state_hid_dim, matrix_dim=2 * hid_dim, normalize=True)
        self.output_attn = BilinearAttention(vector_dim=state_hid_dim, matrix_dim=2 * hid_dim, normalize=True)
        self.pad = pad
        self.bos = bos
        self.eos = eos
        self.state_hid_dim = state_hid_dim
        self.activation = activation
        self.max_length = max_length
        self.undo_encode()


class QueryPiGivenXAndY(QueryPi):
    def encode(self, to_encode, mask, to_encode_2 = None, mask_2 = None, **kwargs):
        batch_size = to_encode.shape[0]
        assert batch_size == mask.shape[0] == to_encode_2.shape[0] == mask_2.shape[0]
        self.input_buffer = self.input_gru(inputs=self.embedder(to_encode), mask=mask)
        self.input_mask = mask
        self.output_buffer = self.output_gru(inputs=self.embedder(to_encode_2), mask=mask_2)
        self.output_mask = mask_2
        self.input_indices = torch.zeros(batch_size, dtype=torch.long, device=to_encode.device)
        self.output_indices = torch.zeros(batch_size, dtype=torch.long, device=to_encode_2.device)

    def query(self, state: torch.Tensor = None, update_output: Optional[torch.Tensor] = None,
              update_input: Optional[torch.Tensor] = None):
        self.input_indices = self.update_indices(self.input_indices, update_input)
        self.output_indices = self.update_indices(self.output_indices, update_output)
        input_weighted = self.get_weighted(self.input_attn, self.input_mask, self.input_buffer, state)
        output_weighted = self.get_weighted(self.output_attn, self.output_mask, self.output_buffer, state)
        return torch.cat([input_weighted, output_weighted], dim=1)


class QueryPiGivenX(BiGRUEncodedQuery):
    def __init__(self, hid_dim: int, num_layers: int, state_hid_dim,
                 embedder: Optional[Embedding] = None, drop: float = 0.5,
                 vocab_size: Optional[int] = None,
                 previously_encoded_hid_dim: Optional[int] = None,
                 previously_encoded: bool = False,
                 activation=torch.relu,
                 max_output_length: int = 100, pad: int = 0, bos: int = 0, eos: int = 0):
        from allennlp.modules.attention.bilinear_attention import BilinearAttention
        from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper

        super(QueryPiGivenX, self).__init__(hid_dim, num_layers, state_hid_dim, embedder, drop, vocab_size,
                                            previously_encoded_hid_dim, previously_encoded,
                                            activation)
        self.output_gru = PytorchSeq2SeqWrapper(torch.nn.GRU(input_size=hid_dim, hidden_size=hid_dim,
                                                             num_layers=num_layers,
                                                             batch_first=True, dropout=drop,
                                                             bidirectional=True), stateful=False)
        self.output_attn = BilinearAttention(vector_dim=state_hid_dim, matrix_dim=2 * hid_dim, normalize=True)

        self.input_gru = PytorchSeq2SeqWrapper(torch.nn.GRU(input_size=hid_dim, hidden_size=hid_dim,
                                                            num_layers=num_layers,
                                                            batch_first=True, dropout=drop,
                                                            bidirectional=True), stateful=False)

        self.max_output_length = max_output_length
        self.pad = pad
        self.bos = bos
        self.eos = eos
        
    def __actually_encode__(self, to_encode: torch.Tensor, mask: torch.Tensor, **kwargs):
        batch_size = to_encode.shape[0]
        output_buffer = torch.empty(batch_size, self.max_output_length, dtype=torch.long, device=to_encode.device)
        output_buffer.fill_(self.pad)
        output_buffer[:, 0].fill_(self.bos)
        output_buffer[:, 1].fill_(self.eos)

        indices = torch.ones(batch_size, dtype=torch.long, device=to_encode.device)
        self.register_buffer('output_buffer', output_buffer, persistent=False)
        self.register_buffer('indices', indices, persistent=False)

        input_indices = torch.ones(batch_size, dtype=torch.long, device=to_encode.device)
        self.register_buffer('input_indices', input_indices)
        to_encode_bos = torch.empty((to_encode.shape[0], 1), dtype=to_encode.dtype, device=to_encode.device)
        to_encode_bos.fill_(self.bos)
        to_encode_concatenated = torch.cat([to_encode_bos, to_encode], dim=1)
        monotonic_input_encoded = self.input_gru(inputs=self.embedder(to_encode_concatenated), mask=mask)
        self.register_buffer('monotonic_input_encoded', monotonic_input_encoded, persistent=False)
        return monotonic_input_encoded

    def update_output_buffer(self, to_update: torch.Tensor):
        try:
            self.output_buffer[torch.arange(to_update.shape[0]), self.indices].masked_scatter((to_update!=self.pad), to_update)
            offset = torch.where((to_update != self.pad), torch.ones_like(to_update, dtype=torch.int), torch.zeros_like(to_update, dtype=torch.int))
            self.indices = self.indices + offset
            self.output_buffer[torch.arange(to_update.shape[0]), self.indices].fill_(self.eos)
        except Exception as e:
            raise e

    def update_input_buffer(self, to_update: torch.Tensor):
        try:
            offset = torch.where((to_update != self.pad), torch.ones_like(to_update, dtype=torch.int),
                                 torch.zeros_like(to_update, dtype=torch.int))
            self.input_indices = self.input_indices + offset
        except Exception as e:
            raise e
        
    def __actually_query__(self, state: torch.Tensor, encoded: torch.Tensor, mask: torch.Tensor, update_output: Optional[torch.Tensor] = None,
                           update_input: Optional[torch.Tensor] = None):
        from allennlp.nn.util import weighted_sum

        assert isinstance(update_output, torch.Tensor)
        assert isinstance(update_input, torch.Tensor)
        self.update_input_buffer(update_input)
        self.update_output_buffer(update_output)
        output_mask = (self.output_buffer != self.pad)
        output_prefix_queried = self.output_gru(inputs=self.embedder(self.output_buffer), mask=output_mask)
        # output_weighted = output_prefix_queried[:, -1, :]
        output_weights = self.output_attn(vector=state, matrix=output_prefix_queried, matrix_mask=output_mask)
        output_weighted = weighted_sum(output_prefix_queried, output_weights)
        monotonic_input = self.monotonic_input_encoded[torch.arange(state.shape[0]), self.input_indices]
        return torch.cat([output_weighted, monotonic_input], dim=1)

    def get_output_dim(self) -> int:
        return 2 * self.hid_dim * 2