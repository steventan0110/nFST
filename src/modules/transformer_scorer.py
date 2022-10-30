import torch
from torch import nn
from torch.nn.init import xavier_normal_
from torch.functional import F
import math
from transformers import GPT2LMHeadModel, GPT2Config


class GPTScorer:
    def __init__(self, args) -> None:
        super().__init__()
        self.vocab_size = args.vocab_size
        self.bos = args.bos
        self.eos = args.eos
        self.pad = args.pad
        self.max_length = args.max_length
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.config = GPT2Config(
            args.vocab_size,
            n_head=args.num_heads,
            n_layer=args.num_layers,
            bos_token_id=self.bos,
            eos_token_id=self.eos,
            n_positions=args.max_length + 1,
            n_embd=args.hid_dim,
        )

        self.model = GPT2LMHeadModel(self.config)
        self.emission, self.transition = None, None

    def sample(self):
        pass

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


class TransformerSampler:
    def __init__(self, model) -> None:
        self.model = model

    def set_masks(self, transition, emission):
        assert isinstance(self.model, GPTScorer)
        self.model.set_masks(emission=emission, transition=transition)

    def set_k(self, k: int):
        self.model.set_k(k)

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

    def sample(self, batch_size, query_args):
        gs_expand = query_args["to_encode"]
        ps_expand = query_args["to_encode_2"]

        inp = gs_expand.new_full((batch_size, 1), self.bos)
        prefixes = []
        log_probs = []
        for timestep in range(model.max_length + 1):
            hx = self.model(inp).last_hidden_state
            print(hx)
            # sample next token using hx and store its logprob

            inp = new_symbol
            # add attention mask based on padding

        if len(prefix) > 0:
            prefix.pop(-1)
        stacked_log_probs = torch.stack(log_probs, dim=0)
        del log_probs
        prefixes_stacked = torch.stack(prefixes, dim=1)
        del prefixes
        summed_log_probs = stacked_log_probs.sum(dim=0)
        return summed_log_probs, prefixes_stacked
