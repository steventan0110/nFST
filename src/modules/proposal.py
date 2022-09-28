from src.modules.queries import QueryPiGivenX, QueryPiGivenXAndY
from src.modules.scorers import FSAGRUScorer
from torch.nn.modules import Embedding, ModuleList


class Proposal:
    def __init__(self, args) -> None:
        self.embedding = Embedding(
            args.vocab_size,
            args.hid_dim,
            args.pad,
        )

        self.input_queries = QueryPiGivenX(
            args.hid_dim,
            num_layers=args.num_layers,
            state_hid_dim=args.hid_dim,
            embedder=self.embedding,
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
            embedder=self.embedding,
            vocab_size=args.vocab_size,
            pad=args.pad,
            bos=args.bos,
            eos=args.eos,
            drop=args.dropout,
        )

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
            embeddings=self.embedding,
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
            embeddings=self.embedding,
            query=self.io_queries,
            tied_embeddings=args.tied_proposal_embeddings,
        )

    def get_proposal_module(self):
        return ModuleList(
            [
                self.denom_proposal_dist,
                self.num_proposal_dist,
                self.input_queries,
                self.io_queries,
            ]
        )

    def get_num_proposal_dist(self):
        return self.num_proposal_dist
