from typing import Optional
import os
import pickle
from easy_latent_seq.util.preprocess_util import Vocab, dotdict, Utils
from modules.path_semiring import OwnAST
from modules.scorers import FSAGRUScorer
import pynini
from pynini import Fst
from sys import stderr
from os.path import exists
import numpy as np
from numpy import load, savez_compressed
import logging

logger = logging.getLogger("PreprocessBase")


class Preprocess:
    def __init__(self, config) -> None:
        self.cfg = config.preprocess
        self.cfg.vocab = config.vocab

    def serialize_z(self, z_name, length_prior_weight) -> None:
        """serialize the abstract mfst machine"""
        vocab_size = Vocab.size()
        (matrices, _, _, _, local_state_matrices) = self.composed_to_matrices(
            bos=self.bos,
            composed=self.tau,
            eos=self.eos,
            pad=self.pad,
            vocab_size=vocab_size,
            length_prior_weight=length_prior_weight,
            fst_fname=z_name,
            serialize_mfst=True,
        )

        assert local_state_matrices is not None
        savez_compressed(
            z_name,
            z_emission=matrices[0],
            z_transition=matrices[1],
            local_state_matrices_emission=local_state_matrices[0],
            local_state_matrices_transition=local_state_matrices[1],
        )

    def build_vocab(self) -> None:
        """build vocab object based on marks in config as well as mapping values"""
        raise NotImplementedError

    @staticmethod
    def composed_to_matrices(
        bos,
        composed,
        eos,
        pad,
        vocab_size,
        fst_fname: Optional[str] = None,
        length_prior_weight: Optional[float] = None,
        serialize_mfst: bool = False,
        assert_one_path: bool = False,
    ):
        do_wfst = False
        lifted, start_states, exit_states = None, None, None
        composed_sum_paths = OwnAST.mfst_weight_projection(composed, minimize=False)
        unweighted_mark_machine = OwnAST.build_unweighted_fsa(
            composed_sum_paths,
            bos=[
                bos,
            ],
            eos=[
                eos,
            ],
            pad=[
                pad,
            ],
            optimize=False,
        )
        if serialize_mfst:
            unweighted_mark_machine_noloop = OwnAST.build_unweighted_fsa(
                composed_sum_paths,
                bos=[
                    bos,
                ],
                eos=[
                    eos,
                ],
                pad=[
                    pad,
                ],
                optimize=True,
                final_loop=False,
            )

            with open(f"{fst_fname}.mfst.pkl", mode="wb") as fh:
                pickle.dump(unweighted_mark_machine_noloop, fh)
        if length_prior_weight or fst_fname:
            with pynini.default_token_type("byte"):
                unweighted_mark_machine_noloop = OwnAST.build_unweighted_fsa(
                    composed_sum_paths,
                    bos=[
                        bos,
                    ],
                    eos=[
                        eos,
                    ],
                    pad=[
                        pad,
                    ],
                    optimize=False,
                    final_loop=False,
                )
                unweighted_mark_machine_pynini_noloop = Utils.mfst_to_pynini(
                    unweighted_mark_machine_noloop
                ).optimize()
        else:
            unweighted_mark_machine_pynini_noloop = None
        if assert_one_path and unweighted_mark_machine_pynini_noloop is not None:
            path_counter = 0
            unweighted_mark_machine_noloop_min = OwnAST.build_unweighted_fsa(
                composed_sum_paths,
                bos=[
                    bos,
                ],
                eos=[
                    eos,
                ],
                pad=[
                    pad,
                ],
                optimize=True,
                final_loop=False,
            )
            paths = unweighted_mark_machine_noloop_min.iterate_paths()
            buf = []
            for _ in paths:
                path_counter += 1
                p = tuple([Vocab.r_lookup(__) for __ in _.input_path])
                buf.append(p)
            assert path_counter == 1, f"paths bad, {path_counter} {buf[0]} {buf[1]}"
            del buf

        unweighted_mark_machine_pynini = Utils.mfst_to_pynini(
            unweighted_mark_machine
        ).optimize()

        if length_prior_weight is not None:
            locally_normalized_mark_machine_pynini = Utils.length_prior_pynini(
                unweighted_mark_machine_pynini_noloop, length_prior_weight
            )
            locally_normalized_mark_machine_pynini = pynini.push(
                locally_normalized_mark_machine_pynini,
                push_weights=True,
                push_labels=False,
            )
            local_state_mask = FSAGRUScorer.get_state_mask_pynini(
                locally_normalized_mark_machine_pynini, vocab_size, pad, to_numpy=True
            )
        else:
            local_state_mask = None
        if do_wfst:
            lifted, start_states, exit_states = Utils.lift(
                unweighted_mark_machine_pynini,
                [Vocab.lookup("wfst-start")],
                [Vocab.lookup("wfst-end")],
                0,
            )

        if fst_fname is not None:
            # UGLY
            Fst.write(unweighted_mark_machine_pynini_noloop, fst_fname)
        matrices = FSAGRUScorer.get_state_mask_pynini(
            unweighted_mark_machine_pynini, vocab_size, pad, to_numpy=True
        )
        return matrices, start_states, exit_states, lifted, local_state_mask
