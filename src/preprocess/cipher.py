import os
import hydra
import logging
from omegaconf import DictConfig
from src.util.preprocess_util import Vocab
from src.preprocess.preprocess import Preprocess
from tqdm import tqdm
from itertools import islice

logger = logging.getLogger("PreprocessNPZ")


class PreprocessNPZ(Preprocess):
    def __init__(self) -> None:
        super().__init__()

    def serialize_z(self) -> None:
        """serialize the abstract mfst machine"""
        if os.path.exists(self.cfg.z_path):
            return
        # TODO: clean up matrices related functions below
        # tau = Utils.far_into_mfst(base_fst, phones_sym,
        #                                 covering=covering, panphon=panphon,
        #                                 syllabic=syllabic, i_range=i_range, unicode_for_output=unicode_for_output,
        #                                 wfst=wfst, i_o_weights=i_o_weights, no_substitution=no_substitution,
        #                                 loaded_machine=loaded_machine, wfst_machine=wfst_machine
        #                                 )
        # matrices, z_wfst_start_states, \
        # z_wfst_exit_states, lifted, \
        # local_state_matrices = self.composed_to_matrices(bos=bos,
        #                                                         composed=tau,
        #                                                         eos=eos, pad=pad,
        #                                                         vocab_size=vocab_size,
        #                                                         length_prior_weight=length_prior_weight,
        #                                                         fst_fname=z_name,
        #                                                         serialize_mfst=True)

        # if wfst:
        #     with open(f'{z_name}.wfst_states.pkl', mode='wb') as fh:
        #         pickle.dump({'z_start': z_wfst_start_states, 'z_exit': z_wfst_exit_states}, fh,
        #                     protocol=pickle.HIGHEST_PROTOCOL, )
        # assert local_state_matrices is not None
        # self.savez_compressed(z_name, z_emission=matrices[0], z_transition=matrices[1],
        #                 local_state_matrices_emission=local_state_matrices[0],
        #                 local_state_matrices_transition=local_state_matrices[1],
        #               )

    def serialize(self) -> None:
        """serialize mfst objects with pickle to be loaded by trainer"""
        # global global_skip_bad_io_pairs
        # from sys import stderr
        # (base_fst, phones_sym, input_range, bos, eos, pad, covering, panphon, syllabic, unicode_for_output,
        #  wfst, i_o_weights, no_substitution, vocab_mapping, loaded_machine, wfst_machine, work) = params

        # tau = Utils.far_into_mfst(base_fst, phones_sym,
        #                           covering=covering, panphon=panphon, syllabic=syllabic,
        #                           i_range=input_range, unicode_for_output=unicode_for_output, wfst=wfst,
        #                           i_o_weights=i_o_weights, no_substitution=no_substitution,
        #                           loaded_machine=loaded_machine, wfst_machine=wfst_machine,
        #                           )
        # Vocab.load(vocab_mapping)
        # Vocab.freeze()
        # vocab_size = Vocab.size()
        # for w in work:
        #     gs, name, ps, dataset_split = w
        #     print(f'serializing {name}.npz', file=stderr)
        #     from os.path import exists
        #     gs_encoded = None
        #     ps_encoded = None
        #     try:
        #         if exists(f'{name}.npz'):
        #             from numpy import load
        #             try:
        #                 loaded = load(f'{name}.npz')
        #                 del loaded
        #                 continue
        #             except Exception as e:
        #                 print(f're serializing {name}.npz')
        #         # TODO: changed for cipher, might cause bug for g2p. I replaced most ord(_) with string mapper
        #         x = Utils.create_from_string([tau._string_mapper[0](_) for _ in gs],
        #                                      semiring_class=tau.semiring)
        #         y = Utils.create_from_string(ps,
        #                                      semiring_class=tau.semiring)
        #         import numpy as np

        #         gs_encoded = np.array(Utils.transliterate([tau._string_mapper[0](_) for _ in gs]) + [eos, ], )
        #         ps_encoded = np.array(Utils.transliterate([tau._string_mapper[1](_) for _ in ps]) + [eos,])

        #         xT = x.compose(tau)
        #         assert xT.num_states > 0, f'{gs}\t{gs_encoded}'
        #         xTy = xT.compose(y)
        #         xTy._string_mapper = tau._string_mapper[1]
        #         assert xTy.num_states > 0, f'{xT.num_states}\t{gs}\t{ps}\t{[ord(_) for _ in gs]}\t{[tau._string_mapper[1](_) for _ in ps]}'
        #         matrices, clamped_wfst_start_states, \
        #         clamped_wfst_exit_states, \
        #         lifted, _ = PreprocessNPZ.composed_to_matrices(bos, xTy, eos, pad, vocab_size,
        #                                                        fst_fname=f'{name}.{dataset_split}.clamped.fst',
        #                                                        serialize_mfst=True)
        #         free_matrices, \
        #         free_wfst_start_states, \
        #         free_wfst_exit_states, \
        #         free_lifted, \
        #         _ = PreprocessNPZ.composed_to_matrices(bos, xT, eos, pad, vocab_size,
        #                                                fst_fname=f'{name}.{dataset_split}.free.fst',
        #                                                serialize_mfst=True)

        #         savez_compressed(f'{name}.npz', num_emission=matrices[0], num_transition=matrices[1],
        #                          denom_emission=free_matrices[0], denom_transition=free_matrices[1],
        #                          gs=gs_encoded, ps=ps_encoded)
        #         import pickle
        #         if wfst:
        #             with open(f'{name}.wfst_states.pkl', mode='wb') as fh:
        #                 pickle.dump({'free_start': free_wfst_start_states,
        #                              'clamped_start': clamped_wfst_start_states,
        #                              'free_exit': free_wfst_exit_states,
        #                              'clamped_exit': clamped_wfst_exit_states}, fh,
        #                             protocol=pickle.HIGHEST_PROTOCOL,)
        #     except Exception as e:
        #         if global_skip_bad_io_pairs:
        #             print(e)
        #             print(f'warning: {name}.npz cannot be serialized. gs: {gs}\tps: {ps}')
        #         else:
        #             raise e
        raise NotImplementedError

    def build_vocab(self) -> None:
        """build vocab object based on marks in config as well as mapping values"""
        Vocab.set_non_reserved_offset(self.cfg.reserved_vocab_size)
        if self.cfg.reserved_syllabic:
            syllabic_marks = [
                "syllable-start",
                "syllable-end",
                "syllable-onset",
                "syllable-nucleus",
                "syllable-coda",
            ]
            for m in syllabic_marks:
                Vocab.add_word(m, is_reserved=True)
            for m in self.output_mapping.values():
                Vocab.add_word(m, is_reserved=True)

        for _ in [
            self.cfg.bos,
            self.cfg.eos,
            self.cfg.pad,
        ] + list(self.cfg.control_symbols):
            Vocab.add_word(_)

    def compose_to_matrices(self):
        pass

    # def serialize_under_mpp(self, buckets, limit, mpp, serialize_prefix):
    #     for dataset_split in ("dev", "train"):
    #         file_path = f"{self.prefix}/{dataset_split}/{self.lang}_{dataset_split}.tsv"
    #         tsv_file = pd.read_csv(
    #             file_path, sep="\t", header=None, names=["grapheme", "phoneme"]
    #         )
    #         for r_idx, r in enumerate(tqdm(islice(tsv_file.iterrows(), limit))):
    #             graphemes = r[1]["grapheme"]
    #             phonemes = r[1]["phoneme"]
    #             gs, name, ps = self.tokenize_and_get_name(
    #                 graphemes, phonemes, serialize_prefix
    #             )
    #             b_idx = r_idx % len(buckets)
    #             buckets[b_idx][-1].append((gs, name, ps, dataset_split))
    #         if len(buckets) == 1:
    #             self.serialize(buckets[0])
    #         else:
    #             if self.bucket_id == -1:
    #                 mpp.map(self.serialize, buckets)
    #             else:
    #                 self.serialize(buckets[self.bucket_id])

    # def tokenize_and_get_name(self, graphemes, phonemes, serialize_prefix, **kwargs):
    #     from modules.util import Utils

    #     p2g = False
    #     if "p2g" in kwargs:
    #         p2g = kwargs["p2g"]
    #     gs, name, ps, g, p = Utils.tokenize_and_get_name(
    #         graphemes,
    #         phonemes,
    #         serialize_prefix,
    #         self.char_tokenizer,
    #         self.tokenizer,
    #         self.output_mapping,
    #         p2g=p2g,
    #     )
    #     return gs, name, ps


@hydra.main(config_path="../conf", config_name="preprocess_npz")
def main(config: DictConfig):
    logger.info(config)
    Preprocess(config)


if __name__ == "__main__":
    main()
