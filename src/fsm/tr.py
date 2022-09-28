from functools import lru_cache
from mfst import FST
from typing import List
import pynini
from src.modules.path_semiring import RefWeight, OwnAST
from src.modules.nros import NRO
from src.util.preprocess_util import Vocab, Utils
from os.path import exists
import logging

logger = logging.getLogger("TR_FST")


class TR:
    """Implementation of Cipher preprocessor"""

    def __init__(
        self,
        mapping_path,
        input_path,
        vocab_path,
        control_symbols,
        bos,
        eos,
        pad,
        language,
    ):
        logger.info(f"Build TR FST for {language}")
        self.m = Utils.load_mapping(mapping_path)
        self.input_path = input_path
        self.language = language
        self.control_symbols = control_symbols
        self.bos = bos
        self.eos = eos
        self.pad = pad

        self.make_vocab(vocab_path)
        self.fst = self._make_fst()
        self.fst_sub = self._make_fst(add_sub=True)
        # self.compose_fst = self._make_compose_fst()
        # self.compose_reg_fst = self._make_compose_reg_fst()
        # self.fst_xy = self.make_fst_xy()
        # self.fst_yx = self.make_fst_yx()

    def get_machine(self, machine_type):
        if machine_type == "base":
            return self.fst
        elif machine_type == "sub":
            return self.fst_sub

    @lru_cache(maxsize=None)
    def input_symbols(self):
        eng_set = set()
        with open(self.input_path, "r") as f:
            symbols = [_.strip().split()[0] for _ in f]
            for s in symbols:
                if not s.startswith("made"):
                    eng_set.add(s)
        return eng_set

    @lru_cache(maxsize=None)
    def output_symbols(self):
        return set([_ for _ in self.m.keys() if _ != 0])

    def _make_compose_fst(self):
        g2p_machine = self.make_g2p_machine()
        p2g_machine = self.make_p2g_machine()
        self.g2p_machine = g2p_machine
        return NRO.nro_compose_m(
            g2p_machine,
            p2g_machine,
            Vocab.lookup("cp1"),
            Vocab.lookup("cp2"),
            Vocab.lookup("cpstart"),
            Vocab.lookup("cpend"),
        )

    def _make_compose_reg_fst(self):
        g2g_machine = self.fst
        # store in self for future pretrain's use
        self.p_acceptor = self.make_p_fst()
        return NRO.nro_compose_m(
            g2g_machine,
            self.p_acceptor,
            Vocab.lookup("cp1"),
            Vocab.lookup("cp2"),
            Vocab.lookup("cpstart"),
            Vocab.lookup("cpend"),
        )

    def make_g2p_machine(self):
        """g->p machine from english grapheme to english phoneme"""
        input_mark = Vocab.lookup("input-mark")
        output_mark = Vocab.lookup("phoneme-mark")
        fst = FST(semiring_class=RefWeight)
        init_state = fst.add_state()
        fst.set_initial_state(init_state)
        fst.set_final_weight(init_state)
        for y_values in self.phonemes():
            # insertion case
            label = tuple([output_mark, Vocab.lookup(y_values)])
            fst.add_arc(
                init_state,
                init_state,
                input_label=0,
                output_label=Vocab.lookup(y_values),
                weight=RefWeight(OwnAST("v", label, None)),
            )
        for x_values in self.input_symbols():
            # deletion case
            label = tuple([input_mark, Vocab.lookup(x_values)])
            fst.add_arc(
                init_state,
                init_state,
                input_label=Vocab.lookup(x_values),
                output_label=0,
                weight=RefWeight(OwnAST("v", label, None)),
            )
        return fst

    def make_p2g_machine(self):
        input_mark = Vocab.lookup("phoneme-mark")
        output_mark = Vocab.lookup("output-mark")
        fst = FST(semiring_class=RefWeight)
        init_state = fst.add_state()
        fst.set_initial_state(init_state)
        fst.set_final_weight(init_state)
        for y_values in self.m.keys():
            # insertion case
            label = tuple([output_mark, Vocab.lookup(y_values)])
            fst.add_arc(
                init_state,
                init_state,
                input_label=0,
                output_label=Vocab.lookup(y_values),
                weight=RefWeight(OwnAST("v", label, None)),
            )
        for x_values in self.phonemes():
            # deletion case
            label = tuple([input_mark, Vocab.lookup(x_values)])
            fst.add_arc(
                init_state,
                init_state,
                input_label=Vocab.lookup(x_values),
                output_label=0,
                weight=RefWeight(OwnAST("v", label, None)),
            )
        # explicitly add language mark
        fst.add_arc(
            init_state,
            init_state,
            input_label=0,
            output_label=Vocab.lookup("<ur>"),
            weight=RefWeight(
                OwnAST("v", tuple([output_mark, Vocab.lookup("<ur>")]), None)
            ),
        )
        fst.add_arc(
            init_state,
            init_state,
            input_label=0,
            output_label=Vocab.lookup("<sd>"),
            weight=RefWeight(
                OwnAST("v", tuple([output_mark, Vocab.lookup("<sd>")]), None)
            ),
        )
        return fst

    def make_fst_xy(self):
        input_mark = Vocab.lookup("input-mark")
        output_mark = Vocab.lookup("output-mark")
        fst = FST(semiring_class=RefWeight)
        init_state = fst.add_state()
        fst.set_initial_state(init_state)
        second_state = fst.add_state()
        fst.set_final_weight(second_state)

        for x_values in self.input_symbols():
            # deletion case
            label = tuple([input_mark, Vocab.lookup(x_values)])
            fst.add_arc(
                init_state,
                init_state,
                input_label=Vocab.lookup(x_values),
                output_label=0,
                weight=RefWeight(OwnAST("v", label, None)),
            )

        fst.add_arc(init_state, second_state, input_label=0, output_label=0)

        for y_values in self.m.keys():
            # insertion case
            label = tuple([output_mark, Vocab.lookup(y_values)])
            fst.add_arc(
                second_state,
                second_state,
                input_label=0,
                output_label=Vocab.lookup(y_values),
                weight=RefWeight(OwnAST("v", label, None)),
            )
        return fst

    def make_fst_yx(self):
        input_mark = Vocab.lookup("input-mark")
        output_mark = Vocab.lookup("output-mark")
        fst = FST(semiring_class=RefWeight)
        init_state = fst.add_state()
        fst.set_initial_state(init_state)
        second_state = fst.add_state()
        fst.set_final_weight(second_state)

        for y_values in self.m.keys():
            # insertion case
            label = tuple([output_mark, Vocab.lookup(y_values)])
            fst.add_arc(
                init_state,
                init_state,
                input_label=0,
                output_label=Vocab.lookup(y_values),
                weight=RefWeight(OwnAST("v", label, None)),
            )
        fst.add_arc(init_state, second_state, input_label=0, output_label=0)
        for x_values in "abcdefghijklmnopqrstuvwxyz":
            # deletion case
            label = tuple([input_mark, Vocab.lookup(x_values)])
            fst.add_arc(
                second_state,
                second_state,
                input_label=Vocab.lookup(x_values),
                output_label=0,
                weight=RefWeight(OwnAST("v", label, None)),
            )
        return fst

    def make_vocab(self, vocab_path):
        if exists(vocab_path):
            return Vocab.load(vocab_path)
        else:
            for _ in [
                self.bos,
                self.eos,
                self.pad,
            ] + list(self.control_symbols):
                Vocab.add_word(_)
            for c in "abcdefghijklmnopqrstuvwxyz":
                Vocab.add_word(c)
            for v in self.m.keys():
                Vocab.add_word(v)
            for p in self.phonemes():
                Vocab.add_word(p)
            eng_set = self.input_symbols()
            for eng_sym in eng_set:
                Vocab.add_word(eng_sym)
            # hardcode language marks
            Vocab.add_word("<ur>")
            Vocab.add_word("<sd>")
            Vocab.add_word("phoneme-mark")
            Vocab.dump(vocab_path)
            Vocab.freeze()

    def make_p_fst(self):
        fst = FST(semiring_class=RefWeight)
        init_state = fst.add_state()
        second_state = fst.add_state()
        fst.set_initial_state(init_state)
        fst.set_final_weight(second_state)
        for y_values in self.m.keys():
            # insertion case
            label = tuple(
                [
                    Vocab.lookup(y_values),
                ]
            )
            fst.add_arc(
                init_state,
                second_state,
                input_label=Vocab.lookup(y_values),
                output_label=Vocab.lookup(y_values),
                weight=RefWeight(OwnAST("v", label, None)),
            )
        # add special label for filtered dataset, only one transition is possible for every sentence pair
        fst.add_arc(
            init_state,
            second_state,
            input_label=0,
            output_label=Vocab.lookup("<ur>"),
            weight=RefWeight(
                OwnAST(
                    "v",
                    tuple(
                        [
                            Vocab.lookup("<ur>"),
                        ]
                    ),
                    None,
                )
            ),
        )
        fst.add_arc(
            init_state,
            second_state,
            input_label=0,
            output_label=Vocab.lookup("<sd>"),
            weight=RefWeight(
                OwnAST(
                    "v",
                    tuple(
                        [
                            Vocab.lookup("<sd>"),
                        ]
                    ),
                    None,
                )
            ),
        )
        return fst.closure()

    def _make_fst(self, add_sub=False):
        input_mark = Vocab.lookup("input-mark")
        output_mark = Vocab.lookup("output-mark")

        fst = FST(semiring_class=RefWeight)
        init_state = fst.add_state()
        fst.set_initial_state(init_state)
        fst.set_final_weight(init_state)
        for y_values in self.m.keys():
            # insertion case
            label = tuple([output_mark, Vocab.lookup(y_values)])
            fst.add_arc(
                init_state,
                init_state,
                input_label=0,
                output_label=Vocab.lookup(y_values),
                weight=RefWeight(OwnAST("v", label, None)),
            )
        # add special label for filtered dataset
        fst.add_arc(
            init_state,
            init_state,
            input_label=0,
            output_label=Vocab.lookup("<ur>"),
            weight=RefWeight(
                OwnAST("v", tuple([output_mark, Vocab.lookup("<ur>")]), None)
            ),
        )
        fst.add_arc(
            init_state,
            init_state,
            input_label=0,
            output_label=Vocab.lookup("<sd>"),
            weight=RefWeight(
                OwnAST("v", tuple([output_mark, Vocab.lookup("<sd>")]), None)
            ),
        )

        for x_values in self.input_symbols():
            # deletion case
            label = tuple([input_mark, Vocab.lookup(x_values)])
            fst.add_arc(
                init_state,
                init_state,
                input_label=Vocab.lookup(x_values),
                output_label=0,
                weight=RefWeight(OwnAST("v", label, None)),
            )
        if add_sub:
            # explict add substitution path
            for x in self.input_symbols():
                for y in self.m.keys():
                    label = tuple(
                        [
                            Vocab.lookup("insertion-mark"),
                            input_mark,
                            Vocab.lookup(x),
                            output_mark,
                            Vocab.lookup(y),
                        ]
                    )
                    fst.add_arc(
                        init_state,
                        init_state,
                        input_label=Vocab.lookup(x),
                        output_label=Vocab.lookup(y),
                        weight=RefWeight(OwnAST("v", label, None)),
                    )

        return fst

    @lru_cache(maxsize=None)
    def consonants(self):
        return {
            "b",
            "tʃ",
            "d",
            "ð",
            "f",
            "g",
            "h",
            "dʒ",
            "k",
            "l",
            "m",
            "n",
            "ŋ",
            "p",
            "s",
            "ʃ",
            "t",
            "θ",
            "v",
            "w",
            "j",
            "z",
            "ʒ",
            "r",
        }

    @lru_cache(maxsize=None)
    def stress(self):
        return {
            "ˈ",
            "ˌ",
        }

    @lru_cache(maxsize=None)
    def vowels(self):
        return {
            "ɑ",
            "æ",
            "ʌ",
            "ɔ",
            "aʊ",
            "ə",
            "aɪ",
            "ɛ",
            "ɝ",
            "eɪ",
            "ɪ",
            "i",
            "oʊ",
            "ɔɪ",
            "ʊ",
            "u",
            "ː",
        }

    @lru_cache(maxsize=None)
    def phonemes(self):
        # return {'ɑ'} # hard code one phoneme to debug
        eng_symbols = "abcdefghijklmnopqrstuvwxyz"
        eng_set = set([_ for _ in eng_symbols])
        return (
            self.consonants().union(self.vowels()).union(self.stress()).union(eng_set)
        )

    @lru_cache(maxsize=None)
    def syllabic(self):
        return self.vowels().union(self.consonants())


if __name__ == "__main__":
    tr = TR(
        "/home/steven/Code/GITHUB/seq-samplers/tr/nfst/ur.graphs.sym",
        "/home/steven/Code/GITHUB/seq-samplers/tr/nfst/ursd-g2p.vocab",
        (
            "u1",
            "u2",
            "ustart",
            "uend",
            "c1",
            "cstart",
            "cend",
            "cp1",
            "cp2",
            "cpstart",
            "cpend",
            "input-mark",
            "output-mark",
            "insertion-mark",
            "panphon-start",
            "panphon-end",
            "syllable-start",
            "syllable-end",
            "syllable-onset",
            "syllable-nucleus",
            "syllable-coda",
            "wfst-start",
            "wfst-end",
            "wfst#2.",
            "wfst#1.",
            "wfst#.5",
            "wfst#0.",
        ),
        19997,
        19998,
        19999,
        "ur",
    )
    # test composition
    tau = tr.g2p_machine
    g2p_test = "o t t e r b e i n ' s	ˈ ɑ ː t ɝ b a ɪ n z"
    # g2p_test = "r e h a	ˈ r i ː h ə"
    m = Utils.load_mapping(
        "/home/steven/Code/GITHUB/seq-samplers/tr/nfst/ur.graphs.sym"
    )
    en, other = g2p_test.split("\t")
    en = "".join(en.split(" "))
    other = "".join(other.split(" "))
    for item in other:
        if not item in tr.phonemes():
            print(item)
    x = Utils.create_from_string([Vocab.lookup(_) for _ in en])
    y = Utils.create_from_string([Vocab.lookup(_) for _ in other])
    xT = x.compose(tau)
    Ty = tau.compose(y)
    xTy = xT.compose(y)
    print(xT.num_states)
    print(xTy.num_states)
