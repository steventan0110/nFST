from typing import Tuple, FrozenSet, Container, Set, Dict

import mfst
from bidict import bidict
from src.util.preprocess_util import Utils


class OwnAST:
    cache = dict()
    empty = mfst.FST(
        acceptor=True, semiring_class=mfst.BooleanSemiringWeight
    ).create_from_string([])

    def __init__(self, op, arg1, arg2):
        self.op = op
        self.arg1 = arg1
        self.arg2 = arg2
        self.__fsa_cache = None
        self.cached_hash = None

    @property
    def well_defined(self) -> bool:
        if self.op == "v":
            return (
                (
                    isinstance(self.arg1, int)
                    or isinstance(self.arg1, tuple)
                    or isinstance(self.arg1, list)
                    or isinstance(self.arg1, str)
                )
                and (self.arg2 is None)
            ) or (self.arg1 is None and (self.arg2 is None))
        elif self.op in {"+", "*", "pow"}:
            return isinstance(self.arg1, OwnAST) and isinstance(self.arg2, OwnAST)
        return False

    def __eq__(self, other):
        return self.fsa == other.fsa

    def __hash__(self):
        if self.cached_hash is None:
            self.cached_hash = hash((self.op, self.arg1, self.arg2))
        return self.cached_hash

    @property
    def fsa(self):
        if self.__fsa_cache is None:
            assert self.well_defined
            if hash(self) in OwnAST.cache:
                self.__fsa_cache = OwnAST.cache[hash(self)]
            else:
                try:
                    self.__fsa_cache = OwnAST.build_unweighted_fsa_helper(
                        self, OwnAST.fst
                    )
                    OwnAST.cache[hash(self)] = self.__fsa_cache
                except Exception as e:
                    raise e
        return self.__fsa_cache

    @staticmethod
    def add_component(main: mfst.FST, to_add: mfst.FST, output_only: bool = False):
        """
        add to_add as a disconnected component to main
        :param output_only:
        :param main:
        :param to_add:
        :return:
        """
        if output_only:
            # if output_only the purpose of this function is to build an (i,o)-mark FST.
            # and therefore the input to_add must be an unweighted FSA
            assert to_add._acceptor is True
        from bidict import bidict

        num_disconnected_states = to_add.num_states
        mapping = bidict()
        exiting_state = main.add_state()
        for s in range(num_disconnected_states):
            mapping[s] = main.add_state()

        for s in range(num_disconnected_states):
            arcs = to_add.get_arcs(s)
            for a in arcs:
                if a.nextstate == -1:
                    # accepting arc.
                    # add new state to main

                    # if output_only, we accept epsilon and output input_label
                    if output_only:
                        main.add_arc(
                            mapping[s], exiting_state, a.weight, 0, a.output_label
                        )
                    else:
                        main.add_arc(
                            mapping[s],
                            exiting_state,
                            a.weight,
                            a.input_label,
                            a.output_label,
                        )
                else:
                    new_nextstate = mapping[a.nextstate]
                    # check if output_only
                    if output_only:
                        main.add_arc(
                            mapping[s], new_nextstate, a.weight, 0, a.output_label
                        )
                    else:
                        main.add_arc(
                            mapping[s],
                            new_nextstate,
                            a.weight,
                            a.input_label,
                            a.output_label,
                        )

        return mapping[to_add.initial_state], exiting_state

    @staticmethod
    def mfst_weight_projection(
        inp: mfst.FST, minimize: bool = True, vocab: bidict = None
    ):
        """
        basically we are going to create a new FSA, where the machines contain all edge-machines on inp
        :param vocab:
        :param inp: MFST
        :param minimize: unweighted FSA
        :return:
        """
        if vocab is None:
            strip_io_symbols = True
        else:
            strip_io_symbols = False
        from tqdm import tqdm

        to_return = mfst.FST(
            acceptor=strip_io_symbols, semiring_class=mfst.BooleanSemiringWeight
        )
        num_states = inp.num_states
        old_new_mapping = bidict()
        for s in range(num_states):
            old_new_mapping[s] = to_return.add_state()

        for s in tqdm(range(num_states)):
            # get all arcs
            for a in inp.get_arcs(s):
                weight: OwnAST = a.weight.value
                nextstate: int = (
                    a.nextstate
                )  # no special handling is needed for nextstate as we don't dwelve into it
                machine = weight.fsa

                # add machine to our main stuff
                entry, exit = OwnAST.add_component(
                    to_return, machine, output_only=not strip_io_symbols
                )
                # TODO connect s to entry, and exits to nextstate
                to_return.add_arc(
                    old_new_mapping[s],
                    entry,
                )
                if nextstate == -1:
                    to_return.set_final_weight(exit)
                else:
                    if not strip_io_symbols:
                        i = vocab[(a.input_label, a.output_label)]
                        o = 0
                    else:
                        i = 0
                        o = 0
                    to_return.add_arc(
                        exit, old_new_mapping[nextstate], input_label=i, output_label=o
                    )
        to_return.set_initial_state(old_new_mapping[inp.initial_state])
        if minimize:
            to_return = to_return.remove_epsilon().determinize()
            if strip_io_symbols:
                to_return = to_return.minimize()
        return to_return

    @staticmethod
    def build_unweighted_fsa_helper(inp: "OwnAST", fst: mfst.FST) -> mfst.FST:
        if hash(inp) in OwnAST.cache:
            return OwnAST.cache[hash(inp)]
        if inp.op == "v":
            if inp.arg1 is None:
                to_return = mfst.FST(
                    semiring_class=mfst.BooleanSemiringWeight, acceptor=True
                )
                to_return.add_state()
                to_return.set_initial_state(0)
                OwnAST.cache[hash(inp)] = to_return
                return to_return
            if isinstance(inp.arg1, int):
                l = [
                    inp.arg1,
                ]
                to_return = fst.create_from_string(l)
                OwnAST.cache[hash(inp)] = to_return
                return to_return
            elif (
                isinstance(inp.arg1, tuple)
                or isinstance(inp.arg1, list)
                or isinstance(inp.arg1, str)
            ):
                to_return = Utils.create_from_string(inp.arg1)
                OwnAST.cache[hash(inp)] = to_return.minimize()
                return to_return
            else:
                raise NotImplementedError
        elif inp.op == "+":
            to_return = (
                OwnAST.build_unweighted_fsa_helper(inp.arg1, fst)
                .union(OwnAST.build_unweighted_fsa_helper(inp.arg2, fst))
                .remove_epsilon()
                .determinize()
                .minimize()
            )
            OwnAST.cache[hash(inp)] = to_return
            return to_return
        elif inp.op == "*":
            to_return = (
                OwnAST.build_unweighted_fsa_helper(inp.arg1, fst)
                .concat(OwnAST.build_unweighted_fsa_helper(inp.arg2, fst))
                .remove_epsilon()
                .determinize()
                .minimize()
            )
            OwnAST.cache[hash(inp)] = to_return
            return to_return
        elif inp.op == "pow":
            base = OwnAST.build_unweighted_fsa_helper(inp.arg1, fst)
            to_return = OwnAST.empty
            for r in range(inp.arg2):
                to_return = to_return.concat(base)
            OwnAST.cache[hash(inp)] = (
                to_return.remove_epsilon().determinize().minimize()
            )
            return to_return
        else:
            raise NotImplementedError

    @staticmethod
    def build_unweighted_fsa(
        inp: mfst.FST,
        bos="<s> ",
        eos=" </s>",
        pad=" <pad>",
        final_loop: bool = True,
        optimize: bool = True,
    ) -> mfst.FST:
        bos_machine = inp.create_from_string(bos)

        eos_machine = inp.create_from_string(eos)
        if final_loop:
            pad_machine = inp.create_from_string(pad).closure()
            eos_machine = eos_machine.concat(pad_machine)

        built = inp
        to_return = bos_machine.concat(built).concat(eos_machine)
        if optimize:
            return to_return.remove_epsilon().determinize().minimize()
        else:
            return to_return

    @staticmethod
    def fst_weight_op(fst, weight_fn):
        new_fst = mfst.FST(semiring_class=RefWeight)
        fst: mfst.FST
        for _ in fst.states:
            new_fst.add_state()
            new_fst.set_final_weight(_, fst.get_final_weight(_))
        new_fst.set_initial_state(fst.initial_state)
        for _ in fst.states:
            for arc in fst.get_arcs(_):
                if arc.nextstate != -1:
                    new_fst.add_arc(
                        from_state=_,
                        to_state=arc.nextstate,
                        input_label=arc.input_label,
                        output_label=arc.output_label,
                        weight=RefWeight(weight_fn(arc.weight.value)),
                    )
        return new_fst

    @staticmethod
    def empty_fst(weight):
        new_fst = mfst.FST(semiring_class=RefWeight)
        init = new_fst.add_state()
        final = new_fst.add_state()
        new_fst.set_initial_state(init)
        new_fst.set_final_weight(final, RefWeight.one)
        new_fst.add_arc(init, final, RefWeight(weight))
        return new_fst

    @staticmethod
    def mfst_to_fst(fst: mfst.FST, vocab, minimize: bool = True):
        # just a thin wrapper for mfst_weight_projection
        return OwnAST.mfst_weight_projection(fst, vocab=vocab, minimize=minimize)

    @staticmethod
    def fst_to_mfst(fst: mfst.FST, vocab: bidict):
        """
        inverse op of mfst_to_fst. assumes that the input tape of fst consists of tuples of actual i-o symbols.
        :param fst:
        :param vocab:
        :return:
        """
        from tqdm import tqdm

        new_fst = mfst.FST(semiring_class=RefWeight)
        old_new_mapping = bidict()
        for s in fst.states:
            new_s = new_fst.add_state()
            old_new_mapping[s] = new_s
        new_fst.set_initial_state(old_new_mapping[fst.initial_state])
        for s in tqdm(fst.states):
            if fst.get_final_weight(s) == fst.semiring_one:
                new_fst.set_final_weight(old_new_mapping[s], RefWeight.one)
            for a in fst.get_arcs(s):
                if a.nextstate == -1:
                    continue
                mark_sym = a.output_label
                if mark_sym == 0:
                    out_mark_sym = ()
                else:
                    out_mark_sym = (mark_sym,)
                if a.input_label == 0:
                    i, o = 0, 0
                else:
                    i, o = vocab.inverse[a.input_label]
                new_fst.add_arc(
                    s,
                    old_new_mapping[a.nextstate],
                    RefWeight(OwnAST("v", out_mark_sym, None)),
                    i,
                    o,
                )
        return new_fst


OwnAST.fst = mfst.FST(mfst.BooleanSemiringWeight)


class FeatureVectorWeight(mfst.AbstractSemiringWeight):
    import numpy as np

    def __init__(self, v: np.ndarray):
        import numpy as np

        assert isinstance(v, np.ndarray)
        self.__value = v

    def __add__(self, other: "FeatureVectorWeight"):
        import numpy as np

        v: np.ndarray = self.__value

        return FeatureVectorWeight(np.logaddexp(v, other.__value))

    def __mul__(self, other: "FeatureVectorWeight"):
        return FeatureVectorWeight(self.__value + other.__value)

    def __div__(self, other: "FeatureVectorWeight"):
        return FeatureVectorWeight(self.__value - other.__value)

    def __pow__(self, n):
        return FeatureVectorWeight(self.__value * n)

    def approx_eq(self, other, delta):
        import numpy as np

        return np.allclose(self.__value, other.__value, delta)

    def __hash__(self):
        return hash(self.__value)

    def __eq__(self, other: "FeatureVectorWeight"):
        import numpy as np

        return bool(np.all(self.__value == other.__value))

    def __truediv__(self, other):
        return FeatureVectorWeight(self.__value // other.__value)

    def openfst_str(self):
        return super().openfst_str()

    def __repr__(self):
        return self.__value.__repr__()

    def __bool__(self):
        return self.__value.__bool__()


FeatureVectorWeight.zero = None
FeatureVectorWeight.one = None


class RefWeight(mfst.AbstractSemiringWeight):
    @property
    def value(self):
        return self.__value

    def __init__(self, v: OwnAST) -> None:
        super(RefWeight, self).__init__()
        self.__value = v

    def __add__(self, other: "RefWeight"):
        return RefWeight(OwnAST("+", self.__value, other.__value))

    def __mul__(self, other):
        return RefWeight(OwnAST("*", self.__value, other.__value))

    def __div__(self, other):
        """
        not implemented!
        :param other:
        :return:
        """
        return super().__div__(other)

    def __pow__(self, n):
        return RefWeight(OwnAST("pow", self.__value, n))

    def member(self):
        return self.__value.well_defined

    def quantize(self, delta=0.5):
        return super().quantize(delta)

    def reverse(self):
        return super().reverse()

    def sampling_weight(self):
        return 1.0

    def approx_eq(self, other, delta):
        return self == other

    def __hash__(self):
        return hash(self.__value)

    def __eq__(self, other: "RefWeight"):
        return self.__value == other.__value

    def __truediv__(self, other):
        return super().__truediv__(other)

    def openfst_str(self):
        return super().openfst_str()

    def __repr__(self):
        return super().__repr__()

    def __bool__(self):
        return super().__bool__()

    @staticmethod
    def convert(w):
        pass


RefWeight.zero = RefWeight(OwnAST("v", None, None))
RefWeight.one = RefWeight(OwnAST("v", 0, None))


class PathWeight(mfst.AbstractSemiringWeight):
    def __div__(self, other: "PathWeight"):
        return super().__div__(other)

    def __pow__(self, n):
        s = self
        for i in range(n):
            s = s * self
        return s

    def member(self):
        return super().member()

    def quantize(self, delta=0.5):
        return super().quantize(delta)

    def reverse(self):
        return super().reverse()

    def sampling_weight(self):
        return super().sampling_weight()

    def approx_eq(self, other: "PathWeight", delta):
        return self == other

    def __hash__(self):
        return hash(self._value)

    def openfst_str(self):
        return super().openfst_str()

    def __add__(self, other: "PathWeight"):
        own_container: FrozenSet[Tuple] = self._value
        return PathWeight(own_container | other._value)

    def __mul__(self, other: "PathWeight"):
        new_container = set()
        for s in self._value:
            for t in other._value:
                new_container.add(tuple(s + t))
        return PathWeight(new_container)

    def __init__(self, v: Container[Tuple]) -> None:
        super().__init__()
        self._value = frozenset(v)

    def __eq__(self, other: "PathWeight"):
        return self._value == other._value

    def __repr__(self):
        return str(self._value)

    @staticmethod
    def build_unweighted(m: mfst.FST, bos=1, eos=2, pad=0) -> mfst.FST:
        bos_machine = mfst.FST(acceptor=True, semiring_class=mfst.BooleanSemiringWeight)
        bos_machine.add_state()
        bos_machine.set_initial_state(0)
        bos_machine.add_state()
        bos_machine.add_arc(0, 1, input_label=bos)
        bos_machine.set_final_weight(1)

        eos_machine = mfst.FST(acceptor=True, semiring_class=mfst.BooleanSemiringWeight)
        eos_machine.add_state()
        eos_machine.set_initial_state(0)
        eos_machine.add_state()
        eos_machine.add_arc(0, 1, input_label=eos)
        eos_machine.add_arc(1, 1, input_label=pad)
        eos_machine.set_final_weight(1)

        to_return = mfst.FST(acceptor=True, semiring_class=mfst.BooleanSemiringWeight)
        for _ in m.states:
            to_return.add_state()

        for s in m.states:
            to_return.add_state()
            for arc in m.get_arcs(
                s,
            ):
                nextstate = arc.nextstate

                set_of_tuples: FrozenSet[Tuple] = arc.weight._value
                if nextstate != -1:
                    for t in set_of_tuples:
                        current_state = s

                        for symbol in t:
                            new_state = to_return.add_state()
                            if symbol != "":
                                to_return.add_arc(
                                    current_state, new_state, input_label=symbol
                                )
                            else:
                                to_return.add_arc(
                                    current_state,
                                    new_state,
                                )
                            current_state = new_state
                        to_return.add_arc(current_state, nextstate)
            if m.get_final_weight(s) == PathWeight.one:
                to_return.set_final_weight(s)
        to_return.set_initial_state(m.initial_state)

        to_return = mfst.FST.concat(bos_machine, to_return).concat(eos_machine)
        return to_return.remove_epsilon().determinize().minimize()


PathWeight.zero = PathWeight(set())
PathWeight.one = PathWeight({("",)})
