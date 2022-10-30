from bidict import bidict
import mfst
from typing import Optional
import pynini
from numpy import load
import numpy as np
import torch
from os.path import exists
import hashlib


class dotdict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Vocab:
    mapping = bidict(
        {
            0: 0,
        }
    )
    freezed = False

    non_reserved_offset = 1

    non_reserved_count = 0
    reserved_count = 1

    @staticmethod
    def set_non_reserved_offset(r: int):
        assert r > 0
        Vocab.non_reserved_offset = r

    @staticmethod
    def add_word(w, is_reserved: bool = False):
        if w not in Vocab.mapping:
            assert not Vocab.freezed, f"{w}"
            # print(f'added to Vocab: {w}')
            if is_reserved:
                assert Vocab.reserved_count < Vocab.non_reserved_offset
                idx = Vocab.reserved_count
                Vocab.reserved_count += 1
            else:
                idx = Vocab.non_reserved_count + Vocab.non_reserved_offset
                Vocab.non_reserved_count += 1
            Vocab.mapping[w] = idx

    @staticmethod
    def lookup(w):
        return Vocab.mapping[w]

    @staticmethod
    def r_lookup(i):
        return Vocab.mapping.inverse[i]

    @staticmethod
    def dump(filename):
        try:
            import pickle5 as pickle
        except Exception as e:
            import pickle
        with open(filename, mode="wb") as fh:
            pickle.dump(
                (
                    Vocab.mapping,
                    Vocab.reserved_count,
                    Vocab.non_reserved_count,
                    Vocab.non_reserved_offset,
                ),
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @staticmethod
    def size():
        return Vocab.non_reserved_offset + Vocab.non_reserved_count

    @staticmethod
    def load(filename):
        Vocab.mapping.clear()
        try:
            import pickle5 as pickle
        except Exception as e:
            import pickle
        with open(filename, mode="rb") as fh:
            try:
                loaded = pickle.load(fh)
                if isinstance(loaded, tuple):
                    (
                        Vocab.mapping,
                        Vocab.reserved_count,
                        Vocab.non_reserved_count,
                        Vocab.non_reserved_offset,
                    ) = loaded
                else:
                    Vocab.mapping = pickle.load(
                        fh,
                    )
                    Vocab.non_reserved_count = len(Vocab.mapping) - 1
                    Vocab.reserved_count = 1
                    Vocab.non_reserved_offset = 1
                    print(f"legacy vocab! {filename}")
            except Exception as e:
                raise e
        Vocab.freeze()

    @staticmethod
    def freeze():
        Vocab.freezed = True


class Utils:
    @staticmethod
    def load_mapping(fname):
        to_return = bidict()
        with open(fname, mode="r") as fh:
            for l in fh:
                p = l[:-1].split("\t")
                code = p[0]
                idx = int(p[1])
                if idx == 0:
                    to_return[idx] = 0
                else:
                    to_return[idx] = code
        return to_return

    @staticmethod
    def lookup_control(bos, eos, pad):
        bos = Vocab.lookup(bos)
        eos = Vocab.lookup(eos)
        pad = Vocab.lookup(pad)
        return bos, eos, pad

    @staticmethod
    def create_from_string(
        string,
        string_mapper: Optional = None,
        semiring_class: Optional = None,
        acceptor: bool = True,
    ):
        """
        Creates a FST which converts the empty string (epsilon) to string.
        String can be a normal python string or an iterable (list or tuple) of integers
        """

        if semiring_class is None:
            semiring_class = mfst.BooleanSemiringWeight
        ret = mfst.FST(
            acceptor=acceptor,
            semiring_class=semiring_class,
            string_mapper=string_mapper,
        )

        last = ret.add_state()
        ret.initial_state = last
        for (
            s
        ) in (
            string
        ):  # string can be any iterable object, eg (a normal string or a tuple of ints)
            state = ret.add_state()
            ret.add_arc(last, state, output_label=s if acceptor else 0, input_label=s)
            last = state
        ret.set_final_weight(last)
        return ret

    @staticmethod
    def tokenize_and_get_name(
        graphemes,
        phonemes,
        serialize_prefix,
        grapheme_tokenizer,
        phoneme_tokenizer,
        output_mapping,
        p2g: bool = False,
    ):
        if p2g:
            new_graphemes = "".join(phonemes.split(" "))
            new_phonemes = " ".join(graphemes)
        else:
            new_graphemes = graphemes
            new_phonemes = phonemes
        tokenized_graphemes = grapheme_tokenizer.tokenize(new_graphemes)
        tokenized_phonemes = phoneme_tokenizer.tokenize(new_phonemes)
        gs = [_.text for _ in tokenized_graphemes]
        ps = [Utils.word_to_code(output_mapping, _.text) for _ in tokenized_phonemes]
        g = "".join(gs)
        p = "-".join([str(_) for _ in ps])
        name = f"{serialize_prefix}-{g}-{p}"
        return gs, name, ps, g, p

    @staticmethod
    def mfst_to_pynini(inp, weight: Optional[float] = None) -> pynini.Fst:
        arcs_mapping = bidict()

        if weight is None:
            to_return = pynini.Fst()
        else:
            to_return = pynini.Fst(arc_type="log")

        if weight is None:
            semiring_weight = pynini.Weight.one(to_return.weight_type())
        else:
            semiring_weight = pynini.Weight(
                weight_type=to_return.weight_type(), weight=weight
            )
        for s in inp.states:
            arcs_mapping[s] = to_return.add_state()
        arcs_mapping[-1] = to_return.add_state()

        for s in inp.states:
            for a in inp.get_arcs(s):
                arc = pynini.Arc(
                    a.input_label,
                    a.output_label,
                    semiring_weight,
                    arcs_mapping[a.nextstate],
                )
                to_return.add_arc(arcs_mapping[s], arc)
        to_return.set_final(arcs_mapping[-1], semiring_weight)
        to_return.set_start(arcs_mapping[inp.initial_state])
        return to_return

    @staticmethod
    def length_prior_pynini(inp: pynini.Fst, weight: float) -> pynini.Fst:
        to_return = pynini.Fst(arc_type="log")
        semiring_weight = pynini.Weight(
            weight_type=to_return.weight_type(), weight=weight
        )
        to_return.add_states(inp.num_states())
        to_return.set_start(inp.start())
        for s in to_return.states():
            for arc in inp.arcs(s):
                new_arc = pynini.Arc(
                    arc.ilabel, arc.olabel, semiring_weight, arc.nextstate
                )
                to_return.add_arc(s, new_arc)
            if inp.final(s) == pynini.Weight.one(inp.weight_type()):
                to_return.set_final(s, pynini.Weight.one(to_return.weight_type()))
        return to_return

    @staticmethod
    def lift(fst, wfst_in_marks, wfst_out_marks, empty_mark: int):

        from pynini import Fst, Weight
        from tqdm import tqdm
        import numpy as np
        from collections import defaultdict

        fst: Fst
        zero = Weight.zero(fst.weight_type())

        ret = np.empty((fst.num_states() + 1, fst.num_states() + 1), dtype=np.long)
        ret.fill(empty_mark)
        # there should be only one start state per FST
        start_states = dict()
        # there can be multiple exit states per FST
        exit_states = defaultdict(set)

        for i in tqdm(range(fst.num_states()), disable=None):
            for arc in fst.arcs(i):
                exiting_arc = False
                nextstate = arc.nextstate
                ilabel = arc.ilabel
                if ilabel in wfst_in_marks:
                    # this is the start of a WFST segment.
                    # we clear all connections to this state to make sure we do not accidentally pathsum into other segments
                    assert ilabel not in start_states
                    start_states[ilabel] = arc.nextstate
                    ret[:, arc.nextstate] = empty_mark
                elif ilabel in wfst_out_marks:
                    # we are leaving a WFST segment.
                    # we should treat this as an exiting arc
                    exiting_arc = True
                    exit_states[ilabel].add(arc.nextstate)

                if fst.final(nextstate) != zero or exiting_arc:
                    # then this is a final state
                    assert isinstance(ret, np.ndarray)
                    ret[i, fst.num_states()] = ilabel
                else:
                    assert isinstance(ret, np.ndarray)
                    ret[i, arc.nextstate] = ilabel
        return ret, start_states, exit_states

    @staticmethod
    def load_fsa_from_npz(npz_fname, wfst_name, vocab_size, pad):
        assert exists(
            npz_fname
        ), f"{npz_fname} does not exist! Please run preprocess_npz.py first."
        l = load(npz_fname)
        to_return = (
            l["num_emission"],
            l["num_transition"],
            l["denom_emission"],
            l["denom_transition"],
            l["gs"],
            l["ps"],
        )
        try:
            # FIXME!
            pass
            # to_return.extend([l['num_wfst_marks'], l['denom_wfst_marks']])
        except Exception as e:
            pass

        if wfst_name is not None:
            from modules.scorers import FSAGRUScorer
            from pynini import Fst

            loaded_fst = Fst.read(wfst_name)
            matrices = FSAGRUScorer.get_state_mask_pynini(
                loaded_fst, vocab_size, pad, to_numpy=True, weighted=True
            )
            to_return = tuple(list(to_return) + list(matrices))
        return to_return

    @staticmethod
    def get_level_zero_mask(t: torch.Tensor, eos):
        threshold = Vocab.non_reserved_offset
        return torch.logical_or(t < threshold, t == eos).to(t.device)

    @staticmethod
    def padded_masked_select(t: torch.Tensor, mask: torch.Tensor, pad):
        """

        :param t:
        :param mask:
        :return:
        """
        assert len(mask.shape) == len(t.shape)
        assert len(mask.shape) == 2
        batch_size = mask.shape[0]
        seq_len = mask.shape[1]

        active_elements_per_row = mask.to(torch.long).sum(dim=1)
        num_padding_needed = (-active_elements_per_row + seq_len)[:, None].expand(
            -1, seq_len
        )
        arange_condition = torch.arange(seq_len, device=t.device)[None, :].expand(
            batch_size, -1
        )
        extra_paddings = torch.where(
            arange_condition < num_padding_needed,
            torch.ones_like(num_padding_needed),
            torch.zeros_like(num_padding_needed),
        )
        working_buffer = torch.empty_like(t)
        working_buffer.fill_(pad)
        working_buffer = torch.cat([t, working_buffer], dim=1)
        working_mask = torch.cat([mask, extra_paddings], dim=1).to(torch.bool)
        selected = torch.masked_select(working_buffer, working_mask).reshape(
            batch_size, seq_len
        )
        for i in range(seq_len):
            if torch.all(selected[:, i] == pad):
                break
        trimmed = selected[:, :i]
        return trimmed.to(t.device), working_buffer

    @staticmethod
    def pad_sequence(sequences, batch_first=False, padding_value=0.0):

        # assuming trailing dimensions and type of all the Tensors
        # in sequences are same and fetching those from sequences[0]
        max_size = sequences[0].shape
        trailing_dims = max_size[1:]
        max_len = max([s.shape[0] for s in sequences])
        if batch_first:
            out_dims = (len(sequences), max_len) + trailing_dims
        else:
            out_dims = (max_len, len(sequences)) + trailing_dims

        out_tensor = np.full(
            out_dims, fill_value=padding_value, dtype=sequences[0].dtype
        )
        for i, tensor in enumerate(sequences):
            length = tensor.shape[0]
            # use index notation to prevent duplicate references to the tensor
            if batch_first:
                out_tensor[i, :length, ...] = tensor
            else:
                out_tensor[:length, i, ...] = tensor

        return out_tensor

    @staticmethod
    def pad_to_longer(t1: torch.Tensor, t2: torch.Tensor, pad):
        """

        :param pad:
        :param t1:
        :param t2:
        :return:
        """
        assert len(t1.shape) == len(t2.shape) == 2
        if t1.shape[1] > t2.shape[1]:
            paddings = torch.empty(
                (t2.shape[0], t1.shape[1] - t2.shape[1]),
                device=t2.device,
                dtype=t2.dtype,
            )
            paddings.fill_(pad)
            t2 = torch.cat((t2, paddings), dim=1)
        elif t1.shape[1] < t2.shape[1]:
            paddings = torch.empty(
                (t1.shape[0], t2.shape[1] - t1.shape[1]),
                device=t1.device,
                dtype=t1.dtype,
            )
            paddings.fill_(pad)
            t1 = torch.cat((t1, paddings), dim=1)
        return t1, t2

    @staticmethod
    def get_hashed_name(gs, ps, name):
        hash_object = hashlib.sha256(repr(tuple(gs + ["<SEP>"] + ps)).encode("utf-8"))
        hex_dig = hash_object.hexdigest()
        return f"{name}/{hex_dig}"
