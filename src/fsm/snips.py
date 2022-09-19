import pickle

from mfst import FST
from src.modules.path_semiring import RefWeight, OwnAST
from src.util.preprocess_util import Vocab, Utils
from os.path import exists
from collections import defaultdict

from functools import lru_cache

from enum import Enum, auto


class InputMachineAlignment(Enum):
    ALIGNED = auto()
    NOT_ALIGNED = auto()
    SOMEWHAT_ALIGNED = auto()


INTENTS = [
    "PlayMusic",
    "AddToPlaylist",
    "RateBook",
    "SearchScreeningEvent",
    "BookRestaurant",
    "GetWeather",
    "SearchCreativeWork",
]

INTENTS_TYPE_MAP = {
    "PlayMusic": [
        "B-artist",
        "B-album",
        "B-track",
        "B-playlist",
        "B-music_item",
        "B-service",
        "B-sort",
        "B-year",
        "B-genre",
    ],
    "AddToPlaylist": [
        "B-artist",
        "B-playlist",
        "B-playlist_owner",
        "B-music_item",
        "B-entity_name",
    ],
    "RateBook": [
        "B-rating_unit",
        "B-object_type",
        "B-object_name",
        "B-rating_value",
        "B-best_rating",
        "B-object_select",
        "B-object_part_of_series_type",
    ],
    "BookRestaurant": [
        "B-sort",
        "B-party_size_number",
        "B-party_size_description",
        "B-spatial_relation",
        "B-city",
        "B-state",
        "B-poi",
        "B-country",
        "B-restaurant_type",
        "B-restaurant_name",
        "B-cuisine",
        "B-timeRange",
        "B-served_dish",
        "B-facility",
    ],
    "GetWeather": [
        "B-state",
        "B-city",
        "B-geographic_poi",
        "B-country",
        "B-timeRange",
        "B-spatial_relation",
        "B-condition_temperature",
        "B-current_location",
        "B-condition_description",
    ],
    "SearchCreativeWork": ["B-object_type", "B-object_name"],
    "SearchScreeningEvent": [
        "B-movie_type",
        "B-timeRange",
        "B-movie_name",
        "B-object_location_type",
        "B-location_name",
        "B-spatial_relation",
        "B-object_name",
        "B-object_type",
    ],
}
B_TYPES = [
    "B-artist",
    "B-album",
    "B-service",
    "B-entity_name",
    "B-playlist",
    "B-object_select",
    "B-object_type",
    "B-rating_value",
    "B-best_rating",
    "B-music_item",
    "B-track",
    "B-playlist_owner",
    "B-year",
    "B-sort",
    "B-movie_name",
    "B-party_size_number",
    "B-state",
    "B-city",
    "B-timeRange",
    "B-object_part_of_series_type",
    "B-movie_type",
    "B-spatial_relation",
    "B-geographic_poi",
    "B-restaurant_type",
    "B-party_size_description",
    "B-object_location_type",
    "B-object_name",
    "B-rating_unit",
    "B-location_name",
    "B-current_location",
    "B-served_dish",
    "B-country",
    "B-condition_temperature",
    "B-poi",
    "B-condition_description",
    "B-genre",
    "B-restaurant_name",
    "B-cuisine",
    "B-facility",
]

SPECIAL_B_TYPES = {
    "B-artist",
    "B-restaurant_type",
    "B-restaurant_name",
    "B-album",
    "B-track",
}


class Snips:
    def __init__(
        self,
        mapping_path,
        vocab_path,
        nro_path,
        control_symbols,
        bos,
        eos,
        pad,
        use_subtype: bool = False,
        alignment: InputMachineAlignment = InputMachineAlignment.ALIGNED,
        somewhat_aligned_degree: int = 2,
        use_old: bool = False,
    ):
        # used to construct vocab
        self.m = Utils.load_mapping(mapping_path)
        self.bio_marks = {"B": "B-tag", "I": "I-tag", "O": "O-tag"}
        self.tag_subtype = "SUBTYPE"
        self.use_subtype = use_subtype
        self.alignment = alignment
        self.somewhat_aligned_degree = somewhat_aligned_degree
        self.use_old = use_old
        print("use old machine: ", use_old)
        print("use subtype: ", self.use_subtype)
        print("Alignment type: ", alignment)
        self.control_symbols = control_symbols
        self.bos = bos
        self.eos = eos
        self.pad = pad
        self.vocab_path = vocab_path
        self.nro_path = nro_path
        self.nro_map = defaultdict(list)
        self.special_mark_map = defaultdict(tuple)
        self.nro_counter = defaultdict(int)
        self.make_vocab(vocab_path)
        self.agnostic_latent_after_fst = self.make_agnostic_latent_after_fst()
        self.agnostic_latent_nafter_fst = self.make_agnostic_latent_nafter_fst()
        self.agnostic_nlatent_after_fst = self.make_agnostic_nlatent_after_fst()
        self.agnostic_nlatent_nafter_fst = self.make_agnostic_nlatent_nafter_fst()
        self.nagnostic_latent_after_fst = self.make_nagnostic_latent_after_fst()
        self.nagnostic_latent_nafter_fst = self.make_nagnostic_latent_nafter_fst()
        self.nagnostic_nlatent_after_fst = self.make_nagnostic_nlatent_after_fst()
        self.nagnostic_nlatent_nafter_fst = self.make_nagnostic_nlatent_nafter_fst()

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
            for v in self.m.values():
                Vocab.add_word(v)
            for intent in INTENTS:
                Vocab.add_word(intent)
            for b_types in B_TYPES:
                Vocab.add_word(b_types)
            for v in self.bio_marks.values():
                Vocab.add_word(v)
            Vocab.add_word("closure-mark")  # hard code

            Vocab.add_word(self.tag_subtype)
            Vocab.add_word("B")
            Vocab.add_word("I")
            Vocab.add_word("O")
            Vocab.add_word(" ")
            Vocab.dump(vocab_path)
            Vocab.freeze()

    @lru_cache
    def _build_word_machine(
        self,
    ):
        if self.use_old:
            fst = FST(semiring_class=RefWeight)
            first_state = fst.add_state()
            second_state = fst.add_state()
            fst.set_initial_state(first_state)
            for char in self.m.values():
                for tag in ["B", "I", "O"]:
                    fst.add_arc(
                        first_state,
                        second_state,
                        input_label=Vocab.lookup(char),
                        output_label=Vocab.lookup(tag),
                        weight=RefWeight(
                            OwnAST(
                                "v",
                                (Vocab.lookup(char), Vocab.lookup(self.bio_marks[tag])),
                                None,
                            )
                        ),
                    )
                fst.add_arc(
                    second_state,
                    second_state,
                    input_label=Vocab.lookup(char),
                    output_label=0,
                    weight=RefWeight(OwnAST("v", (Vocab.lookup(char),), None)),
                )
            fst.set_final_weight(second_state)
            return fst

        fst = FST(semiring_class=RefWeight)
        first_state = fst.add_state()
        final_state = fst.add_state()

        fst.set_initial_state(first_state)
        fst.set_final_weight(final_state)

        self.add_tags_to_fst(first_state, final_state, fst)
        self.add_chars_to_fst(final_state, final_state, fst)
        return fst

    @lru_cache
    def _build_word_only(self):
        fst = FST(semiring_class=RefWeight)
        first_state = fst.add_state()
        final_state = first_state
        fst.set_initial_state(first_state)
        fst.set_final_weight(final_state)
        self.add_chars_to_fst(final_state, final_state, fst)
        return fst

    @lru_cache
    def _build_tag_only(self):
        fst = FST(semiring_class=RefWeight)
        first_state = fst.add_state()
        final_state = fst.add_state()

        fst.set_initial_state(first_state)
        fst.set_final_weight(final_state)
        self.add_tags_to_fst(first_state, final_state, fst)
        return fst

    def add_tags_to_fst(self, start_state, end_state, fst):
        for tag in self.bio_marks.keys():
            fst.add_arc(
                start_state,
                end_state,
                input_label=0,
                output_label=Vocab.lookup(tag),
                weight=RefWeight(
                    OwnAST("v", (Vocab.lookup(self.bio_marks[tag]),), None)
                ),
            )

            if self.use_subtype:
                fst.add_arc(
                    start_state,
                    end_state,
                    input_label=0,
                    output_label=Vocab.lookup(tag),
                    weight=RefWeight(
                        OwnAST(
                            "v",
                            (
                                Vocab.lookup(self.bio_marks[tag]),
                                Vocab.lookup(self.tag_subtype),
                            ),
                            None,
                        )
                    ),
                )

    def add_chars_to_fst(self, start_state, end_state, fst):
        for char in self.m.values():
            if char == " ":
                continue
            fst.add_arc(
                start_state,
                end_state,
                input_label=Vocab.lookup(char),
                output_label=0,
                weight=RefWeight(OwnAST("v", (Vocab.lookup(char),), None)),
            )

    @lru_cache
    def _input_word_machine_somewhat_aligned(self, somewhat: int = 2):
        append_space_machine = self._append_space()
        machine = FST(semiring_class=RefWeight)
        states = dict()
        for i in range(-somewhat, somewhat + 1):
            states[i] = machine.add_state()
            if i - 1 in states:
                word_entry, word_exit = OwnAST.add_component(
                    machine,
                    self._build_word_only(),
                )
                space_entry, space_exit = OwnAST.add_component(
                    machine,
                    self._space_machine(),
                )
                machine.add_arc(states[i - 1], word_entry)
                machine.add_arc(word_exit, space_entry)
                machine.add_arc(space_exit, states[i])

                tag_entry, tag_exit = OwnAST.add_component(
                    machine,
                    self._build_tag_only(),
                )
                machine.add_arc(states[i], tag_entry)
                machine.add_arc(tag_exit, states[i - 1])

        machine.set_initial_state(states[0])
        machine.set_final_weight(states[0])

        return append_space_machine.compose(machine)

    @lru_cache
    def _append_space(self):
        append_space = FST(semiring_class=RefWeight)
        append_space_init = append_space.add_state()
        append_space.set_initial_state(append_space_init)
        for char in self.m.values():
            append_space.add_arc(
                append_space_init,
                append_space_init,
                input_label=Vocab.lookup(char),
                output_label=Vocab.lookup(char),
                weight=RefWeight(OwnAST("v", (), None)),
            )
        append_space_final = append_space.add_state()
        append_space.add_arc(
            append_space_init,
            append_space_final,
            input_label=0,
            output_label=Vocab.lookup(" "),
            weight=RefWeight(OwnAST("v", (), None)),
        )
        append_space.set_final_weight(append_space_final)
        return append_space

    @lru_cache
    def _input_word_machine_not_aligned(self):
        word_machine: FST = self._build_word_only()
        tag_machine: FST = self._build_tag_only()
        space_machine: FST = self._space_machine()
        word_space_tag = word_machine.union(tag_machine).union(space_machine)
        return (word_space_tag).closure()

    @lru_cache
    def _input_word_machine(self):
        """(word_machine, <SP>)* word_machine"""
        word_machine = self._build_word_machine()
        space_machine = self._space_machine()
        word_space_machine = (word_machine.concat(space_machine).closure()).concat(
            word_machine
        )
        return word_space_machine

    @lru_cache
    def _space_machine(self):
        space_machine = FST(semiring_class=RefWeight)
        space_machine_init_state = space_machine.add_state()
        space_machine_second_state = space_machine.add_state()
        space_machine.set_initial_state(space_machine_init_state)
        space_machine.set_final_weight(space_machine_second_state)
        space_machine.add_arc(
            space_machine_init_state,
            space_machine_second_state,
            input_label=Vocab.lookup(" "),
            output_label=0,
            weight=RefWeight(OwnAST("v", (Vocab.lookup(" "),), None)),
        )
        space_machine.add_arc(
            space_machine_second_state,
            space_machine_second_state,
            input_label=Vocab.lookup(" "),
            output_label=0,
            weight=RefWeight(OwnAST("v", (Vocab.lookup(" "),), None)),
        )
        return space_machine

    def _build_non_agnostic_btype_machine(self, intent):
        fst = FST(semiring_class=RefWeight)
        init_state = fst.add_state()
        fst.set_initial_state(init_state)
        second_state = fst.add_state()
        for t in INTENTS_TYPE_MAP[intent]:
            fst.add_arc(
                init_state,
                second_state,
                input_label=Vocab.lookup("B"),
                output_label=Vocab.lookup(t),
                weight=RefWeight(OwnAST("v", (Vocab.lookup(t),), None)),
            )
        fst.add_arc(
            second_state,
            second_state,
            input_label=Vocab.lookup("I"),
            output_label=Vocab.lookup("I"),
            weight=RefWeight(OwnAST("v", (Vocab.lookup(self.bio_marks["I"]),), None)),
        )
        fst.add_arc(
            second_state,
            init_state,
            input_label=0,
            output_label=0,
            weight=RefWeight(OwnAST("v", (Vocab.lookup("closure-mark"),), None)),
        )
        fst.add_arc(
            init_state,
            init_state,
            input_label=Vocab.lookup("O"),
            output_label=Vocab.lookup("O"),
            weight=RefWeight(OwnAST("v", (Vocab.lookup(self.bio_marks["O"]),), None)),
        )

        fst.set_final_weight(init_state)
        return fst

    def _build_non_agnostic_intent_fst(self, intent, latent: bool = False):
        fst = FST(semiring_class=RefWeight)
        first_state = fst.add_state()
        last_state = fst.add_state()
        fst.set_initial_state(first_state)
        if latent:
            fst.add_arc(
                first_state,
                last_state,
                input_label=0,
                output_label=0,
                weight=RefWeight(OwnAST("v", (Vocab.lookup(intent),), None)),
            )
        else:
            fst.add_arc(
                first_state,
                last_state,
                input_label=0,
                output_label=Vocab.lookup(intent),
                weight=RefWeight(OwnAST("v", (Vocab.lookup(intent),), None)),
            )
        fst.set_final_weight(last_state)
        return fst

    def _build_general_btype_machine(self):
        fst = FST(semiring_class=RefWeight)
        init_state = fst.add_state()
        fst.set_initial_state(init_state)
        second_state = fst.add_state()
        fst.add_arc(
            init_state,
            second_state,
            input_label=Vocab.lookup("I"),
            output_label=Vocab.lookup("I"),
            weight=RefWeight(OwnAST("v", (Vocab.lookup(self.bio_marks["I"]),), None)),
        )
        fst.add_arc(
            second_state,
            second_state,
            input_label=Vocab.lookup("I"),
            output_label=Vocab.lookup("I"),
            weight=RefWeight(OwnAST("v", (Vocab.lookup(self.bio_marks["I"]),), None)),
        )
        fst.add_arc(
            second_state,
            init_state,
            input_label=0,
            output_label=0,
            weight=RefWeight(OwnAST("v", (Vocab.lookup("closure-mark"),), None)),
        )
        fst.add_arc(
            init_state,
            init_state,
            input_label=Vocab.lookup("O"),
            output_label=Vocab.lookup("O"),
            weight=RefWeight(OwnAST("v", (Vocab.lookup(self.bio_marks["O"]),), None)),
        )
        for t in B_TYPES:
            fst.add_arc(
                init_state,
                init_state,
                input_label=Vocab.lookup("B"),
                output_label=Vocab.lookup(t),
                weight=RefWeight(OwnAST("v", (Vocab.lookup(t),), None)),
            )
        fst.set_final_weight(init_state)
        return fst

    def _build_intent_fst(self, latent: bool = False):
        fst = FST(semiring_class=RefWeight)
        first_state = fst.add_state()
        last_state = fst.add_state()
        fst.set_initial_state(first_state)
        if latent:
            for intent in INTENTS:
                fst.add_arc(
                    first_state,
                    last_state,
                    input_label=0,
                    output_label=0,
                    weight=RefWeight(OwnAST("v", (Vocab.lookup(intent),), None)),
                )
        else:
            for intent in INTENTS:
                fst.add_arc(
                    first_state,
                    last_state,
                    input_label=0,
                    output_label=Vocab.lookup(intent),
                    weight=RefWeight(OwnAST("v", (Vocab.lookup(intent),), None)),
                )
        fst.set_final_weight(last_state)
        return fst

    def make_agnostic_latent_after_fst(self):
        intent_fst = self._build_intent_fst(latent=True)
        input_word_machine = self._input_word_machine()
        b_type_fst = self._build_general_btype_machine()
        fst = input_word_machine.compose(b_type_fst)
        fst = fst.concat(intent_fst)
        return fst

    def make_agnostic_latent_nafter_fst(self):
        intent_fst = self._build_intent_fst(latent=True)
        input_word_machine = self._input_word_machine()
        b_type_fst = self._build_general_btype_machine()
        fst = intent_fst.concat(input_word_machine.compose(b_type_fst))
        return fst

    def make_agnostic_nlatent_after_fst(self):
        intent_fst = self._build_intent_fst()
        input_word_machine = self._input_word_machine()
        b_type_fst = self._build_general_btype_machine()
        fst = input_word_machine.compose(b_type_fst)
        fst = fst.concat(intent_fst)
        return fst

    def make_agnostic_nlatent_nafter_fst(self):
        intent_fst = self._build_intent_fst()
        input_word_machine = self._input_word_machine()
        b_type_fst = self._build_general_btype_machine()
        fst = intent_fst.concat(input_word_machine.compose(b_type_fst))
        return fst

    def make_nagnostic_latent_after_fst(self):
        fst = FST(semiring_class=RefWeight)
        input_word_machine = self._input_word_machine()
        for intent in INTENTS:
            b_type_fst = self._build_non_agnostic_btype_machine(intent)
            intent_fst = self._build_non_agnostic_intent_fst(intent, latent=True)
            intent_fst = b_type_fst.concat(intent_fst)  # concat after
            fst = fst.union(intent_fst)
        return input_word_machine.compose(fst)

    def make_nagnostic_latent_nafter_fst(self):
        fst = FST(semiring_class=RefWeight)
        input_word_machine = self._input_word_machine()
        for intent in INTENTS:
            b_type_fst = self._build_non_agnostic_btype_machine(intent)
            intent_fst = self._build_non_agnostic_intent_fst(intent, latent=True)
            intent_fst = intent_fst.concat(b_type_fst)
            fst = fst.union(intent_fst)
        return input_word_machine.compose(fst)

    def make_nagnostic_nlatent_after_fst(self):
        fst = FST(semiring_class=RefWeight)
        input_word_machine = self._input_word_machine()
        for intent in INTENTS:
            b_type_fst = self._build_non_agnostic_btype_machine(intent)
            intent_fst = self._build_non_agnostic_intent_fst(intent)
            intent_fst = b_type_fst.concat(intent_fst)  # concat after
            fst = fst.union(intent_fst)
        return input_word_machine.compose(fst)

    def make_nagnostic_nlatent_nafter_fst(self):
        fst = FST(semiring_class=RefWeight)
        input_word_machine = self._input_word_machine()
        for intent in INTENTS:
            b_type_fst = self._build_non_agnostic_btype_machine(intent)
            intent_fst = self._build_non_agnostic_intent_fst(intent)
            intent_fst = intent_fst.concat(b_type_fst)
            fst = fst.union(intent_fst)
        return input_word_machine.compose(fst)

    def build_bracketed_fst(self, fst, btype):
        # FIXME bracket_left and bracket_right are not used
        # btype is the special type we are trying to build for
        bracket_left, bracket_right, start, end = self.get_special_mark(
            "scoredfst", btype
        )
        from modules.path_semiring import OwnAST

        init_fst = OwnAST.empty_fst(OwnAST("v", (start,), None))
        end_fst = OwnAST.empty_fst(OwnAST("v", (end,), None))
        bracketed_fst = init_fst.concat(fst).concat(end_fst)
        return bracketed_fst, bracket_left, bracket_right, start, end

    def get_special_mark(self, op, btype=None):
        # count = str(self.nro_counter[op])
        start = "{}-start".format(op)
        end = "{}-end".format(op)
        if op == "concat" or op == "closure":
            bracket = "{}-bracket".format(op)
            if not exists(self.vocab_path):
                Vocab.add_word(bracket)
                Vocab.add_word(start)
                Vocab.add_word(end)
            # self.nro_counter[op] += 1
            # self.nro_map[op].append((Vocab.lookup(bracket), Vocab.lookup(start), Vocab.lookup(end)))
            return Vocab.lookup(bracket), Vocab.lookup(start), Vocab.lookup(end)
        elif op == "scoredfst":
            # handle the special braket
            assert btype is not None
            if btype in self.special_mark_map:
                return self.special_mark_map[btype]
            else:
                start = "{}-start".format(btype)
                end = "{}-end".format(btype)
                # FIXME bracket left right is not used
                bracket_left = "{}-bracket-left".format(btype)
                bracket_right = "{}-bracket-right".format(btype)
                if not exists(self.vocab_path):
                    Vocab.add_word(bracket_left)
                    Vocab.add_word(bracket_right)
                    Vocab.add_word(start)
                    Vocab.add_word(end)
                self.special_mark_map[btype] = (
                    Vocab.lookup(bracket_left),
                    Vocab.lookup(bracket_right),
                    Vocab.lookup(start),
                    Vocab.lookup(end),
                )
                return (
                    Vocab.lookup(bracket_left),
                    Vocab.lookup(bracket_right),
                    Vocab.lookup(start),
                    Vocab.lookup(end),
                )
        else:  # compose or union
            bracket_left = "{}-bracket-left".format(op)
            bracket_right = "{}-bracket-right".format(op)
            if not exists(self.vocab_path):
                Vocab.add_word(bracket_left)
                Vocab.add_word(bracket_right)
                Vocab.add_word(start)
                Vocab.add_word(end)
            # self.nro_counter[op] += 1
            # self.nro_map[op].append((Vocab.lookup(bracket_left), Vocab.lookup(bracket_right), Vocab.lookup(start),
            # Vocab.lookup(end)))
            return (
                Vocab.lookup(bracket_left),
                Vocab.lookup(bracket_right),
                Vocab.lookup(start),
                Vocab.lookup(end),
            )

    def nro(self):
        # add special mark for non-agnostic, non-latent, non-after machine
        fst = FST(semiring_class=RefWeight)
        # input_word_machine = self._input_word_machine()
        input_word_machine = self.get_input_word_machine()

        for intent in INTENTS:
            b_type_fst = self._build_non_agnostic_btype_machine_nro(intent)
            intent_fst = self._build_non_agnostic_intent_fst(intent)
            intent_fst = intent_fst.concat(b_type_fst)
            fst = fst.union(intent_fst)
        fst = input_word_machine.compose(fst)
        if not exists(self.vocab_path):
            Vocab.dump(
                self.vocab_path
            )  # vocab is updated by additional special marks now
        # dumpy special marks as well
        if not exists(self.nro_path):
            with open(self.nro_path, "wb") as handle:
                pickle.dump(
                    self.special_mark_map, handle, protocol=pickle.HIGHEST_PROTOCOL
                )
        return fst

    @lru_cache
    def get_input_word_machine(self):
        if self.alignment == InputMachineAlignment.ALIGNED:
            input_word_machine = self._input_word_machine()
        elif self.alignment == InputMachineAlignment.NOT_ALIGNED:
            input_word_machine = self._input_word_machine_not_aligned()
        elif self.alignment == InputMachineAlignment.SOMEWHAT_ALIGNED:
            input_word_machine = self._input_word_machine_somewhat_aligned(
                self.somewhat_aligned_degree
            )
        else:
            raise NotImplementedError(
                f"Input alignment not implemented! {self.alignment}"
            )
        return input_word_machine

    def _build_non_agnostic_btype_machine_nro(self, intent):
        to_return = FST(semiring_class=RefWeight)
        for t in INTENTS_TYPE_MAP[intent]:
            fst = self.specific_btype(t)
            # union with nro
            # if t in SPECIAL_B_TYPES: treat all as special types
            fst, _, _, _, _ = self.build_bracketed_fst(fst, t)
            to_return = to_return.union(fst)

        # build a separete O machine
        o_machine = FST(semiring_class=RefWeight)
        o_machine_state = o_machine.add_state()
        o_machine_second_state = o_machine.add_state()
        o_machine.set_initial_state(o_machine_state)
        o_machine.set_final_weight(o_machine_second_state)
        o_machine.add_arc(
            o_machine_state,
            o_machine_second_state,
            input_label=Vocab.lookup("O"),
            output_label=Vocab.lookup("O"),
            weight=RefWeight(OwnAST("v", (Vocab.lookup(self.bio_marks["O"]),), None)),
        )
        to_return = to_return.union(o_machine)
        # clusure on unioned BIO machine
        to_return = to_return.closure()
        return to_return

    @lru_cache
    def specific_btype(self, t):
        fst = FST(semiring_class=RefWeight)
        init_state = fst.add_state()
        fst.set_initial_state(init_state)
        second_state = fst.add_state()
        fst.add_arc(
            init_state,
            second_state,
            input_label=Vocab.lookup("B"),
            output_label=Vocab.lookup(t),
            weight=RefWeight(OwnAST("v", (Vocab.lookup(t),), None)),
        )
        fst.add_arc(
            second_state,
            second_state,
            input_label=Vocab.lookup("I"),
            output_label=Vocab.lookup("I"),
            weight=RefWeight(OwnAST("v", (Vocab.lookup(self.bio_marks["I"]),), None)),
        )
        fst.set_final_weight(second_state)
        return fst


if __name__ == "__main__":
    snips_fst = Snips(
        "/home/steven/Code/GITHUB/seq-samplers/snips/nfst/mapping.snips",
        "/home/steven/Code/GITHUB/seq-samplers/snips/nfst/snips-nro.vocab",
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
    )
    en = [
        63,
        59,
        52,
        48,
        66,
        52,
        120,
        55,
        52,
        59,
        63,
        120,
        60,
        52,
        120,
        66,
        52,
        48,
        65,
        50,
        55,
        120,
        67,
        55,
        52,
        120,
        55,
        52,
        59,
        59,
        120,
        60,
        62,
        61,
        52,
        72,
        120,
        66,
        48,
        54,
        48,
    ]
    ci = [169, 169, 169, 169, 169, 154, 168, 134]
