import torch
from typing import Tuple, Container, List
from easy_latent_seq.util.preprocess_util import Utils, Vocab
from easy_latent_seq.modules.path_semiring import RefWeight, OwnAST


class ZTable:
    mapping = dict()
    exit_states = set()


class NROWFSTFunc:
    def __init__(self, wfst_start, wfst_end, g, pad, substract_z):
        self.wfst_start = wfst_start
        self.wfst_end = wfst_end
        self.g = g
        self.pad = pad
        self.substract_z = substract_z

    def apply(self, t: torch.Tensor):
        with torch.no_grad():
            assert torch.all(t[:, 0] == self.wfst_start)
            wfst_mask: torch.Tensor = t == self.wfst_end
            wfst_end_indices = torch.argmax(
                wfst_mask.to(torch.get_default_dtype()), dim=1
            )
            assert not torch.any(wfst_end_indices == 0)
            t_suffix = t[:, 1:]
            t_masked = torch.masked_fill(t_suffix, wfst_mask[:, 1:], self.pad)
        if not self.substract_z:
            return self.g(t_masked)
        return self.g(t_masked) - ZTable.mapping[self.wfst_start]


class NRO:
    @staticmethod
    def nro_concatenate_m(fsts: List, bracket_left, concatenate_start, concatenate_end):
        init_fst = OwnAST.empty_fst(OwnAST("v", (concatenate_start,), None))
        end_fst = OwnAST.empty_fst(OwnAST("v", (concatenate_end,), None))
        bracket_fst = OwnAST.empty_fst(OwnAST("v", (bracket_left,), None))
        to_return = init_fst
        for fst in fsts:
            bracketed = bracket_fst.concat(fst)
            to_return = to_return.concat(bracketed)
        return to_return.concat(end_fst)

    @staticmethod
    def nro_concatenate_g(
        bracket_left, concatenate_start, concatenate_end, pad, eos, gs: List
    ):
        def to_return(t: torch.Tensor, indices: torch.Tensor):
            with torch.no_grad():
                aligned_tensors = []
                assert torch.all(t[:, 0] == concatenate_start)
                cat_mask: torch.Tensor = t == concatenate_end
                cat_end_indices = torch.argmax(
                    cat_mask.to(torch.get_default_dtype()), dim=1
                )
                assert not torch.any(cat_end_indices == 0)
                t_suffix = t[:, 1:]
                t_masked = torch.masked_fill(t_suffix, cat_mask[:, 1:], pad)
                indices, count = Utils.idx_masked_tensor(t_masked, bracket_left)
                assert torch.all(count == len(gs))
                for i in range(len(gs)):
                    m, first = Utils.mask_out_and_first(t_masked, indices, i, pad)
                    aligned = Utils.left_align(m, first, pad, eos)
                    aligned_tensors.append(aligned)
            assert len(aligned_tensors) == len(
                gs
            ), f"# of extracted tensors: {len(aligned_tensors)}\t# of functions: {len(gs)}"
            scored = gs[0](aligned_tensors[0])
            for g, tensor in zip(gs[1:], aligned_tensors[1:]):
                scored += g(tensor)
            return scored

        return to_return

    @staticmethod
    def nro_union_m(fst1, fst2, bracket_1_left, bracket_2_left, union_start, union_end):
        init_fst1 = OwnAST.empty_fst(OwnAST("v", (union_start, bracket_1_left), None))
        init_fst2 = OwnAST.empty_fst(OwnAST("v", (union_start, bracket_2_left), None))

        end_fst = OwnAST.empty_fst(OwnAST("v", (union_end,), None))
        return ((init_fst1.concat(fst1)).union(init_fst2.concat(fst2))).concat(end_fst)

    @staticmethod
    def nro_union_g(
        bracket_1_left, bracket_2_left, union_start, union_end, pad, eos, g1, g2
    ):
        def to_return(t: torch.Tensor):
            with torch.no_grad():
                assert torch.all(t[:, 0] == union_start), (
                    union_start,
                    (t[:, 0] != union_start).nonzero(),
                )
                assert torch.all(
                    (t[:, 1] == bracket_1_left) | (t[:, 1] == bracket_2_left)
                )
                masked: torch.Tensor = t[:, 2:]
                masked.masked_fill_(masked == eos, pad)
                masked.masked_fill_(masked == union_end, eos)
            first_score = g1(masked)
            second_score = g2(masked)
            assert first_score.shape == second_score.shape
            assert len(first_score.shape) == 1
            assert first_score.shape[0] == t.shape[0]
            return torch.where(t[:, 1] == bracket_1_left, first_score, second_score)

        return to_return

    @staticmethod
    def nro_closure_m(fst1, bracket_left, closure_start, closure_end):
        init_fst = OwnAST.empty_fst(OwnAST("v", (closure_start,), None))
        end_fst = OwnAST.empty_fst(OwnAST("v", (closure_end,), None))
        bracket_fst = OwnAST.empty_fst(OwnAST("v", (bracket_left,), None))
        concatenated = bracket_fst.concat(fst1)
        closure = concatenated.closure()
        return (init_fst.concat(closure)).concat(end_fst)

    @staticmethod
    def nro_closure_g(bracket_left, closure_start, closure_end, pad, eos, g1):
        def to_return(t: torch.Tensor):
            with torch.no_grad():
                aligned_tensors = []
                assert torch.all(t[:, 0] == closure_start)
                closure_mask: torch.Tensor = t == closure_end
                closure_end_indices = torch.argmax(
                    closure_mask.to(torch.get_default_dtype()), dim=1
                )
                assert not torch.any(closure_end_indices == 0)
                t_suffix = t[:, 1:]
                t_masked = torch.masked_fill(t_suffix, closure_mask[:, 1:], pad)
                indices, count = Utils.idx_masked_tensor(t_masked, bracket_left)
                max_count = torch.max(count).item()
                for i in range(max_count):
                    m, first = Utils.mask_out_and_first(t_masked, indices, i, pad)
                    aligned = Utils.left_align(m, first, pad, eos)
                    aligned_tensors.append(aligned)
            if len(aligned_tensors) > 0:
                stacked = torch.stack(aligned_tensors, dim=1)
                flattened = stacked.reshape(t.shape[0] * max_count, -1)
                flattened_zeros = torch.zeros(
                    (t.shape[0] * max_count,), device=flattened.device
                )

                r = torch.arange(
                    flattened.shape[0], dtype=torch.long, device=flattened.device
                )
                non_zeros: torch.Tensor = flattened[:, 0] != eos
                non_zero_indices = non_zeros.nonzero(as_tuple=False).flatten()
                flattened_nonzeros = flattened[non_zero_indices]
                nonzero_scored = g1(flattened_nonzeros)
                flattened_zeros[non_zero_indices] = nonzero_scored
                scored = flattened_zeros.reshape(t.shape[0], max_count).sum(dim=1)
            else:
                # if we don't have a single aligned tensor, it means that none of the paths actually went through the closure even once.
                # therefore we should just return a zero vector
                # (we could have fed g a zero tensor too, which should yield the same answer)
                scored = torch.zeros((t.shape[0],), device=t.device)
            return scored

        return to_return

    @staticmethod
    def nro_wfst_m(fst1, wfst_start, wfst_end):
        init_fst = OwnAST.empty_fst(OwnAST("v", (wfst_start,), None))
        end_fst = OwnAST.empty_fst(OwnAST("v", (wfst_end,), None))
        return (init_fst.concat(fst1)).concat(end_fst)

    @staticmethod
    def nro_wfst_g(wfst_start, wfst_end, g, pad, substract_z: bool = True):
        func_object = NROWFSTFunc(wfst_start, wfst_end, g, pad, substract_z)
        return func_object.apply

    @staticmethod
    def nro_compose_m(
        fst1, fst2, bracket_1_left, bracket_2_left, compose_start, compose_end
    ):
        def weight_fn(old_ast: OwnAST, bracket_left) -> OwnAST:
            left_bracket_ast = OwnAST("v", (bracket_left,), None)
            new_ast = OwnAST("*", left_bracket_ast, old_ast)
            return new_ast

        fst1_bracket = lambda ast: weight_fn(ast, bracket_1_left)
        fst2_bracket = lambda ast: weight_fn(ast, bracket_2_left)
        fst1_bracketed = OwnAST.fst_weight_op(fst1, fst1_bracket)
        fst2_bracketed = OwnAST.fst_weight_op(fst2, fst2_bracket)
        init_fst = OwnAST.empty_fst(OwnAST("v", (compose_start,), None))
        end_fst = OwnAST.empty_fst(OwnAST("v", (compose_end,), None))

        return init_fst.concat(fst1_bracketed.compose(fst2_bracketed)).concat(end_fst)

    @staticmethod
    def nro_compose_g(
        bracket_1_left, bracket_2_left, compose_start, compose_end, pad, eos, g1, g2
    ):
        def to_return(t: torch.Tensor):
            with torch.no_grad():
                first: torch.Tensor = torch.zeros_like(t)
                second: torch.Tensor = torch.zeros_like(t)
                # pad_tensor: torch.Tensor = torch.empty_like(t)
                first_indices = torch.zeros(
                    (t.shape[0],), dtype=t.dtype, device=t.device
                )
                second_indices = torch.zeros(
                    (t.shape[0],), dtype=t.dtype, device=t.device
                )
                first_run = torch.zeros(
                    (t.shape[0],), dtype=torch.bool, device=t.device
                )
                hit_end = torch.zeros((t.shape[0],), dtype=torch.bool, device=t.device)

                batch_size = t.shape[0]
                seq_len = t.shape[1]
                # pad_tensor.fill_(pad)

                first.fill_(pad)
                second.fill_(pad)

                # the first symbol must be compose_start
                assert torch.all(t[:, 0] == compose_start)

                for i in range(1, seq_len):
                    # update first indices: if first_run, first_indices increments by 1.
                    # otherwise, second_indices increments by 1.
                    if torch.all(hit_end):
                        break
                    hit_end = hit_end | (t[:, i] == compose_end)
                    first[torch.arange(batch_size), first_indices] = t[:, i]
                    second[torch.arange(batch_size), second_indices] = t[:, i]

                    current_is_first_start: torch.Tensor = t[:, i] == bracket_1_left
                    current_is_second_start: torch.Tensor = t[:, i] == bracket_2_left
                    first_run = current_is_first_start | first_run
                    first_run = (~current_is_second_start) & first_run
                    first_indices = torch.where(
                        first_run & (~current_is_first_start),
                        first_indices + 1,
                        first_indices,
                    )
                    second_indices = torch.where(
                        first_run | current_is_second_start,
                        second_indices,
                        second_indices + 1,
                    )
                assert torch.all(hit_end)

                first.masked_fill_(first == eos, pad)
                second.masked_fill_(second == eos, pad)
                first.masked_fill_(first == compose_end, eos)
                second.masked_fill_(second == compose_end, eos)
            first_score = g1(first)
            second_score = g2(second)
            return first_score + second_score

        return to_return

    @staticmethod
    def extract_first_second(
        t: torch.Tensor,
        compose_start,
        compose_end,
        bracket_1_left,
        bracket_2_left,
        eos,
        pad,
    ):
        with torch.no_grad():
            first_mask = torch.zeros_like(t, dtype=torch.bool)
            second_mask = torch.zeros_like(t, dtype=torch.bool)
            first: torch.Tensor = torch.zeros_like(t)
            second: torch.Tensor = torch.zeros_like(t)
            # pad_tensor: torch.Tensor = torch.empty_like(t)
            first_indices = torch.zeros(
                (t.shape[0],), dtype=torch.long, device=t.device
            )
            second_indices = torch.zeros(
                (t.shape[0],), dtype=torch.long, device=t.device
            )
            first_run = torch.zeros((t.shape[0],), dtype=torch.bool, device=t.device)
            hit_end = torch.zeros((t.shape[0],), dtype=torch.bool, device=t.device)

            batch_size = t.shape[0]
            seq_len = t.shape[1]
            # pad_tensor.fill_(pad)

            first.fill_(pad)
            second.fill_(pad)

            cpend_filler = torch.zeros((t.shape[0],), dtype=t.dtype, device=t.device)
            cpend_filler.fill_(compose_end)
            # the first symbol must be compose_start
            assert torch.all(t[:, 0] == compose_start)

            for i in range(1, seq_len):
                # update first indices: if first_run, first_indices increments by 1.
                # otherwise, second_indices increments by 1.
                t_at_i = t[:, i]
                t_at_i_masked = torch.where(t_at_i == eos, cpend_filler, t_at_i)
                first[
                    torch.arange(batch_size, dtype=torch.long), first_indices
                ] = torch.where(
                    first[torch.arange(batch_size, dtype=torch.long), first_indices]
                    == compose_end,
                    cpend_filler,
                    t_at_i,
                )
                second[
                    torch.arange(batch_size, dtype=torch.long), second_indices
                ] = torch.where(
                    second[torch.arange(batch_size, dtype=torch.long), second_indices]
                    == compose_end,
                    cpend_filler,
                    t_at_i,
                )

                if torch.all(hit_end):
                    break
                hit_end = hit_end | (t_at_i == compose_end)

                current_is_first_start: torch.Tensor = t_at_i == bracket_1_left
                current_is_second_start: torch.Tensor = t_at_i == bracket_2_left
                first_run = current_is_first_start | first_run
                first_run = (~current_is_second_start) & first_run
                first_flag = first_run & (~current_is_first_start)
                first_indices = torch.where(
                    first_flag, first_indices + 1, first_indices
                )
                second_flag = ~(first_run | current_is_second_start)
                second_indices = torch.where(
                    second_flag, second_indices + 1, second_indices
                )
                t_at_i_is_not_eos = t_at_i != eos
                t_at_i_is_not_cpend = t_at_i != compose_end
                t_at_i_is_not_pad = t_at_i != pad
                not_ending_indicator = (
                    t_at_i_is_not_cpend & t_at_i_is_not_eos & t_at_i_is_not_pad
                )
                first_mask[:, i] = first_flag & not_ending_indicator
                second_mask[:, i] = second_flag & not_ending_indicator

            assert torch.all(hit_end)
            first[torch.arange(batch_size, dtype=torch.long), first_indices].fill_(
                compose_end
            )
            second[torch.arange(batch_size, dtype=torch.long), second_indices].fill_(
                compose_end
            )

            first.masked_fill_(first == eos, pad)
            second.masked_fill_(second == eos, pad)
            first.masked_fill_(first == compose_end, eos)
            second.masked_fill_(second == compose_end, eos)
        return first, second, first_mask, second_mask
