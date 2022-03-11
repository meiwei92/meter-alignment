from __future__ import annotations

from abc import ABC
from collections import OrderedDict
from typing import List, OrderedDict as OrderedDictType

from metric import MusicNote, TimePointSequence, Tatum

import model.base as base_model
import model.hierarchy as hierarchy_model


class TatumTrackingModelState(base_model.MidiModelState, ABC):
    """

    """
    def __init__(self) -> None:
        super(TatumTrackingModelState).__init__()
        self.tatums: List[Tatum] = []
        self.tatum_times: List[float] = []
        self.hierarchy_state = None

    def get_hierarchy_state(self):
        return self.hierarchy_state

    def set_hierarchy_state(self, state: hierarchy_model.HierarchyModelState):
        self.hierarchy_state = state

    def is_started(self):
        current_tatum_times = self.get_current_tatum_times()
        return current_tatum_times is not None and len(current_tatum_times) > 0

    def get_current_tatum_times(self) -> List[int]:
        raise NotImplementedError("get_current_tatum_times() not implemented for this subtype of TatumTrackingModelState!!")

    def get_measure_count(self) -> int:
        """
        Return the number of bars the model state has passed through so far
        :return:
        """
        raise NotImplementedError("get_measure_count() not implemented for this subtype of TatumTrackingModelState!!")

    def is_duplicate_of(self, state: TatumTrackingModelState) -> bool:
        return isinstance(state, self.__class__)


class TatumTrackingGrammarModelState(TatumTrackingModelState):
    def __init__(self, sequence: TimePointSequence = None, state: TatumTrackingGrammarModelState = None) -> None:
        super(TatumTrackingGrammarModelState).__init__()

        if sequence is not None:
            self.timepoint_sequence = sequence
            self.tatums: List[Tatum] = sequence.get_tatums()
            self.tatum_times: List[float] = []

            for t in self.tatums:
                self.tatum_times.append(t.get_time())

            self.most_recent_time = 0
            self.most_recent_idx = 0

            self.beats = sequence.anacrusis_length
        elif state is not None:
            self.timepoint_sequence = state.timepoint_sequence
            self.tatums = state.tatums
            self.tatum_times = state.tatum_times
            self.most_recent_time = state.most_recent_time
            self.most_recent_idx = state.most_recent_idx
        else:
            raise ValueError("Either timepoint sequence or state must be given!")

    def get_score(self) -> float:
        return 0.0

    def transition(self, notes: List[MusicNote] = None) -> OrderedDictType[base_model.MidiModelState]:
        new_state: OrderedDictType[TatumTrackingGrammarModelState, None] = OrderedDict()
        if notes is not None and len(notes) > 0:
            self.most_recent_time = notes[0].get_start_point().get_time_at_tick()
            idx = self.most_recent_idx
            while idx < len(self.tatums):
                if self.tatums[idx].get_time() > self.most_recent_time:
                    self.most_recent_idx = idx
                    break
                idx += 1

        new_state.update({self: None})
        return new_state

    def close(self) -> OrderedDictType[base_model.MidiModelState]:
        new_state: OrderedDictType[TatumTrackingGrammarModelState, None] = OrderedDict()

        self.most_recent_idx = len(self.tatums)
        self.most_recent_time = self.tatums[-1].get_time()

        new_state.update({self: None})
        return new_state

    def get_measure_count(self) -> int:
        recent_tatum_idx = min(self.most_recent_idx, len(self.tatums) - 1)
        recent_tatum: Tatum = self.tatums[recent_tatum_idx]
        return recent_tatum.measure

    def get_current_tatum_times(self) -> List[float]:
        return self.tatum_times[:self.most_recent_idx]

    def deep_copy(self) -> TatumTrackingModelState:
        return TatumTrackingGrammarModelState(state=self)
