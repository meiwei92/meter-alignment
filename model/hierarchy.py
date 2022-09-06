from __future__ import annotations

from abc import ABC
from collections import OrderedDict
from typing import List, OrderedDict as OrderedDictType

import metric
import model.base as base_model
import model.beat as beat_model
import model.voice as voice_model
from model.base import MidiModelState


class HierarchyModelState(base_model.MidiModelState, ABC):
    def __init__(self) -> None:
        super(HierarchyModelState).__init__()
        self.voice_splitting_state = None
        self.tatum_tracking_state = None

    def get_tatum_tracking_state(self) -> beat_model.TatumTrackingModelState:
        return self.tatum_tracking_state

    def set_tatum_tracking_state(self, state: beat_model.TatumTrackingModelState):
        self.tatum_tracking_state = state

    def get_voice_splitting_state(self) -> voice_model.VoiceSplittingModelState:
        return self.voice_splitting_state

    def set_voice_splitting_state(self, state: voice_model.VoiceSplittingModelState):
        self.voice_splitting_state = state

    def get_measure_number(self) -> int:
        raise NotImplementedError("get_bar_count() not implemented for this subtype of TatumTrackingModelState!!")

    def is_duplicate_of(self, state: HierarchyModelState) -> bool:
        return isinstance(state, self.__class__)


class HierarchyGrammarModelState(HierarchyModelState):
    def __init__(self, sequence: metric.TimePointSequence = None, measure=None, most_recent_time=None) -> None:
        super(HierarchyGrammarModelState).__init__()

        self.timepoint_sequence = sequence
        self.anacrusis_subbeat_length = sequence.get_anacrusis_subbeats()

        if measure is None:
            measure = metric.MetricalMeasure.from_timepoint(timepoint_sequence=sequence,
                                                     timepoint=sequence.first_point)

        self.measure = measure
        self.most_recent_time = 0 if most_recent_time is None else most_recent_time

    def get_score(self):
        return 1.0

    def transition(self, notes: List[metric.MusicNote] = None) -> OrderedDictType[base_model.MidiModelState]:
        new_state: OrderedDictType[HierarchyGrammarModelState, None] = OrderedDict()
        if notes is not None and len(notes) > 0:
            self.most_recent_time = notes[0].get_start_point().get_time_at_tick()

        new_state.update({self: None})
        return new_state

    def close(self) -> OrderedDictType[base_model.MidiModelState]:
        new_state: OrderedDictType[HierarchyGrammarModelState, None] = OrderedDict()
        new_state.update({self: None})
        return new_state

    def get_measure_number(self) -> int:
        return int('inf')

    def deep_copy(self) -> HierarchyGrammarModelState:
        return HierarchyGrammarModelState(sequence=self.timepoint_sequence,
                                          measure=self.measure,
                                          most_recent_time=self.most_recent_time)




