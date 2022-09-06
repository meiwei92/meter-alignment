from __future__ import annotations

import copy
from pympler import asizeof
from typing import List, OrderedDict as OrderedDictType

import metric


class MidiModelState:
    def get_score(self) -> float:
        raise NotImplementedError("Method get_score must be implemented!")

    def transition(self, notes: List[metric.MusicNote] = None) -> OrderedDictType[MidiModelState]:
        raise NotImplementedError("Method transition must be implemented!")

    def close(self) -> OrderedDictType[MidiModelState]:
        raise NotImplementedError("Method close must be implemented!")

    def deep_copy(self) -> MidiModelState:
        return copy.deepcopy(self)

    def is_duplicate_of(self, state: MidiModelState) -> bool:
        raise NotImplementedError("Method is_duplicate_of must be implemented!")


class MidiModel:
    def __init__(self):
        pass

    def transition(self, notes: List[metric.MusicNote] = None) -> OrderedDictType[MidiModelState]:
        raise NotImplementedError("Method transition must be implemented!")

    def close(self) -> OrderedDictType[MidiModelState]:
        raise NotImplementedError("Method close must be implemented!")

    def get_hypotheses(self) -> OrderedDictType[MidiModelState, None]:
        raise NotImplementedError("Method close must be implemented!")

