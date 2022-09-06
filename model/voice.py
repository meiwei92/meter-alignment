from __future__ import annotations

# import copy
from abc import ABC
from collections import OrderedDict
from typing import List, DefaultDict, OrderedDict as OrderedDictType

import metric
import model.base as model_base


class VoiceSplittingModelState(model_base.MidiModelState, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.voices: List[List[metric.Voice]] = []

    #@todo support multiple concurrent notes
    def get_current_voices(self) -> List[metric.Voice]:
        raise NotImplementedError("Method get_current_voices() not implemented!!!!")

    def remove_note(self, note: metric.MusicNote, min_note_length: int = 100000):
        if min_note_length < 0:
            return False

        for voice in self.get_current_voices():
            if voice.get_most_recent_note() == note:
                prev = voice.get_previous()
                if prev is None:
                    return False

                prev_note = prev.get_most_recent_note()
                if prev_note is None:
                    return False

                if note.get_start_tick() - prev_note.get_start_tick() < min_note_length:
                    return True

                return False

        return False

    def keep_note(self, note: metric.MusicNote, min_note_length: int = 100000):
        return not self.remove_note(note, min_note_length)

    def is_duplicate_of(self, state: VoiceSplittingModelState) -> bool:
        return isinstance(state, self.__class__)

    def get_voices(self) -> List[metric.Voice]:
        raise NotImplementedError("Method get_voices() not implemented!!!!!")


class VoiceSplittingGrammarModelState(VoiceSplittingModelState):
    def __init__(self, sequence: metric.TimePointSequence) -> None:
        super(VoiceSplittingGrammarModelState, self).__init__()
        self.most_recent_time = 0

        current_tp = sequence.first_point
        while current_tp is not None:
            notes_at_timepoint: DefaultDict[int, List[metric.MusicNote]] = sequence.get_notes_for_timepoint(current_tp)

            for idx, notes_of_voice in notes_at_timepoint.items():
                new_voices: List[metric.Voice] = []

                while len(self.voices) <= idx:
                    self.voices.append([])

                for n in notes_of_voice:
                    old_voices: List[metric.Voice] = self.voices[idx]
                    if len(old_voices) == 0:
                        new_voices.append(metric.Voice(current_note=n))
                    else:
                        for ov in old_voices:
                            new_voices.append(metric.Voice(current_note=n, prev_voice=ov))

                if len(new_voices) > 0:
                    self.voices[idx] = new_voices

            current_tp = current_tp.get_next()

    def get_current_voices(self) -> List[metric.Voice]:
        current_voices: List[metric.Voice] = []

        # @todo makemost CurrentTime/Tick a parameter
        for v_list in self.voices:
            if len(v_list) >= 2:
                raise NotImplementedError("Given List has multiple Voices. Not supported yet.")
            if len(v_list) == 0:
                continue

            v = v_list[0]
            while v is not None and v.get_most_recent_note().get_start_point().get_time_at_tick() > self.most_recent_time:
                v = v.get_previous()

            if v is not None and v not in current_voices:
                current_voices.append(v)

        return current_voices

    def get_score(self) -> float:
        return 0.0

    def transition(self, notes: List[metric.MusicNote] = None) -> OrderedDictType[model_base.MidiModelState]:
        new_state: OrderedDictType[model_base.MidiModelState] = OrderedDict()
        self.most_recent_time = notes[0].get_start_point().get_time_at_tick()
        new_state.update({self: None})
        return new_state

    def close(self) -> OrderedDictType[model_base.MidiModelState]:
        new_state: OrderedDictType[model_base.MidiModelState] = OrderedDict()
        self.most_recent_time += 1
        new_state.update({self: None})
        return new_state





