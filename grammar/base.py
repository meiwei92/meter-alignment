from __future__ import annotations

import metric
from model.beat import TatumTrackingModelState
from model.voice import VoiceSplittingModelState
from model.meter import MeterModel
from enum import Enum
from typing import *


class LpcfgLevel(Enum):
    MEASURE = 0
    BEAT = 1
    SUBBEAT = 2


class LpcfgWeight(Enum):
    EVEN = 0
    WEAK = -1
    STRONG = 1


class LpcfgQuantum(Enum):
    # A rest. No notes are played at this time.
    REST = 0
    # A note onset. A new note begins at this time.
    ONSET = 2
    # A tie. A note is being played at this time, but its onset was earlier.
    # This must be preceded by 0 or more other TIEs, preceded again by an ONSET.
    TIE = 1


class LpcfgVoice:
    '''
    Represents a voice in the context of the LPCF-Grammar
    '''

    def __init__(self, voice: metric.Voice, tatums: List[metric.Tatum], min_note_length: int = -1):
        self.beats_per_measure_list: List[int] = []
        self.subbeats_per_beat_list: List[int] = []
        self.tatums_per_measure_list: List[int] = []

        if tatums is None:
            raise ValueError("Given list of tatums must not be None")

        self.quantums = [LpcfgQuantum.REST for _ in range(len(tatums))]

        notes: Optional[List[metric.MusicNote]] = None
        if voice is not None:
            notes: List[metric.MusicNote] = voice.get_notes()

        if notes is None:
            # If no notes defined, either cause no voice given or if no notes exist in voice
            # We set the quantums of all tatums to REST.
            return

        if len(notes) > len(tatums):
            raise ValueError("Length of notes in voice must be smaller than list of tatums")

        note_active = False
        current_note_idx = 0
        current_tatum_idx = 0

        while current_tatum_idx < len(tatums):
            current_tatum_time = tatums[current_tatum_idx].time
            note: metric.MusicNote = notes[current_note_idx]

            if not note_active:
                loop = True
                while loop:
                    note: metric.MusicNote = notes[current_note_idx]
                    prev_note: metric.MusicNote = None if (current_note_idx == 0) else notes[current_note_idx - 1]

                    if (prev_note is not None) and (min_note_length >= 0):
                        delta_time = note.get_start_time() - prev_note.get_start_time()
                        if delta_time < min_note_length / 1000000:
                            # Nothing needs to be done for this note
                            if current_note_idx + 1 < len(notes):
                                current_note_idx += 1
                                continue
                    loop = False

                if note.get_start_time() < current_tatum_time:
                    prev_tatum_time = float('inf') if (current_tatum_idx == 0) else tatums[current_tatum_idx - 1].time
                    diff = abs(note.get_start_time() - current_tatum_time)
                    diff_prev = abs(note.get_start_time() - prev_tatum_time)

                    if diff < diff_prev:
                        self.quantums[current_tatum_idx] = LpcfgQuantum.ONSET
                    else:
                        self.quantums[current_tatum_idx - 1] = LpcfgQuantum.ONSET
                        self.quantums[current_tatum_idx] = LpcfgQuantum.TIE

                    note_active = True
            else:
                next_tatum_idx = current_tatum_idx + 1
                next_tatum_time = float('-inf') if (next_tatum_idx >= len(tatums)) else tatums[next_tatum_idx].time
                if note.get_end_time() >= next_tatum_time:
                    self.quantums[current_tatum_idx] = LpcfgQuantum.TIE
                else:
                    current_note_idx += 1
                    note_active = False

                    diff = abs(note.get_end_time() - current_tatum_time)
                    diff_next = abs(note.get_end_time() - next_tatum_time)

                    if diff > diff_next:
                        self.quantums[current_tatum_idx] = LpcfgQuantum.TIE
                    else:
                        continue

            current_tatum_idx += 1

    def get_beats_per_measure(self, measure_number: int):
        return self.beats_per_measure_list[measure_number]

    def get_subbeats_per_beat(self, measure_number: int):
        return self.subbeats_per_beat_list[measure_number]

    def get_tatums_per_measure(self, measure_number: int):
        return self.tatums_per_measure_list[measure_number]

    def set_metrics(self, beats_per_measure: List[int], subbeats_per_beat: List[int], tatums_per_measure: List[int]):
        self.beats_per_measure_list = beats_per_measure
        self.subbeats_per_beat_list = subbeats_per_beat
        self.tatums_per_measure_list = tatums_per_measure

    @staticmethod
    def generate_from(model: MeterModel, sequence: metric.TimePointSequence, min_note_length: int = -1,
                      tatum_denominator: int = 32) -> List[LpcfgVoice]:
        tatum_states: List[TatumTrackingModelState] = model.get_tatum_hypotheses()
        tatums: List[metric.Tatum] = tatum_states[0].tatums

        # measure_offset: int = tatums[0].measure
        #
        # beats_per_measure, subbeat_per_measure, tatums_per_measure = time_signature_values()
        beats_per_measure_list: List[int] = []
        subbeats_per_beat_list: List[int] = []
        tatums_per_measure_list: List[int] = []
        idx = 0

        while idx < len(tatums):
            t: metric.Tatum = tatums[idx]

            time_measure_data = sequence.time_measure_data_of_tick(t.get_tick())
            t_signature: metric.TimeSignature = time_measure_data[metric.TimePointSequence.TimeMeasureData.TIME_SIGNATURE]

            beats_per_measure_list.append(t_signature.get_beats_per_measure())
            subbeats_per_beat_list.append(t_signature.get_subbeats_per_beat())

            tatums_per_measure = time_measure_data[metric.TimePointSequence.TimeMeasureData.TATUMS_PER_MEASURE]
            tatums_per_measure_list.append(tatums_per_measure)

            idx += tatums_per_measure - tatums[idx].tatum

        voice_states: List[VoiceSplittingModelState] = model.get_voice_hypotheses()
        voices: List[metric.Voice] = voice_states[0].get_current_voices()

        lpcfg_voices = list()
        for cv in voices:
            lpcfg_voice = LpcfgVoice(cv, tatums, min_note_length)
            lpcfg_voice.set_metrics(beats_per_measure=beats_per_measure_list,
                                    subbeats_per_beat=subbeats_per_beat_list,
                                    tatums_per_measure=tatums_per_measure_list)
            lpcfg_voices.append(lpcfg_voice)

        return lpcfg_voices


class LpcfgTree:
    def __init__(self):
        pass

    @staticmethod
    def from_voice(voice: LpcfgVoice, tatums_per_measure: int = 32) -> List[LpcfgTree]:
        quantum_length = len(voice.quantums)
        measure_count = quantum_length / tatums_per_measure

        if int(measure_count) != measure_count:
            raise ValueError("Given number of tatums per measure can't produce a valid number of measures.")

        result_trees = list()
        for i in range(1, int(measure_count) + 1):
            start = (i - 1) * tatums_per_measure
            end = i * tatums_per_measure
            tree = LpcfgTree.from_measure(voice.quantums[start:end])
            result_trees.append(tree)

        return list()

    @staticmethod
    def from_measure(measure: List[LpcfgQuantum], extended: bool = False) -> LpcfgTree:

        return None


class LpcfgTreeNode:
    def __init__(self, level: LpcfgLevel):
        self.level: LpcfgLevel = level
        self.weight = 0
        self.children = list()
