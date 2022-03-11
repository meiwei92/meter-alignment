import partitura as ptt
from partitura.score import (
    TimePoint as PTTTimePoint,
    Tempo as PTTTempo,
    TimeSignature as PTTTimeSignature,
    Note as PTTNote
)
from pathlib import Path
from parsing import FileParser
from base import *
from metric import *
from typing import *
import numpy as np


# PTTT = TypeVar('PTTT', PTTTempo, PTTTimeSignature, PTTNote)


class MidiFileParser(FileParser):
    def __init__(self, file: Path, anacrusis_length: int,
                 use_channel: bool = True, verbose: bool = False,
                 subbeat_length: int = 4, min_note_length: int = 100000) -> None:
        super().__init__(file, anacrusis_length, use_channel, verbose, subbeat_length, min_note_length)

        self.sequence: TimePointSequence = TimePointSequence(subbeat_length)
        self.notes = list()

    def parse(self) -> TimePointSequence:
        file_path_string = self.file.resolve().as_posix()
        score_data = ptt.load_score_midi(file_path_string)
        performance_data = ptt.load_performance_midi(file_path_string)

        self.__map_partitura_data(score_data, performance_data)
        return self.sequence

    def __map_partitura_data(self, score_data, performance_data) -> None:
        for part in score_data:
            current_timepoint: PTTTimePoint = part.first_point
            while current_timepoint is not None:
                beat = self.__get_singleton(ptt_timepoint=current_timepoint, fetch_type=PTTTempo,
                                            mapper=lambda x: x.bpm if x is not None else None)

                time_signature = self.__get_singleton(ptt_timepoint=current_timepoint, fetch_type=PTTTimeSignature,
                                                      mapper=lambda x: TimeSignature(beats_per_measure=x.beats,
                                                                                     beat_length_denominator=x.beat_type))

                notes = self.__get_list(ptt_timepoint=current_timepoint, fetch_type=PTTNote,
                                        mapper=lambda x: self.__map_to_musicnote(x, performance_data=performance_data))

                if not self.sequence.has_timepoint_at_tick(current_timepoint.t):
                    self.sequence.add_timepoint(tick=current_timepoint.t,
                                                bpm=beat, ppb=current_timepoint.quarter,
                                                time_signature=time_signature,
                                                notes=notes)
                else:
                    compatible = self.sequence.check_compatibility_of_metadata(tick=current_timepoint.t,
                                                                               bpm=beat,
                                                                               ppb=current_timepoint.quarter,
                                                                               time_signature=time_signature)
                    # print(f"Comptibility for tick {current_timepoint.t} "
                    #       f"[{beat}, {current_timepoint.quarter}, {time_signature}] -> '{compatible}'")

                    if not compatible:
                        raise InvalidFormatError("Given Metadata of File conflicts!!!")

                    self.sequence.add_notes(tick=current_timepoint.t, notes=notes)

                current_timepoint = current_timepoint.next

    @staticmethod
    def __map_to_musicnote(ptt_note: PTTNote, performance_data) -> Optional[MusicNote]:
        pd = performance_data.note_array[performance_data.note_array['id'] == ptt_note.id]

        if len(pd) == 0:
            # No Data, means no Note.
            return None

        if len(pd) > 1:
            raise ConflictingValuesError(f"performance_data has '{len(pd)}' entries for the given note, "
                                         f"required would be '1'.")

        pd = pd[0]

        velocity = pd['velocity']
        pitch = pd['pitch']
        track = pd['track']
        channel = pd['channel']
        start_tick = ptt_note.start.t
        end_tick = ptt_note.end.t

        result_note = MusicNote(start_tick=start_tick, end_tick=end_tick,
                                velocity=velocity, pitch=pitch,
                                track=track, channel=channel)
        return result_note

    @staticmethod
    def __get_singleton(ptt_timepoint: PTTTimePoint, fetch_type: Type[Union[PTTTempo, PTTTimeSignature, PTTNote]],
                        mapper: Callable = lambda x: x) -> Any:
        values = ptt_timepoint.starting_objects[fetch_type]

        if len(values) > 1:
            raise ConflictingValuesError(
                f"Given Timepoint {ptt_timepoint.t} has multiple definitions of Type '{fetch_type.__name__}'")

        result: Optional[fetch_type] = None
        if len(values) == 1:
            tmp_obj = next(iter(values))
            result = mapper(tmp_obj) if mapper is not None else tmp_obj

        return result

    @staticmethod
    def __get_list(ptt_timepoint: PTTTimePoint, fetch_type: Type[Union[PTTTempo, PTTTimeSignature, PTTNote]],
                   mapper: Callable = lambda x: x) -> List[Any]:
        values = ptt_timepoint.starting_objects[fetch_type]
        result_list = []

        for item in iter(values):
            result_list.append(mapper(item) if mapper is not None else item)

        return result_list
