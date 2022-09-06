from __future__ import annotations

from collections import defaultdict
from typing import *

import metric
from basic import ConflictingValuesError, relu, default
from enum import Enum


class MetricalMeasure:
    def __init__(self, beats_per_measure: int, subbeats_per_beat: int,
                 length: int = 0, anacrusis: int = 0) -> None:
        super().__init__()
        self.bpm = beats_per_measure
        self.sbpb = subbeats_per_beat
        self.length = length
        self.anacrusis = anacrusis

    def get_beats_per_measure(self):
        return self.bpm

    def get_subbeats_per_beat(self):
        return self.sbpb

    def get_length(self):
        return self.length

    def get_anacrusis(self):
        return self.anacrusis

    @staticmethod
    def from_timesignature(timesignature: TimeSignature, tatums_denominator: int = 32):
        bpm = timesignature.get_beats_per_measure()
        subbeats_per_beat = 2

        if bpm > 3 and bpm % 3 == 0:
            bpm = bpm / 3
            subbeats_per_beat = 3

        tatums_per_measure = timesignature.get_tatums_per_measure(tatum_denominator=tatums_denominator)
        length = int(tatums_per_measure / bpm / subbeats_per_beat)

        return MetricalMeasure(beats_per_measure=bpm, subbeats_per_beat=subbeats_per_beat, length=length)

    @staticmethod
    def from_timepoint(timepoint_sequence: TimePointSequence, timepoint: TimePoint = None,
                       tatums_denominator: int = 32):
        if timepoint is None:
            timepoint = timepoint_sequence.first_point

        timesignature = timepoint_sequence.get_timesignature_for_timepoint(timepoint)
        measure = MetricalMeasure.from_timesignature(timesignature=timesignature,
                                                     tatums_denominator=tatums_denominator)

        if timepoint_sequence.get_subbeat_length() <= 0:
            measure.length = timepoint_sequence.get_subbeat_length()

        return measure


class Tatum:
    def __init__(self, tatum: int = 0, measure: int = 0, time: float = 0, tick: int = 0) -> None:
        super().__init__()
        self.tatum: int = tatum
        self.measure: int = measure
        self.time: float = time
        self.tick: int = tick

    def get_tatum(self) -> int:
        return self.tatum

    def get_measure(self) -> int:
        return self.measure

    def get_time(self) -> float:
        return self.time

    def get_tick(self) -> int:
        return self.tick


class Voice:
    def __init__(self, current_note: MusicNote, prev_voice: Voice = None):
        self.notes: List[MusicNote] = []
        self.prev: Voice = prev_voice
        self.most_recent_note: MusicNote = current_note

        self.first_note_time = current_note.get_start_point().get_time_at_tick()
        if self.prev is not None:
            self.first_note_time = self.prev.first_note_time
            self.notes.extend(self.prev.get_notes())

        self.notes.append(current_note)

    def get_previous(self) -> Optional[Voice]:
        return self.prev

    def get_most_recent_note(self) -> Optional[MusicNote]:
        return self.most_recent_note

    def get_notes(self) -> List[MusicNote]:
        return self.notes


class MusicNote:
    def __init__(self, start_tick: int, end_tick: int,
                 velocity: int, pitch: int,
                 track: int, channel: int):
        self.start_point = None
        self.end_point = None
        self.start_tick = start_tick
        self.end_tick = end_tick
        self.velocity = velocity
        self.pitch = pitch
        self.track = track
        self.channel = channel

    def set_start_point(self, tp: TimePoint) -> NoReturn:
        self.start_point = tp

    def get_start_point(self) -> TimePoint:
        return self.start_point

    def get_start_time(self):
        return None if self.get_start_point() is None else self.get_start_point().get_time_at_tick()

    def set_end_point(self, tp: TimePoint) -> NoReturn:
        self.end_point = tp

    def get_end_point(self) -> TimePoint:
        return self.end_point

    def get_end_time(self):
        return None if self.get_end_point() is None else self.get_end_point().get_time_at_tick()

    def get_start_tick(self) -> int:
        return self.start_tick

    def get_end_tick(self) -> int:
        return self.end_tick

    def get_velocity(self) -> int:
        return self.velocity

    def get_pitch(self) -> int:
        return self.pitch

    def get_track(self) -> int:
        return self.track

    def get_channel(self) -> int:
        return self.channel

    def get_onset_tatum(self, tatums: List[Tatum]) -> Optional[Tatum]:
        return self.get_nearest_tatum(time=self.get_start_time(), tatums=tatums)

    def get_offset_tatum(self, tatums: List[Tatum]) -> Optional[Tatum]:
        return self.get_nearest_tatum(time=self.get_start_time(), tatums=tatums)

    def get_nearest_tatum(self, time: float, tatums: List[Tatum]) -> Optional[Tatum]:
        if tatums is None or len(tatums) == 0:
            return None

        if len(tatums) == 1:
            return tatums[0]

        middle = int(len(tatums) / 2)
        left = self.get_nearest_tatum(time=time, tatums=tatums[0:middle])
        right = self.get_nearest_tatum(time=time, tatums=tatums[middle:-1])

        distance = float('inf')
        nearest = None

        for tatum in (left, right):
            if tatum is not None:
                abs_distance = abs(time - left.get_time())
                if abs_distance < distance:
                    distance = abs_distance
                    nearest = tatum

        return nearest


class Tempo:
    __DEFAULT_PPB = 480
    __DEFAULT_BPM = 120
    __DEFAULT_TEMPO = None

    def __init__(self, bpm: Optional[int] = None, ppb: Optional[int] = None,
                 migration_tempo: Optional[Tempo] = None, beat_denominator: int = 4) -> NoReturn:
        # If given values are 0 or a negative number set them to None since behaviour for these values is not defined
        # Therefore they should be replaced by default values
        if bpm is not None and bpm <= 0:
            bpm = None
        if ppb is not None and ppb <= 0:
            ppb = None

        # If the given qpm or ppq are None (or negative) then the values are set to the values from the
        # given migration_tempo or if no migration tempo is not given with the hardcoded defaults.
        self.bpm = default(bpm, default(migration_tempo,
                                        default_value=Tempo.__DEFAULT_BPM,
                                        mapper=lambda x: x.get_beats_per_minute()))
        self.ppb = default(ppb, default(migration_tempo,
                                        default_value=Tempo.__DEFAULT_PPB,
                                        mapper=lambda x: x.get_puls_per_beat()))

        self.beat_denominator = beat_denominator

    def __eq__(self, o: Tempo) -> bool:
        equal = (o is not None)  # if comparing object is None -> no equality

        if equal:
            # if comparing object has different type -> no equality
            equal = (isinstance(o, self.__class__))

        if equal:
            equal = (self.beat_denominator == o.beat_denominator)

        if equal:
            equal = (self.bpm == o.get_beats_per_minute())

        if equal:
            equal = (self.ppb == o.get_puls_per_beat())

        return equal

    def __hash__(self) -> int:
        return hash((self.bpm, self.ppb, self.beat_denominator))

    @staticmethod
    def get_default_tempo() -> Tempo:
        return DefaultTempo()

    def is_default(self):
        return False

    def get_beats_per_minute(self, target_beat_denominator: int = None) -> Optional[int]:
        if target_beat_denominator is None:
            return self.bpm

        # Our current Tempos bpm is 120 beats per quarter and our target_beat should in the future
        # be denominated by 2 (which means Half - 2/2, 3/2, etc.).
        # We would get a factor of 2.0 between target_beat and the definition_beat of our current Tempo object.
        factor = self.__get_factor__(target_beat_denominator=target_beat_denominator)
        # Therefor the original beats per minute must be adjusted by division with the factor.
        # Since 120 beats per quarter are 60 beats per half, etc.
        return int(self.bpm / factor)

    def set_beats_per_minute(self, bpm: Optional[int], beat_denominator: int = 4) -> NoReturn:
        # If given values are 0 or a negative number set them to None since behaviour for these values is not defined
        # Therefore they should be replaced by default values
        if bpm is not None and bpm <= 0:
            bpm = None
        self.bpm = default(bpm, Tempo.__DEFAULT_BPM)

        if beat_denominator != self.beat_denominator:
            # calculate new pulses per beat with new beat denominator
            new_ppb = self.get_puls_per_beat(target_beat_denominator=beat_denominator)
            # than change the current beat_denominator of this object
            # if done afterwards or never => results in infinity loop of the setters
            self.beat_denominator = beat_denominator
            # last set pulses per beat to new value
            self.set_puls_per_beat(ppb=new_ppb, beat_denominator=beat_denominator)

    def get_puls_per_beat(self, target_beat_denominator: int = None) -> Optional[int]:
        if target_beat_denominator is None:
            return self.ppb
        # Our current Tempos ppb is 480 pulses per quarter and our target_beat should in the future
        # be denominated by 2 (which means Half - 2/2, 3/2, etc.).
        # We would get a factor of 2.0 between target_beat and the definition_beat of our current Tempo object.
        factor = self.__get_factor__(target_beat_denominator=target_beat_denominator)
        # Therefor the original pulses per beat must be adjusted by multiplication with the factor.
        # Since 480 pulses per quarter are 960 pulses per half, etc.
        return self.ppb * factor

    def set_puls_per_beat(self, ppb: Optional[int], beat_denominator: int = 4) -> NoReturn:
        # If given values are 0 or a negative number set them to None since behaviour for these values is not defined
        # Therefore they should be replaced by default values
        if ppb is not None and ppb <= 0:
            ppb = None
        self.ppb = default(ppb, Tempo.__DEFAULT_PPB)

        if beat_denominator != self.beat_denominator:
            # calculate new beat per minutes with new beat denominator
            new_bpm = self.get_beats_per_minute(target_beat_denominator=beat_denominator)
            # than change the current beat_denominator of this object
            # if done afterwards or never => results in infinity loop of the setters
            self.beat_denominator = beat_denominator
            # last set beats per minute to new value
            self.set_beats_per_minute(bpm=new_bpm, beat_denominator=beat_denominator)

    def get_seconds_per_beat(self, target_beat_denominator: int = None) -> float:
        return 1 / self.get_beats_per_second(target_beat_denominator=target_beat_denominator)

    def get_milliseconds_per_beat(self, target_beat_denominator: int = None) -> float:
        return self.get_seconds_per_beat(target_beat_denominator=target_beat_denominator) * 1000

    def get_microseconds_per_beat(self, target_beat_denominator: int = None) -> float:
        return self.get_seconds_per_beat(target_beat_denominator=target_beat_denominator) * 1000000

    def get_beats_per_second(self, target_beat_denominator: int = None) -> float:
        beats_per_minute = self.get_beats_per_minute(target_beat_denominator=target_beat_denominator)
        return beats_per_minute / 60

    def get_ticks_per_second(self, target_beat_denominator: int = None) -> float:
        return 1 / self.get_seconds_per_tick(target_beat_denominator=target_beat_denominator)

    def get_seconds_per_tick(self, target_beat_denominator: int = None) -> float:
        seconds_per_beat = self.get_seconds_per_beat(target_beat_denominator=target_beat_denominator)
        pulses_per_beat = self.get_puls_per_beat(target_beat_denominator=target_beat_denominator)
        return seconds_per_beat / pulses_per_beat

    def get_ticks_per_microsecond(self, target_beat_denominator: int = None) -> float:
        return 1 / self.get_microseconds_per_tick(target_beat_denominator=target_beat_denominator)

    def get_microseconds_per_tick(self, target_beat_denominator: int = None) -> float:
        microseconds_per_beat = self.get_microseconds_per_beat(target_beat_denominator=target_beat_denominator)
        pulses_per_beat = self.get_puls_per_beat(target_beat_denominator=target_beat_denominator)
        return microseconds_per_beat / pulses_per_beat

    def __get_factor__(self, target_beat_denominator: int = None) -> float:
        if target_beat_denominator is None:
            return 1.0

        return self.beat_denominator / target_beat_denominator


class DefaultTempo(Tempo):
    def __init__(self):
        super().__init__()

    def is_default(self):
        return True


class TimeSignature:
    """
    Defines a TimeSignature in a music piece.

    Attributes
    ----------
    beats_per_measure : int
        The number of beats in a measure
    beat_length_denominator : int
        The denominator of the note length of a beat.  (4 for quarter notes, 2 for half notes, etc.)
    """
    __DEFAULT_BEATS_PER_MEASURE = 4
    __DEFAULT_BEAT_LENGTH_DENOMINATOR = 4

    def __init__(self, beats_per_measure: Optional[int] = None,
                 beat_length_denominator: Optional[int] = None) -> None:
        bpm_none = (beats_per_measure is None)
        bld_none = (beats_per_measure is None)

        if bpm_none ^ bld_none:
            raise ValueError("If beats per measure is set beat length denominator is required and vice versa!")

        # If given values are 0 or a negative number set them to None since behaviour for these values is not defined
        # Therefore they should be replaced by default values
        if beats_per_measure is not None and beats_per_measure <= 0:
            beats_per_measure = None
        if beat_length_denominator is not None and beat_length_denominator <= 0:
            beat_length_denominator = None

        # If the given beats_per_measure or beat_length are None (or negative) then the values are set to the values
        # from the given migration_signature or if no migration tempo is not given with the hardcoded defaults.
        self.beats_per_measure = default(beats_per_measure,
                                         default_value=TimeSignature.__DEFAULT_BEATS_PER_MEASURE)
        self.beat_length_denominator = default(beat_length_denominator,
                                               default_value=TimeSignature.__DEFAULT_BEAT_LENGTH_DENOMINATOR)

    def __eq__(self, o: TimeSignature) -> bool:
        equal = (o is not None)  # if comparing object is None -> no equality

        if equal:
            # if comparing object has different type -> no equality
            equal = (isinstance(o, self.__class__))

        if equal:
            equal = (self.beats_per_measure == o.get_beats_per_measure())

        if equal:
            equal = (self.beat_length_denominator == o.get_beat_length_denominator())

        return equal

    def __hash__(self) -> int:
        return hash((self.beats_per_measure, self.beat_length_denominator))

    @staticmethod
    def get_default_timesignature() -> TimeSignature:
        return DefaultTimeSignature()

    def is_default(self):
        return False

    def get_beats_per_measure(self) -> int:
        return self.beats_per_measure

    def get_beat_length_denominator(self) -> int:
        return self.beat_length_denominator

    def get_tatums_per_measure(self, tatum_denominator: int = 32) -> int:
        return self.beats_per_measure * self.get_tatums_per_measure_beat(tatum_denominator=tatum_denominator)

    def get_tatums_per_measure_beat(self, tatum_denominator: int = 32) -> int:
        return tatum_denominator / self.beat_length_denominator

    def get_subbeats_per_beat(self) -> int:
        bpm = self.beats_per_measure
        sbpb = 2

        if bpm > 3 and bpm % 3 == 0:
            bpm /= 3
            sbpb = 3

        return sbpb


class DefaultTimeSignature(TimeSignature):
    def __init__(self) -> None:
        super().__init__()

    def is_default(self):
        return True


class KeySignature:
    def __init__(self) -> None:
        super().__init__()


class TimePoint:
    def __init__(self, previous_point: TimePoint = None, tick: int = None, time_at_tick: float = None) -> NoReturn:
        self.previous_point: TimePoint = previous_point
        self.tick: int = tick
        self.time_at_tick: float = time_at_tick
        self.next_point: TimePoint = None
        if previous_point is not None:
            self.previous_point.set_next(self)

    def __eq__(self, o: TimePoint) -> bool:
        equal = (o is not None)  # if comparing object is None -> no equality

        if equal:
            # if comparing object has different type -> no equality
            equal = (isinstance(o, self.__class__))

        if equal:
            equal = (self.tick == o.get_tick())

        return equal

    def __hash__(self) -> int:
        return hash(self.tick)

    def __lt__(self, other: TimePoint) -> bool:
        return self.tick < other.get_tick()

    def __le__(self, other: TimePoint) -> bool:
        return self.tick <= other.get_tick()

    def __ge__(self, other: TimePoint) -> bool:
        return self.tick > other.get_tick()

    def __gt__(self, other: TimePoint) -> bool:
        return self.tick >= other.get_tick()

    def get_tick(self) -> int:
        return self.tick

    def get_time_at_tick(self) -> float:
        return self.time_at_tick

    def set_time_at_tick(self, time_at_tick: float) -> None:
        self.time_at_tick = time_at_tick

    def get_next(self) -> TimePoint:
        return self.next_point

    def set_next(self, next_point: TimePoint) -> None:
        self.next_point = next_point

    def get_previous(self) -> TimePoint:
        return self.previous_point

    def set_previous(self, previous_point: TimePoint) -> None:
        self.previous_point = previous_point


class TimePointSequence:
    timepoint_data: defaultdict[int, defaultdict[Type[Union[None, TimePoint, Tempo, TimeSignature, MusicNote]],
                                                 Union[None, TimePoint, Tempo, TimeSignature, defaultdict]]]

    def __init__(self, subbeat_length: int, anacrusis_length: int = 0, use_channel: bool = False,
                 max_notes: int = 1) -> NoReturn:
        self.anacrusis_length: int = anacrusis_length
        self.subbeat_length: int = self.__validate_subbeat_length(subbeat_length)
        self.timepoint_data = defaultdict(lambda: defaultdict(lambda: None))
        self.first_point: Optional[TimePoint] = None
        self.last_point: Optional[TimePoint] = None
        self.note_separation_attribute = 'channel' if use_channel else 'track'
        self.max_notes = max_notes

    def add_timepoint(self, tick: int,
                      bpm: Optional[int] = None, ppb: Optional[int] = None,
                      time_signature: Optional[TimeSignature] = None,
                      notes: List[MusicNote] = None) -> NoReturn:
        """
        Inserts a new timepoint into the timepoint sequence at the given tick and sets the musical
        metadata like tempos and time signature of the measures. If no metadata is given for a specific
        timepoint the values from the previous timepoints are used from the previous timepoint.

        Its also possible to set beginning musicnotes when inserting a timepoint or one can use the method
        TimePointSequence.add_notes(#).

        :param tick: The number of the puls at which this information was retrieved
        :param bpm: beats per minute - Amount of quarter notes that are counted in a minute
        :param ppb: pulses per beat - How many pulses are retrieved from one quarter to another
        :param time_signature: value of the beats inside a measure and how many of this beats make a measure (4/4 or 6/8 measure)
        :param notes: The notes that begin at the timepoint
        """
        if tick is None:
            raise ValueError("tick must not be None!")

        if not self.has_timepoint_at_tick(tick):
            new_time_point = TimePoint(tick=tick)
            prev_tp, next_tp = self.get_neighbouring_timepoints(tick)

            # Correct references of new timepoint and neigbouring timepoints
            if prev_tp is not None:
                new_time_point.set_previous(prev_tp)
                prev_tp.set_next(new_time_point)
            else:
                self.first_point = new_time_point

            if next_tp is not None:
                new_time_point.set_next(next_tp)
                next_tp.set_previous(new_time_point)
            else:
                self.last_point = new_time_point

            # Generate/retrieve metadata objects like tempo and time signature
            tempo = self.get_tempo_for_timepoint(prev_tp)
            if ppb is not None or bpm is not None:
                tempo = Tempo(bpm=bpm, ppb=ppb, migration_tempo=tempo)
            elif tempo is None:
                tempo = Tempo.get_default_tempo()

            if time_signature is None:
                time_signature = self.get_timesignature_for_timepoint(prev_tp)
                if time_signature is None:
                    time_signature = TimeSignature.get_default_timesignature()

            # add timepoint and additional information to the data array
            self.timepoint_data[tick][TimePoint] = new_time_point
            self.timepoint_data[tick][Tempo] = tempo
            self.timepoint_data[tick][TimeSignature] = time_signature
            self.timepoint_data[tick][MusicNote] = defaultdict(lambda: list())

            # Propagate changes in metadata in chain till next explicit metadata change
            self.__propagate_data_forward(next_tp, propagation_class=Tempo)
            self.__propagate_data_forward(next_tp, propagation_class=TimeSignature)

            # Add given notes
            self.add_notes(tick=tick, notes=notes)

            # Set time in seconds for each tick after the inserted one
            self.for_every_timepoint(lambda tp: self.calculate_time_for_timepoint(tp), new_time_point)

        else:
            raise ValueError(f"Wanting to insert already existing "
                             f"tick {tick} with [{bpm}, {ppb},{time_signature},{notes}]")

    def add_notes(self, tick: int, notes: List[MusicNote] = None) -> NoReturn:
        if tick is None:
            raise ValueError("tick must not be None!")

        if notes is None:
            return

        note_dict = self.get_notes_for_tick(tick=tick)
        start_timepoint = self.get_timepoint_for_tick(tick)

        for n in notes:
            if n is None:
                continue

            if tick != n.start_tick:
                raise ConflictingValuesError("Given notes contains a note that doesn't start at given tick!")

            # Add Note for the right track/channel
            channel_track = getattr(n, self.note_separation_attribute, -1)
            note_list = note_dict[channel_track]
            if len(note_list) > self.max_notes:
                raise ValueError(f"Timepoint can only take {self.max_notes} at max for "
                                 f"each {self.note_separation_attribute}.")

            # Add ending timepoint if not exist
            if not self.has_timepoint_at_tick(n.get_end_tick()):
                self.add_timepoint(tick=n.get_end_tick())

            end_timepoint = self.get_timepoint_for_tick(n.get_end_tick())

            # set pointers of musid note to timepoints
            n.set_start_point(start_timepoint)
            n.set_end_point(end_timepoint)

            note_list.append(n)

    def has_timepoint_at_tick(self, tick: int):
        """
        Returns True if the sequence has already a Timepoint and metadata defined at the given tick
        :param tick: Tick/Puls that should be checked for existing timepoint
        """
        return len(self.timepoint_data[tick]) > 0

    def check_compatibility_of_metadata(self, tick: int,
                                        bpm: Optional[int] = None, ppb: Optional[int] = None,
                                        time_signature: Optional[TimeSignature] = None):
        """
        Checks if given metadata is compatible with the already existing data.

        :param tick:
        :param bpm:
        :param ppb:
        :param time_signature:
        :return: True if compatible, otherwise False
        """
        # If not tick is given no comparison value => The given data is compatible
        if tick is None:
            return True

        # If no metadata is given to check => Since no changes from the original data assumed as True
        if bpm is None and ppb is None and time_signature is None:
            return True

        # If there is no timepoint defined at this tick => also no comparison value defined
        current_timepoint = self.get_timepoint_for_tick(tick)
        if current_timepoint is None:
            return True

        # compatible metadata means same quarter per minute, pulse per quarter and time signature
        compatible = True
        if bpm is not None or ppb is not None:
            tempo = self.get_tempo_for_timepoint(current_timepoint)
            if bpm is not None:
                compatible = compatible and (bpm == tempo.get_beats_per_minute())

            if ppb is not None:
                compatible = compatible and (ppb == tempo.get_puls_per_beat())

        if time_signature is not None:
            time_signature_current = self.get_timesignature_for_timepoint(current_timepoint)
            compatible = compatible and (time_signature_current == time_signature)

        return compatible

    def __propagate_data_forward(self, timepoint: TimePoint,
                                 propagation_class: Type[Union[Tempo, TimeSignature]] = None) -> NoReturn:
        """
        If no metadata changes are defined, the sequence assumes that the values from the previous timepoint
        continue at new inserted timepoints. If however the metadata is changed then all timepoints from the
        new/inserted one to next one with explicit change must also use this new metadata for further calculations.

        This metadata changes the references from the old metadata objects to the new ones.
        :param timepoint: starting time point whichs values should be propageted
        :param propagation_class: which type of metadata should be propagated
        """
        if timepoint is None:
            return

        next_timepoint = timepoint.get_next()
        if next_timepoint is None:
            # If we have no ancestor the given object is the last one and no propagation is needed.
            return

        next_obj = self.timepoint_data[next_timepoint.get_tick()][propagation_class]

        prev_timepoint = timepoint.get_previous()
        if prev_timepoint is not None:
            prev_obj = self.timepoint_data[prev_timepoint.get_tick()][propagation_class]
            if prev_obj is not next_obj:
                # If new time_point is inserted between two existing timepoints and these two have not the same
                # object reference. Then no propagation is needed.
                # Simplified: A change is inserted between two changes.
                return

        current_obj = self.timepoint_data[timepoint.get_tick()][propagation_class]
        if current_obj.is_default():
            return

        if prev_timepoint is None and not next_obj.is_default():
            return

        self.__reset_references(next_obj, current_obj, next_timepoint)

    """
    Defines a combined Type for Compiler hints. No real significance to the code
    """
    T = TypeVar('T', Tempo, TimeSignature)

    def __reset_references(self, old_reference_object: T, new_reference_object: T, starting_time_point: TimePoint):
        """
        Resets the references of the metadata, from an old object to a new one. References are changed
        from starting timepoint till the old_reference object doesn't equal the metadata object at the
        later timepoints in the sequence.

        :param old_reference_object:
        :param new_reference_object:
        :param starting_time_point:
        :return:
        """
        current_tp = starting_time_point
        while current_tp is not None:
            current_obj = self.timepoint_data[current_tp.get_tick()][old_reference_object.__class__]
            if current_obj is old_reference_object:
                self.timepoint_data[current_tp.get_tick()][old_reference_object.__class__] = new_reference_object
            else:
                break
            current_tp = current_tp.get_next()

    def calculate_time_for_timepoint(self, timepoint: TimePoint):
        """
        Time in seconds is calculated, via the following formula.
            ->  time_at_previous_timepoint + (time_needed_per_tick * delta_ticks)

        The time_needed_ per_ tick is defined by the tempo of the previous timepoint. Since the tempo is valid till it
        my be changed at a later timepoint. The first possible change would be the given timepoint.

        :param timepoint:
        """
        if timepoint is None:
            return None

        prev_tp: TimePoint = timepoint.get_previous()

        if prev_tp is not None:
            # If the previous timepoint is None, then the given Timepoint is the starting point. The tempo of the starting
            # point is assumed by the author to define the tempo from tick 0 to the tick of the first ancestor point of
            # the starting point.
            tempo_at_prev_tp: Tempo = self.get_tempo_for_timepoint(default(prev_tp, default_value=timepoint))

            tick_at_prev_tp: int = default(prev_tp.get_tick(), default_value=0)
            time_at_prev_tp: float = default(prev_tp.get_time_at_tick(), default_value=0)

            delta_ticks = timepoint.get_tick() - tick_at_prev_tp
            delta_time = tempo_at_prev_tp.get_seconds_per_tick() * delta_ticks

            timepoint.set_time_at_tick(time_at_prev_tp + delta_time)
        else:
            timepoint.set_time_at_tick(0)

    def for_every_timepoint(self, function: Callable[TimePoint],
                            starting_point: Optional[TimePoint] = None,
                            break_condition: Callable[TimePoint, bool] = lambda: False) -> NoReturn:
        """
        Calls given function for each TimePoint in the TimePointSequence
        :param break_condition:
        :param starting_point:
        :param function:
        :return:
        """
        current_tp = default(starting_point, self.first_point)
        while current_tp is not None and not break_condition():
            function(current_tp)
            current_tp = current_tp.get_next()

    def get_neighbouring_timepoints(self, tick: int) -> Tuple[Optional[TimePoint], Optional[TimePoint]]:
        """
        Retruns the neighbouring timepoints for a given tick
        :param tick:
        :return:
        """
        if tick is None:
            raise ValueError("Given tick must not be None!")

        if self.first_point is None and self.last_point is None:
            return None, None

        if self.first_point.get_tick() > tick:
            return None, self.first_point

        if self.last_point.get_tick() < tick:
            return self.last_point, None

        lower_tp: Optional[TimePoint] = self.first_point
        while lower_tp is not None and lower_tp.get_next().get_tick() < tick:
            lower_tp = lower_tp.get_next()

        return lower_tp, lower_tp.next_point

    def get_timepoint_for_tick(self, tick: int = None) -> Optional[TimePoint]:
        if self.first_point is None:
            return None

        if tick is None:
            return None

        return self.timepoint_data[tick][TimePoint]

    def get_timesignature_for_tick(self, tick: int = None) -> Optional[TimeSignature]:
        if self.first_point is None:
            return None

        if tick is None:
            return None

        return self.timepoint_data[tick][TimeSignature]

    def get_timesignature_for_timepoint(self, timepoint: TimePoint = None) -> Optional[TimeSignature]:
        if self.first_point is None:
            return None

        if timepoint is None:
            return None

        return self.get_timesignature_for_tick(timepoint.get_tick())

    def get_tempo_for_tick(self, tick: int = None) -> Optional[Tempo]:
        if self.first_point is None:
            return None

        if tick is None:
            return None

        return self.timepoint_data[tick][Tempo]

    def get_tempo_for_timepoint(self, timepoint: TimePoint = None) -> Optional[Tempo]:
        if self.first_point is None:
            return None

        if timepoint is None:
            return None

        return self.get_tempo_for_tick(timepoint.get_tick())

    def get_notes_for_tick(self, tick: int, channel_track: Optional[int] = None) -> Union[
        None, List[MusicNote], defaultdict[int, List[MusicNote]]]:
        if self.first_point is None:
            return None

        if tick is None:
            return None

        all_notes = self.timepoint_data[tick][MusicNote]
        return all_notes if channel_track is None else all_notes[channel_track]

    def get_notes_for_timepoint(self, timepoint: TimePoint, channel_track: Optional[int] = None) -> Union[
        None, List[MusicNote], defaultdict[int, List[MusicNote]]]:
        if self.first_point is None:
            return None

        if timepoint is None:
            return None

        return self.get_notes_for_tick(timepoint.get_tick(), channel_track=channel_track)

    def get_concated_note_list(self, timepoint: TimePoint) -> List[MusicNote]:
        notes: defaultdict[int, List[MusicNote]] = self.get_notes_for_timepoint(timepoint=timepoint)
        out = []
        for sublist in notes.values():
            out.extend(sublist)
        return out

    def get_tatums(self) -> List[Tatum]:
        tatums: List[Tatum] = []
        measure_num, tatum_num, tick, timepoint = -1, -1, 0, self.first_point
        last_tick, prev_tick, prev_time_at_tatum = self.last_point.get_tick(), 0, 0

        while timepoint.get_next() is not None and timepoint.get_tick() <= last_tick:
            time_measure_data = self.time_measure_data_of_timepoint(timepoint)
            tatums_per_measure = time_measure_data[metric.TimePointSequence.TimeMeasureData.TATUMS_PER_MEASURE]
            ticks_per_tatum = time_measure_data[metric.TimePointSequence.TimeMeasureData.TICKS_PER_TATUM]
            seconds_per_tick = time_measure_data[metric.TimePointSequence.TimeMeasureData.SECONDS_PER_TICK]

            if tatum_num < 0:
                if self.anacrusis_length > 0:
                    anacrusis_tatums = int(self.anacrusis_length / ticks_per_tatum)
                    tatum_num = tatums_per_measure - anacrusis_tatums
                    # tick = -self.anacrusis_length
                else:
                    measure_num = 0

            next_tp = timepoint.get_next()
            while tick <= last_tick and tick <= next_tp.get_tick():
                # calculate time and add tatum to list
                tick_delta = tick - relu(prev_tick)
                time_at_tatum = relu(prev_time_at_tatum + (seconds_per_tick * tick_delta))
                tatums.append(Tatum(tatum=tatum_num, measure=measure_num, time=time_at_tatum, tick=tick))

                # change tatum and measure number for next loop
                tatum_num += 1
                if tatum_num >= tatums_per_measure:
                    tatum_num = 0
                    measure_num += 1

                # save prev time and tick for delta-calculation in next loop
                prev_tick = tick
                prev_time_at_tatum = time_at_tatum

                # increase tick to next tatum
                tick += ticks_per_tatum

            timepoint = next_tp

        return tatums

    def time_measure_data_of_timepoint(self, timepoint: TimePoint):
        if timepoint is None:
            return None

        time_signature = self.get_timesignature_for_timepoint(timepoint)
        tempo = self.get_tempo_for_timepoint(timepoint)

        return self.__get_time_measure_data(time_signature, tempo)

    def time_measure_data_of_tick(self, tick: int):
        # tick of anacrusis
        if tick < 0:
            point = self.first_point
            return self.time_measure_data_of_timepoint(timepoint=point)
        time_signature = self.get_timesignature_for_tick(tick)
        tempo = self.get_tempo_for_tick(tick)

        return self.__get_time_measure_data(time_signature, tempo)

    class TimeMeasureData(Enum):
        TEMPO = 'tempo'
        TIME_SIGNATURE = 'time_signature'
        TATUMS_PER_BEAT = 'tatums_per_beat'
        TATUMS_PER_SUBBEAT = 'tatums_per_subbeat'
        TATUMS_PER_MEASURE = 'tatums_per_measure'
        TICKS_PER_BEAT = 'ticks_per_beat'
        TICKS_PER_TATUM = 'ticks_per_tatum'
        SECONDS_PER_TICK = 'seconds_per_tick'
        SECONDS_PER_TATUM = 'seconds_per_tatum'

    def __get_time_measure_data(self, time_signature: TimeSignature, tempo: Tempo):
        # correct tatums to subbeat structure
        tatums_per_subbeat = self.subbeat_length
        tatums_per_beat = tatums_per_subbeat * time_signature.get_subbeats_per_beat()
        tatums_per_measure = tatums_per_beat * time_signature.get_beats_per_measure()

        beat_denominator = time_signature.get_beat_length_denominator()
        ticks_per_beat = tempo.get_puls_per_beat(target_beat_denominator=beat_denominator)
        ticks_per_tatum = ticks_per_beat / tatums_per_beat

        seconds_per_tick = tempo.get_seconds_per_tick(target_beat_denominator=beat_denominator)
        seconds_per_tatum = ticks_per_tatum * seconds_per_tick

        return {
                TimePointSequence.TimeMeasureData.TEMPO: tempo,
                TimePointSequence.TimeMeasureData.TIME_SIGNATURE: time_signature,
                TimePointSequence.TimeMeasureData.TATUMS_PER_SUBBEAT: tatums_per_subbeat,
                TimePointSequence.TimeMeasureData.TATUMS_PER_BEAT: tatums_per_beat,
                TimePointSequence.TimeMeasureData.TATUMS_PER_MEASURE: tatums_per_measure,
                TimePointSequence.TimeMeasureData.TICKS_PER_BEAT: ticks_per_beat,
                TimePointSequence.TimeMeasureData.TICKS_PER_TATUM: ticks_per_tatum,
                TimePointSequence.TimeMeasureData.SECONDS_PER_TICK: seconds_per_tick,
                TimePointSequence.TimeMeasureData.SECONDS_PER_TATUM: seconds_per_tatum
                }

    def get_subbeat_length(self):
        return self.subbeat_length

    def get_anacrusis_subbeats(self, tatum_denominator: int = 32):
        time_measure_data = self.time_measure_data_of_timepoint(self.first_point)
        first_time_signature = time_measure_data[TimePointSequence.TimeMeasureData.TIME_SIGNATURE]

        ticks_per_tatum = time_measure_data[TimePointSequence.TimeMeasureData.TICKS_PER_TATUM]
        tatums_per_measure = time_measure_data[TimePointSequence.TimeMeasureData.TATUMS_PER_MEASURE]

        measure = MetricalMeasure.from_timesignature(timesignature=first_time_signature,
                                                     tatums_denominator=tatum_denominator)
        subbeats_per_measure = measure.get_beats_per_measure() * measure.get_subbeats_per_beat()

        ticks_per_measure = ticks_per_tatum * tatums_per_measure
        ticks_per_subbeat = ticks_per_measure / subbeats_per_measure

        return self.anacrusis_length / ticks_per_subbeat

    def validate_sequence(self, allowed_beat_measures: List[int],
                          allowed_beat_length_denominator: List[int],
                          single_timesignature: bool = True) -> NoReturn:
        # @todo implement
        pass

    @staticmethod
    def __validate_subbeat_length(subbeat_length: int) -> int:
        if type(subbeat_length) != int:
            raise TypeError(f"Given parameter 'subbeat_length' not an integer but "
                            f"an {type(subbeat_length)}")

        return -1 if subbeat_length == 0 else subbeat_length
