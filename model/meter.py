from __future__ import annotations
from typing import List, Tuple, OrderedDict as OrderedDictType, DefaultDict, Optional
from collections import OrderedDict, defaultdict

from metric import MusicNote, TimePointSequence
from model.base import MidiModel, MidiModelState
from model.beat import TatumTrackingModelState, TatumTrackingGrammarModelState
from model.voice import VoiceSplittingModelState, VoiceSplittingGrammarModelState
from model.hierarchy import HierarchyModelState, HierarchyGrammarModelState


class MeterModel(MidiModel):
    new_voice_states: DefaultDict[VoiceSplittingModelState, List[VoiceSplittingModelState]]
    new_tatum_states: DefaultDict[TatumTrackingModelState, DefaultDict[Tuple[MusicNote], List[TatumTrackingModelState]]]

    def __init__(self, hierarchy_state: HierarchyModelState = None,
                 voice_state: VoiceSplittingModelState = None,
                 tatum_state: TatumTrackingModelState = None,
                 beam_size: int = 200):
        super(MeterModel).__init__()
        self.beam_size = -1 if beam_size is None else beam_size
        self.started = False

        self.new_voice_states = defaultdict(lambda: [])
        self.new_tatum_states = defaultdict(lambda: defaultdict(lambda: []))

        self.started_states: OrderedDictType[MeterModelState, None] = OrderedDict()
        self.hypothesis_states: OrderedDictType[MeterModelState, None] = OrderedDict()
        mms = MeterModelState(meter_model=self,
                              hierarchy_state=hierarchy_state,
                              voice_state=voice_state,
                              tatum_state=tatum_state)
        self.hypothesis_states.update({mms: None})

    def get_beam_size(self) -> int:
        return self.beam_size

    def is_beam_full(self) -> bool:
        return (not self.beam_size <= -1) and (len(self.started_states) >= self.beam_size)

    def transition(self, notes: List[MusicNote] = None) -> OrderedDictType[MeterModelState]:
        return self.__transition_close_worker(notes=notes, none_as_close=False)

    def close(self) -> OrderedDictType[MeterModelState]:
        return self.__transition_close_worker(notes=None, none_as_close=True)

    def __transition_close_worker(self, notes: List[MusicNote] = None, none_as_close=False) -> OrderedDictType[MeterModelState]:
        new_states: OrderedDict[MeterModelState, None] = OrderedDict()
        do_close = (notes is None and none_as_close)

        self.started_states = OrderedDict()
        self.new_tatum_states = defaultdict(lambda: defaultdict(lambda: []))
        self.new_voice_states = defaultdict(lambda: [])

        if not self.started:
            self.started = True

        for mms in self.hypothesis_states:
            if do_close:
                ts = mms.close()
            else:
                ts = mms.transition(notes)

            for ns in ts.keys():
                new_states.update({ns: None})

                if ns.is_started():
                    self.started_states.update({ns: None})

            ## fixForBeam ????
            if not do_close:
                #todo logging
                pass

        self.hypothesis_states = new_states
        return self.hypothesis_states

    def get_hypotheses(self) -> OrderedDictType[MidiModelState, None]:
        pass


class MeterGrammarModel(MeterModel):
    def __init__(self, sequence: TimePointSequence, beam_size: int = 200):
        hs: HierarchyModelState = HierarchyGrammarModelState(sequence=sequence)
        vs: VoiceSplittingModelState = VoiceSplittingGrammarModelState(sequence=sequence)
        ts: TatumTrackingModelState = TatumTrackingGrammarModelState(sequence=sequence)

        super(MeterGrammarModel, self).__init__(hierarchy_state=hs, voice_state=vs, tatum_state=ts, beam_size=beam_size)


class MeterPredictionModel(MeterModel):
    def __init__(self, sequence: TimePointSequence, beam_size: int = 200):
        super(MeterPredictionModel, self).__init__(beam_size=beam_size)
        pass


class MeterModelState(MidiModelState):
    def __init__(self, meter_model: MeterModel,
                 hierarchy_state: HierarchyModelState,
                 voice_state: VoiceSplittingModelState = None,
                 tatum_state: TatumTrackingModelState = None) -> None:
        super(MeterModelState).__init__()
        tts_none = tatum_state is None
        vss_none = voice_state is None

        # voice_state and tatum_state must be both set or both None. One set and one None is not allowed.
        if tts_none == vss_none:
            self.meter_model: MeterModel = meter_model

            self.voice_splitting_state: VoiceSplittingModelState = voice_state
            self.hierarchy_state: HierarchyModelState = hierarchy_state
            self.tatum_tracking_state: TatumTrackingModelState = tatum_state

            # this branch can only be reached if both values have the same truth value
            # therefore just checking one is enough
            if tts_none:
                self.voice_splitting_state = hierarchy_state.get_voice_splitting_state()
                self.tatum_tracking_state = self.hierarchy_state.get_tatum_tracking_state().deep_copy()
                self.hierarchy_state = self.hierarchy_state.deep_copy()

            self.tatum_tracking_state.set_hierarchy_state(self.hierarchy_state)
        else:
            raise ValueError("Given value-combination is not supported")

    def set_tatum_tracking_state(self, state: TatumTrackingModelState):
        self.tatum_tracking_state = state

    def set_voice_splitting_state(self, state: VoiceSplittingModelState):
        self.voice_splitting_state = state

    def set_hierarchy_state(self, state: HierarchyModelState):
        self.hierarchy_state = state

    def transition(self, notes: List[MusicNote] = None) -> OrderedDictType[MeterModelState]:
        return self.__transition_close_worker(notes=notes, none_as_close=False)

    def close(self) -> OrderedDictType[MeterModelState]:
        return self.__transition_close_worker(notes=None, none_as_close=True)

    def __transition_close_worker(self, notes: List[MusicNote] = None, none_as_close=False) -> OrderedDictType[MeterModelState, None]:
        new_state: OrderedDictType[MeterModelState, None] = OrderedDict()
        do_close = (notes is None and none_as_close)

        beam_full = self.meter_model.is_beam_full()
        if beam_full:
            voice_outside_beam = self.get_score() < next(reversed(self.meter_model.started_states)).get_score()
            if voice_outside_beam:
                return new_state

        new_voice_states: List[VoiceSplittingModelState] = self.meter_model.new_voice_states[self.voice_splitting_state]
        if len(new_voice_states) == 0:
            if do_close:
                state_dict = self.voice_splitting_state.close()
            else:
                state_dict = self.voice_splitting_state.transition(notes)

            new_voice_states = [*state_dict.keys()]
            self.meter_model.new_voice_states.update({self.voice_splitting_state: new_voice_states})

        new_tatum_states = []
        new_notes_list = []
        for v_state in new_voice_states:

            if beam_full:
                beat_score = v_state.get_score() + \
                             self.tatum_tracking_state.get_score() + \
                             self.tatum_tracking_state.get_hierarchy_state().get_score()
                tatum_outside_beam = beat_score <= next(reversed(self.meter_model.started_states)).get_score()
                if tatum_outside_beam:
                    if not do_close:
                        new_notes_list.append([])
                    new_tatum_states.append([])
                    continue

            tatum_state_copy = self.tatum_tracking_state.deep_copy()
            tatum_state_copy.set_hierarchy_state(self.tatum_tracking_state.hierarchy_state)

            if do_close:
                nts = tatum_state_copy.close()
                nts = [*nts.keys()]
                new_tatum_states.append(nts)
            else:
                new_notes: List[MusicNote] = []
                for n in notes:
                    if self.voice_splitting_state.keep_note(n):
                        new_notes.append(n)
                new_notes_list.append(new_notes)

                if tatum_state_copy.is_started():
                    nts = tatum_state_copy.transition(notes)
                    nts = [*nts.keys()]
                    new_tatum_states.append(nts)
                else:
                    tatums_map = self.meter_model.new_tatum_states[tatum_state_copy]
                    if len(tatums_map) == 0:
                        pass

                    branched_states = tatums_map[tuple(new_notes)]
                    if len(branched_states) == 0:
                        bs = tatum_state_copy.transition(new_notes)
                        branched_states = [*bs.keys()]
                        tatums_map.update({tuple(new_notes): branched_states})

                    new_tatum_states.append(branched_states)

        for i in range(len(new_tatum_states)):
            new_voice_state = new_voice_states[i]
            tatum_states = new_tatum_states[i]
            nnotes = new_notes_list[i] if not do_close else []

            new_states_tmp = []

            for tstate in tatum_states:
                beat_score = new_voice_state.get_score() + \
                             tstate.get_score() + \
                             tstate.get_hierarchy_state().get_score()

                if beam_full:
                    state_outside_beam = next(reversed(self.meter_model.started_states)).get_score() >= beat_score
                    if state_outside_beam:
                        # todo: Logging
                        continue

                hierarchy_state_copy: HierarchyModelState = self.hierarchy_state.deep_copy()
                hierarchy_state_copy.set_voice_splitting_state(new_voice_state)
                hierarchy_state_copy.set_tatum_tracking_state(tstate)
                tstate.set_hierarchy_state(hierarchy_state_copy)

                if not do_close:
                    # todo: Special case
                    pass

                if do_close:
                    new_hierarchy_states = hierarchy_state_copy.close()
                else:
                    new_hierarchy_states = hierarchy_state_copy.transition(nnotes)

                new_hierarchy_states = [*new_hierarchy_states.keys()]
                for hms in new_hierarchy_states:
                    self.add_new_state(hms, new_states_tmp, check_duplicate=True)

                for mms in new_states_tmp:
                    new_state.update({mms: None})

        return new_state

    def get_score(self) -> float:
        vs_score = 0 if self.voice_splitting_state is None else self.voice_splitting_state.get_score()
        tt_score = 0 if self.tatum_tracking_state is None else self.tatum_tracking_state.get_score()
        h_score = 0 if self.hierarchy_state is None else self.hierarchy_state.get_score()

        return vs_score + tt_score + h_score

    def add_new_state(self, hms: HierarchyModelState,
                      temp_states_list: List[MeterModelState] = [], check_duplicate: bool = True):

        mms = MeterModelState(meter_model=self.meter_model, hierarchy_state=hms)

        if not check_duplicate:
            temp_states_list.append(mms)
        else:
            duplicated_state: Optional[MeterModelState] = None

            for s in temp_states_list:
                if s.is_duplicate_of(mms):
                    duplicated_state = s
                    break

            if duplicated_state is not None:
                if duplicated_state.get_score() < mms.get_score():
                    temp_states_list.append(mms)
                    temp_states_list.remove(duplicated_state)

                    # todo add logging for removal of duplicated_state
                else:
                    # todo add logging for removal of mms
                    pass
            else:
                temp_states_list.append(mms)

    def is_duplicate_of(self, state: MeterModelState) -> bool:
        hierachy_duplicate = self.hierarchy_state.is_duplicate_of(state.hierarchy_state)
        tatum_duplicate = self.tatum_tracking_state.is_duplicate_of(state.tatum_tracking_state)
        voice_duplicate = True

        # if something change voice_duplicate (todo at MeterDetection phase)
        # voice_duplicate = self.voice_splitting_state.is_duplicate_of(state.voice_splitting_state)

        return hierachy_duplicate and tatum_duplicate and voice_duplicate

    def get_measure_count(self):
        return self.tatum_tracking_state.get_measure_count()

    def is_started(self) -> bool:
        measure_count = self.get_measure_count()
        return measure_count > 0
