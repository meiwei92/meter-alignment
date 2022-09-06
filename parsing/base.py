from pathlib import Path
from metric import TimePointSequence
# from base import JsonReprObject


class FileParser:
    def __init__(self, file: Path, anacrusis_length: int,
                 use_channel: bool = True, verbose: bool = False,
                 subbeat_length: int = 4, min_note_length: int = 100000) -> None:
        self.file = file
        self.anacrusis_length = anacrusis_length
        self.use_channel = use_channel
        self.verbose = verbose
        self.subbeat_length = subbeat_length
        self.min_note_length = min_note_length

    def parse(self) -> TimePointSequence:
        raise NotImplementedError("Method parse must be implemented!")

    def set_parameters(self):
        raise NotImplementedError("Method set_parameters must be implemented!")
