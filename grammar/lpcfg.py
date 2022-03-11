import sys
import argparse

from typing import *
from pathlib import Path
from parsing import MidiFileParser, KernFileParser, NoteBFileParser, FileParser
from model.meter import MeterModel, MeterGrammarModel
from metric.base import TimePointSequence, MusicNote

""" 
List defines the supported formats that are available to save the resulting grammar file
"""
SUPPORTED_GRAMMAR_FILE_EXTENSION = ['.lpcfg']

"""
Dictionary for the implemented EventParser. 
The entry 'supported' points at a sub-dictionary, which holds a mapping of EventParser types to 
the file formats that can be parsed
The entry 'default' indicates the default type that should be used as EventParser
"""
EVENT_PARSERS = {
    'supported': {
        MidiFileParser: ['.midi', '.mid'],
        KernFileParser: ['.krn'],
        NoteBFileParser: ['.nb']
    },
    'default': MidiFileParser
}

ANACRUSIS_FILE_EXTENSION = '.anacrusis'


class Grammar:
    """

    """

    def __init__(self) -> None:
        super().__init__()

    def serialize(self):
        pass


class GrammarGenerator:
    """

    """

    def __init__(self, grammar: Path, anacrusis: Path, files: Path,
                 use_channel: bool = True, lexicalization: bool = False,
                 exclude_trees: bool = False, processes: int = 1,
                 not_extended: bool = False, verbose: bool = False,
                 subbeat_length: int = 4, min_note_length: int = 100000) -> None:
        self.use_channel = use_channel
        self.lexicalization = lexicalization
        self.exclude_trees = exclude_trees
        self.processes = processes
        self.not_extended = not_extended
        self.verbose = verbose
        self.subbeat_length = subbeat_length
        self.min_note_length = min_note_length
        self.grammar_path = grammar
        self.anacrusis_path = anacrusis
        self.music_file_path = files

    @staticmethod
    def validate_grammar_path(path: Path) -> bool:
        if path is None:
            raise ValueError("Given parameter path can not be None!")
        if path.suffix not in SUPPORTED_GRAMMAR_FILE_EXTENSION:
            raise ValueError(f"Path extention '{path.suffix}' is not supported, must be one of "
                             f"'{', '.join(SUPPORTED_GRAMMAR_FILE_EXTENSION)}'.")
        return True

    @staticmethod
    def __get_files_from_path(path: Path) -> List[Path]:
        if path is None:
            raise ValueError("Given parameter path can not be None!")
        if path.is_file():
            return [path]
        else:
            return list(filter(lambda x: x.is_file(), path.rglob('*')))

    @staticmethod
    def __get_anacrusis_length(music_file: Path, anacrusis_path: Path, verbose: bool = False) -> int:
        """
        Method retrieves the anacrusis lenght from the referenced anacrusis file. If the given anacusis_path
        references a file with the correct extension '.anacrusis', then this file is used. If the anacrusis_path
        references a directory, then a file in the form '<music_file_name>.anacrusis' is needed. If its not found a
        default anacrusis length of 0 is returned. If a file could be derived then the content of the file is returned.

        Example: music_file has name 'test-fugue.mid' then the method lokks for a 'test-fugue.mid.anacrusis' file.

        :param music_file: Given Music file, for example a Midi-File
        :param anacrusis_path: Given Path to an specific anacrusis file or a folder which contains multiple anacrusis
        files
        :param verbose: if set the algorithm outputs more information
        :return: The derived anacrusis length of the given music file
        """
        if anacrusis_path is None:
            raise ValueError("Given parameter anacrusis_path can not be None!")
        if music_file is None:
            raise ValueError("Given parameter music_file can not be None!")

        if anacrusis_path.is_file() and anacrusis_path.name.endswith(ANACRUSIS_FILE_EXTENSION):
            # If given path is a file and ends with '.anacrusis' use the given file
            anacrusis_file = anacrusis_path
        elif anacrusis_path.is_file():
            raise ValueError(f"Given anacrusis_path '{anacrusis_path.name}' points to a file, but has the wrong "
                             f"file-extension. '{ANACRUSIS_FILE_EXTENSION}'-file is needed!")
        else:
            # else look if the given directory holds a file that has the name '<musicfile_name>.anacrusis'
            anacrusis_file = next(anacrusis_path.rglob(f"{music_file.name}{ANACRUSIS_FILE_EXTENSION}"), None)

        value = 0
        if anacrusis_path is not None:
            with open(anacrusis_file) as f:
                file_value = f.readline()
                try:
                    value = int(file_value)
                except ValueError:
                    if verbose:
                        print(f"File '{anacrusis_file.name}' contains value '{file_value}', which cannot be "
                              f"converted to an integer. Using therefore 0 as default!")
        return value

    def parse_file(self) -> Grammar:
        pass

    def generate_grammar(self) -> Grammar:
        if self.processes != 1:
            raise NotImplementedError("Multithreading is not implemented yet!!")

        music_file_list = self.__get_files_from_path(self.music_file_path)
        for file_num, mfile in enumerate(music_file_list, start=1):
            if self.verbose:
                print(f"Parsing {file_num}/{len(music_file_list)}: {mfile}")

            anacrusis_length = self.__get_anacrusis_length(mfile, self.anacrusis_path, self.verbose)
            parser_type = self.__get_parser_type_for_file(file=mfile)
            parser = parser_type(file=mfile, anacrusis_length=anacrusis_length,
                                 use_channel=self.use_channel, verbose=self.verbose,
                                 subbeat_length=self.subbeat_length, min_note_length=self.min_note_length)
            time_point_sequence: TimePointSequence = parser.parse()
            model: MeterModel = MeterGrammarModel(time_point_sequence)

            current_tp = time_point_sequence.first_point
            while current_tp.get_next() is not None:
                notes: List[MusicNote] = time_point_sequence.get_concated_note_list(timepoint=current_tp)
                if len(notes) > 0:
                    model.transition(notes=notes)
                current_tp = current_tp.get_next()

        # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        #     executor.map()
        result_grammar = Grammar()
        return result_grammar

    @staticmethod
    def __get_parser_type_for_file(file: Path) -> Type:
        """
        Gets an music file and derives the associated EventParser type
        :param file: Path to the file which should be parsed
        :return: Type of the EventParser for given file
        """
        parser_dict = EVENT_PARSERS['supported']
        parser_class = None

        # iterate over the supported file extensions and derive EventParser type if possible
        for p_class, supported_extensions in parser_dict.items():
            if file.suffix in supported_extensions:
                if parser_class is not None:
                    raise ValueError(f"{file.suffix} defined for '{parser_class}' "
                                     f"and {p_class}!")
                parser_class = p_class

        # if file extension not supported try the default parser
        if parser_class is None:
            parser_class = EVENT_PARSERS['default']

        return parser_class


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Function returns an Argumentparser for the commandline parameters.
    :return:
    """
    argument_parser = argparse.ArgumentParser(description='Build grammar for given Files.')

    argument_parser.add_argument('-s', '--subbeat_length', metavar="N", type=int, default=4,
                                 help='Defined the subbeat length.')

    argument_parser.add_argument('-m', '--min_note_length', metavar="N", type=int, default=100000,
                                 help='Defines the minimal length of a note in Milliseconds. Doesn\'t take shorter '
                                      'notes into account while generating the grammar file.')

    argument_parser.add_argument('-T', '--track', dest='use_channel',
                                 action="store_const", const=False, default=True,
                                 help='???')

    argument_parser.add_argument('-l', '--lexicalization', action="store_const",
                                 const=True, default=False,
                                 help='Do NOT use lexicalization.')

    argument_parser.add_argument('-x', '--exclude_trees', action="store_const",
                                 const=True, default=False,
                                 help='If set, trees are not saved inside the grammar file')

    argument_parser.add_argument('-g', '--grammar', metavar="FILEPATH", required=True, type=Path,
                                 help='Path to the output file')

    argument_parser.add_argument('-p', '--processes', metavar="INT", type=int, default=1,
                                 help='Number of processes that work parallel')

    argument_parser.add_argument('-ne', '--not-extended', action="store_const", const=True, default=False,
                                 help='Do NOT extend each note within each voice to the next note\'s onset.')

    argument_parser.add_argument('-f', '--files', metavar="FILE/DIRPATH", type=Path, default=Path("."),
                                 help='Path to the music file(s) [Midi]')

    argument_parser.add_argument('-a', '--anacrusis', metavar="FILE/DIRPATH", type=Path, default=Path("."),
                                 help='Path to the anacrusis file(s)')

    argument_parser.add_argument('-v', '--verbose', action="store_const", const=True, default=False,
                                 help='If set more output in the console')

    return argument_parser


if __name__ == "__main__":

    # _perf = ptt.load_performance_midi("../data/corpora/WTCInv/invent1.mid")
    # _perf = ptt.load_performance_midi("../data/corpora/WTCInv/bach-0867-fugue.mid")
    # _score = ptt.load_score_midi("../data/corpora/WTCInv/invent1.mid")
    # _score = ptt.load_score_midi("../data/corpora/WTCInv/bach-0867-fugue.mid")
    # print("Loaded")
    parser = create_argument_parser()
    args = vars(parser.parse_args(sys.argv[1:]))

    generator = GrammarGenerator(**args)
    grammar = generator.generate_grammar()
    grammar.serialize()
