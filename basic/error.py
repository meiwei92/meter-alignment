class ConflictingValuesError(ValueError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class MissingValuesError(ValueError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class InvalidFormatError(ValueError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
