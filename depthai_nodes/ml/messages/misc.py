import depthai as dai

from ..messages import Classifications


class AgeGender(dai.Buffer):
    def __init__(self):
        super().__init__()
        self._age: float = None
        self._gender = Classifications()

    @property
    def age(self) -> float:
        return self._age

    @age.setter
    def age(self, value: float):
        if not isinstance(value, float):
            raise TypeError(
                f"start_point must be of type float, instead got {type(value)}."
            )
        self._age = value

    @property
    def gender(self) -> Classifications:
        return self._gender

    @gender.setter
    def gender(self, value: Classifications):
        if not isinstance(value, Classifications):
            raise TypeError(
                f"gender must be of type Classifications, instead got {type(value)}."
            )
        self._gender = value
