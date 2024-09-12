from typing import Tuple

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


class VehicleAttributes(dai.Buffer):
    def __init__(self):
        super().__init__()
        self._vehicle_type: Tuple[str, float] = ("", 0)
        self._vehicle_color: Tuple[str, float] = ("", 0)

    @property
    def vehicle_type(self) -> Tuple[str, float]:
        return self._vehicle_type

    @vehicle_type.setter
    def vehicle_type(self, value: Tuple[str, float]):
        if not isinstance(value, Tuple):
            raise TypeError(f"vehicle_type must be a Tuple, instead got {type(value)}.")
        if not isinstance(value[0], str) and not isinstance(value[1], float):
            raise TypeError(
                f"vehicle_type must be a Tuple of (str, float), got ({value[0], value[1] })."
            )

        self._vehicle_type = value

    @property
    def vehicle_color(self) -> Tuple[str, float]:
        return self._vehicle_color

    @vehicle_color.setter
    def vehicle_color(self, value: Tuple[str, float]):
        if not isinstance(value, Tuple):
            raise TypeError(
                f"vehicle_color must be a Tuple, instead got {type(value)}."
            )
        if not isinstance(value[0], str) and not isinstance(value[1], float):
            raise TypeError(
                f"vehicle_color must be a Tuple of (str, float), got ({value[0], value[1] })."
            )

        self._vehicle_color = value
