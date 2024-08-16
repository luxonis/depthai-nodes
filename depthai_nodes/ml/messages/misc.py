from typing import List

import depthai as dai


class AgeGender(dai.Buffer):
    def __init__(self):
        super().__init__()
        self._age: float = None
        self._gender_prob: List[float] = None

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
    def gender_prob(self) -> List[float]:
        return self._gender_prob

    @gender_prob.setter
    def gender_prob(self, value: List[float]):
        if not isinstance(value, List):
            raise TypeError(
                f"gender_prob must be of type List, instead got {type(value)}."
            )
        for item in value:
            if not isinstance(item, float):
                raise TypeError(
                    f"gender_prob list values must be of type float, instead got {type(value)}."
                )
        self._gender_prob = value
