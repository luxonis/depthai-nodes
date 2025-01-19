from typing import Callable, Optional

from .parameter import Parameter


class FloatParameter(Parameter[float]):
    def __init__(
        self,
        getter: Callable[[], float],
        setter: Callable[[float], None],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        super().__init__(float, getter, setter, name, description)

    def get_type_name(self) -> str:
        return "float"  # Haven't used self._type.__name__ in order to make the C++ conversion easy

    def get_as_string(self) -> str:
        return str(self.get())

    def set_from_string(self, string_value: str):
        value = float(string_value)
        return self.set(value)
