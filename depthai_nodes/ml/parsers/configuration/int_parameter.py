from typing import Callable, Optional

from .parameter import Parameter


class IntParameter(Parameter[int]):
    def __init__(
        self,
        getter: Callable[[], int],
        setter: Callable[[int], None],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        super().__init__(int, getter, setter, name, description)

    def get_type_name(self) -> str:
        return "int"  # Haven't used self._type.__name__ in order to make the C++ conversion easy

    def get_as_string(self) -> str:
        return str(self.get())

    def set_from_string(self, string_value: str):
        value = int(string_value)
        return self.set(value)
