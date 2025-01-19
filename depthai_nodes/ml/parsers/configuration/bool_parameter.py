from typing import Callable, Optional

from .parameter import Parameter


class BoolParameter(Parameter[bool]):
    def __init__(
        self,
        getter: Callable[[], bool],
        setter: Callable[[bool], None],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        super().__init__(bool, getter, setter, name, description)

    def get_type_name(self) -> str:
        return "bool"  # Haven't used self._type.__name__ in order to make the C++ conversion easy

    def get_as_string(self) -> str:
        return str(self.get())

    def set_from_string(self, string_value: str):
        value = bool(string_value)
        return self.set(value)
