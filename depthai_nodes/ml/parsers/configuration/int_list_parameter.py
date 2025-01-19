from typing import Callable, List, Optional

from .parameter import Parameter


class IntListParameter(Parameter[List[int]]):
    def __init__(
        self,
        getter: Callable[[], List[int]],
        setter: Callable[[List[int]], None],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        super().__init__(List[int], getter, setter, name, description)

    def get_type_name(self) -> str:
        return "list[int]"  # Haven't used typing.get_origin in order to make the C++ conversion easy

    def get_as_string(self) -> str:
        return str(self.get())

    def set_from_string(self, string_value: str):
        clean_str = string_value.strip("[]")
        value = [int(x) for x in clean_str.split(",")]
        return self.set(value)
