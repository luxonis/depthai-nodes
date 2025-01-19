from abc import ABC, abstractmethod
from typing import Callable, Generic, Optional, Type, TypeVar

T = TypeVar("T")


class Parameter(Generic[T], ABC):
    def __init__(
        self,
        type: Type[T],
        getter: Callable[[], T],
        setter: Callable[[T], None],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self._getter = getter
        self._setter = setter
        self._type = type
        if description:
            self._decription = description
        else:  # Can be ommitted when porting the code to C++
            self._decription = setter.__doc__
        if name:
            self._name = name
        else:  # Can be ommitted when porting the code to C++
            self._name = setter.__name__

    def get(self) -> T:
        return self._getter()

    def set(self, value: T) -> None:
        self._setter(value)

    @property
    def description(self) -> Optional[str]:
        return self._decription

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def get_type_name(self) -> str:
        pass

    @abstractmethod
    def get_as_string(self) -> str:
        pass

    @abstractmethod
    def set_from_string(self, string_value: str):
        pass
