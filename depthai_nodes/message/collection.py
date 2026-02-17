from __future__ import annotations

from typing import Generic, List, Optional, Type, TypeVar

import depthai as dai

T = TypeVar("T")


class Collection(dai.Buffer, Generic[T]):
    """
    A generic DepthAI message containing a list of items of a single type T.

    Notes:
    - Python generics are erased at runtime, so if you want runtime type safety
      you need to pass/attach the item type (item_cls).
    """

    def __init__(self, item_cls: Type[T], items: List[T]):
        super().__init__()
        self._item_cls: Type[T] = item_cls
        self.items: List[T] = items

    @property
    def item_cls(self) -> Optional[Type[T]]:
        """The enforced runtime type for items (optional)."""
        return self._item_cls

    @property
    def items(self) -> List[T]:
        return self._items

    @items.setter
    def items(self, value: List[T]) -> None:
        if not isinstance(value, list):
            raise TypeError(f"items must be a list, got {type(value)}")

        bad = [type(v) for v in value if not isinstance(v, self._item_cls)]
        if bad:
            raise TypeError(
                f"All items must be of type {self._item_cls.__name__}, got: {bad}"
            )

        self._items = value

    def append(self, item: T) -> None:
        if not isinstance(item, self._item_cls):
            raise TypeError(
                f"Item must be of type {self._item_cls.__name__}, got {type(item)}"
            )
        self._items.append(item)

    def extend(self, items: List[T]) -> None:
        # Reuse setter validation
        self.items = [*self._items, *items]
