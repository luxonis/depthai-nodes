from __future__ import annotations

from typing import Generic, List, Optional, Type, TypeVar

import depthai as dai

T = TypeVar("T")


class Collection(dai.Buffer, Generic[T]):
    """
    A generic DepthAI message containing a list of items of a single type T.

    Notes:
    - Python generics are erased at runtime, so runtime type checking is inferred
      from the first item when the collection becomes non-empty.
    """

    def __init__(self, items: List[T]):
        super().__init__()
        self._item_cls: Optional[Type[T]] = type(items[0]) if items else None
        self.items: List[T] = items

    @property
    def item_cls(self) -> Optional[Type[T]]:
        """The inferred runtime type for items, once known."""
        return self._item_cls

    @property
    def items(self) -> List[T]:
        return self._items

    @items.setter
    def items(self, value: List[T]) -> None:
        if not isinstance(value, list):
            raise TypeError(f"items must be a list, got {type(value)}")

        if self._item_cls is None and value:
            self._item_cls = type(value[0])

        if self._item_cls is not None:
            bad = [type(v) for v in value if not isinstance(v, self._item_cls)]
            if bad:
                raise TypeError(
                    f"All items must be of type {self._item_cls.__name__}, got: {bad}"
                )

        self._items = value

    def append(self, item: T) -> None:
        if self._item_cls is None:
            self._item_cls = type(item)
        elif not isinstance(item, self._item_cls):
            raise TypeError(
                f"Item must be of type {self._item_cls.__name__}, got {type(item)}"
            )
        self._items.append(item)

    def extend(self, items: List[T]) -> None:
        # Reuse setter validation
        self.items = [*self._items, *items]
