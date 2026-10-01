from __future__ import annotations

from typing import Generic, TypeVar

import depthai as dai

from .utils.copy_message import copy_message

T = TypeVar("T")


class Collection(dai.Buffer, Generic[T]):
    """Store a list of items of one runtime type in a DepthAI message.

    The first item establishes the accepted type. Later assignments, appends, and
    extensions must contain instances of that type. An initially empty collection infers
    its type when the first item is added.

    Args:
        items: Initial list of items.

    Attributes:
        items: Stored items. Assignment validates their types.
        item_cls: Inferred runtime item type, or ``None`` until an item is added.

    Raises:
        TypeError: If ``items`` is not a list or contains incompatible item types.
    """

    def __init__(self, items: list[T]):
        """Initialize the collection and infer its runtime item type.

        Args:
            items: Initial items. An empty list defers type inference until the first
                insertion.
        """
        super().__init__()
        self._item_cls: type[T] | None = type(items[0]) if items else None
        self.items: list[T] = items

    @property
    def item_cls(self) -> type[T] | None:
        """The inferred runtime type for items, once known."""
        return self._item_cls

    @property
    def items(self) -> list[T]:
        """Return the stored list of items.

        The list is mutable. Direct list mutations bypass collection type checks; use
        ``append()``, ``extend()``, or assignment to ``items`` when validation is
        needed.
        """
        return self._items

    @items.setter
    def items(self, value: list[T]) -> None:
        """Replace the item list after checking runtime types.

        Args:
            value: Replacement list. An empty list preserves the previously inferred
                item type.

        Raises:
            TypeError: If the value is not a list or contains items incompatible with
                the inferred type.
        """
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
        """Append one item, inferring the type if necessary.

        Args:
            item: Item compatible with the type of the first item ever inserted.

        Raises:
            TypeError: If the item is incompatible with the inferred runtime type.
        """
        if self._item_cls is None:
            self._item_cls = type(item)
        elif not isinstance(item, self._item_cls):
            raise TypeError(
                f"Item must be of type {self._item_cls.__name__}, got {type(item)}"
            )
        self._items.append(item)

    def extend(self, items: list[T]) -> None:
        # Reuse setter validation
        """Append items after validating the combined list.

        Args:
            items: Items to append. Existing contents remain unchanged if validation
                fails.

        Raises:
            TypeError: If any item is incompatible with the inferred runtime type.
        """
        self.items = [*self._items, *items]

    def copy(self) -> Collection:
        """Copy items and preserve container timestamps and sequence number.

        Returns:
            A new ``Collection`` with items copied through ``copy_message``. This
            returns the base container even when called on a subclass; subclass-specific
            fields are not copied.

        Raises:
            TypeError: If an item cannot be copied by ``copy_message``.
        """
        new_list = []
        for item in self.items:
            new_list.append(copy_message(item))
        new_collection = Collection(new_list)
        new_collection.setSequenceNum(self.getSequenceNum())
        new_collection.setTimestampDevice(self.getTimestampDevice())
        new_collection.setTimestamp(self.getTimestamp())
        return new_collection
