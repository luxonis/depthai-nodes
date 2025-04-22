class RequireSuperInit(type):
    """Checking whether every class in the hierarchy calls super().__init__() inside
    their __init__ method."""

    def __call__(cls, *args, **kwargs):
        inst = cls.__new__(cls, *args, **kwargs)
        inst._b_init_called = False
        cls.__init__(inst, *args, **kwargs)

        if not getattr(inst, "_b_init_called", False):
            raise RuntimeError(
                f"{cls.__name__}.__init__ didn't call super().__init__()!"
            )
        return inst


class NodeMock(metaclass=RequireSuperInit):
    def __init__(self):
        self._b_init_called = True  # the metaclass catches if it wasn’t set during construction (i.e. if one of the child classes didn’t call super().__init__())
