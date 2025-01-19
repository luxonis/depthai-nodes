from typing import Any, Dict, List

from .parameter import Parameter


class RuntimeParameters:
    def __init__(self):
        self._parameters: Dict[str, Parameter[Any]] = {}

    def add(self, parameter: Parameter[Any]) -> "RuntimeParameters":
        self._parameters[parameter.name] = parameter
        return self

    def get(self, name: str) -> Parameter[Any]:
        return self._parameters[name]

    def get_names(self) -> List[str]:
        return list(self._parameters.keys())
