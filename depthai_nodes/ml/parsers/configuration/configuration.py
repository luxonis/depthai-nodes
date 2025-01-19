from dataclasses import dataclass
from typing import Optional


@dataclass
class Configuration:
    parameter: str
    string_value: str
    type_name: str
    description: Optional[str]
