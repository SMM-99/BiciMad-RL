from dataclasses import dataclass

@dataclass
class Station:
    id: int
    name: str
    capacity: int
    bikes: int
