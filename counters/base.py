from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class CounterState:
    reps: int = 0
    state: str = 'INIT'
    extras: Dict[str, Any] = field(default_factory=dict)

class BaseCounter:
    def __init__(self, name:str):
        self.name = name
        self.s = CounterState(reps=0, state='INIT', extras={})

    def update(self, kps, angles:Dict[str,float]):
        raise NotImplementedError

    def snapshot(self):
        return dict(reps=self.s.reps, state=self.s.state, **self.s.extras)
