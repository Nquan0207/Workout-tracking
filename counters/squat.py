from .base import BaseCounter
from typing import Dict

class SquatCounter(BaseCounter):
    def __init__(self, top_deg=160, bottom_deg=90, hold_frames=3):
        super().__init__('squat')
        self.top_deg = top_deg
        self.bottom_deg = bottom_deg
        self.hold_frames = hold_frames
        self.hold = 0
        self.s.state = 'TOP'

    def update(self, kps, angles:Dict[str,float]):
        knee = min(angles['left_knee'], angles['right_knee'])  # use deeper side
        if self.s.state == 'TOP':
            if knee < self.top_deg:
                self.s.state = 'DESCENT'
        elif self.s.state == 'DESCENT':
            if knee <= self.bottom_deg:
                self.s.state = 'BOTTOM'
                self.hold = 1
        elif self.s.state == 'BOTTOM':
            if knee <= self.bottom_deg:
                self.hold += 1
            if knee > self.bottom_deg and self.hold >= self.hold_frames:
                self.s.state = 'ASCENT'
        elif self.s.state == 'ASCENT':
            if knee >= self.top_deg:
                self.s.state = 'TOP'
                self.s.reps += 1
        self.s.extras['knee_angle'] = round(float(knee), 2)
        return self.snapshot()
