from .base import BaseCounter
from typing import Dict

class PullupCounter(BaseCounter):
    def __init__(self, top_elbow_max=70, bottom_elbow_min=160, hold_frames=1):
        super().__init__('pullup')
        self.top_elbow_max = top_elbow_max
        self.bottom_elbow_min = bottom_elbow_min
        self.top_exit_elbow_min = min(self.bottom_elbow_min - 5, self.top_elbow_max + 15)
        self.hold_frames = hold_frames
        self.hold = 0
        self.rep_armed = True
        self.s.state = 'BOTTOM'

    def update(self, kps, angles:Dict[str,float]):
        left_elbow = float(angles['left_elbow'])
        right_elbow = float(angles['right_elbow'])
        elbow = (left_elbow + right_elbow) / 2.0
        if self.s.state == 'BOTTOM':
            if elbow < self.bottom_elbow_min:
                self.s.state = 'ASCENT'
                self.hold = 0
        elif self.s.state == 'ASCENT':
            if elbow <= self.top_elbow_max:
                self.hold += 1
                if self.hold >= self.hold_frames:
                    self.s.state = 'TOP'
                    if self.rep_armed:
                        self.s.reps += 1
                        self.rep_armed = False
            elif elbow >= self.bottom_elbow_min:
                self.s.state = 'BOTTOM'
                self.hold = 0
        elif self.s.state == 'TOP':
            if elbow <= self.top_elbow_max:
                self.hold += 1
            elif elbow >= self.top_exit_elbow_min:
                self.s.state = 'DESCENT'
        elif self.s.state == 'DESCENT':
            if elbow >= self.bottom_elbow_min:
                self.s.state = 'BOTTOM'
                self.hold = 0
                self.rep_armed = True
            elif elbow <= self.top_elbow_max:
                self.s.state = 'TOP'
                self.hold = 1
        self.s.extras['elbow_angle'] = round(float(elbow), 2)
        self.s.extras['left_elbow'] = round(left_elbow, 2)
        self.s.extras['right_elbow'] = round(right_elbow, 2)
        self.s.extras['top_exit_elbow'] = round(float(self.top_exit_elbow_min), 2)
        return self.snapshot()
