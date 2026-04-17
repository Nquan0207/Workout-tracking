from .base import BaseCounter
from typing import Dict

class PushupCounter(BaseCounter):
    def __init__(self, top_elbow=145, bottom_elbow=110, hold_frames=1, min_trunk_angle=0, min_knee_angle=0,
                 ready_frames=1, arm_frames=1, min_rep_gap_frames=10, min_cycle_frames=8):
        super().__init__('pushup')
        self.top_elbow = top_elbow
        self.bottom_elbow = bottom_elbow
        self.hold_frames = hold_frames
        self.min_trunk_angle = min_trunk_angle
        self.min_knee_angle = min_knee_angle
        self.ready_frames = ready_frames
        self.hold = 0
        self._ready_count = 0
        self.arm_frames = arm_frames
        self._armed_count = 0
        self.min_rep_gap_frames = int(min_rep_gap_frames)
        self.min_cycle_frames = int(min_cycle_frames)
        self._frame_idx = 0
        self._last_rep_frame = -10**9
        self._cycle_start_frame = None
        self.s.state = 'TOP'

    def update(self, kps, angles:Dict[str,float]):
        self._frame_idx += 1
        elbow = min(angles['left_elbow'], angles['right_elbow'])
        trunk = angles.get('trunk_angle', 0.0)
        knee_min = angles.get('knee_angle_min', 180.0)
        if trunk < self.min_trunk_angle or knee_min < self.min_knee_angle:
            self._ready_count = 0
            self._armed_count = 0
        else:
            self._ready_count = min(self.ready_frames, self._ready_count + 1)
        ready = (self._ready_count >= self.ready_frames)
        if not ready:
            self.s.state = 'TOP'
            self.hold = 0
            self._armed_count = 0
            self.s.extras.clear()
            self.s.extras.update({
                'elbow_angle': round(float(elbow),2),
                'trunk_angle': round(float(trunk),2),
                'knee_angle_min': round(float(knee_min),2),
                'ready': False,
                'armed': False
            })
            return self.snapshot()
        armed_prev = (self._armed_count >= self.arm_frames)
        if self.s.state == 'TOP':
            if elbow < self.top_elbow and armed_prev:
                self.s.state = 'DESCENT'
                self._armed_count = 0
                self._cycle_start_frame = self._frame_idx
            elif elbow >= self.top_elbow:
                self._armed_count = min(self.arm_frames, self._armed_count + 1)
            else:
                self._armed_count = 0
        else:
            self._armed_count = 0

        if self.s.state == 'DESCENT':
            if elbow <= self.bottom_elbow:
                self.s.state = 'BOTTOM'
                self.hold = 1
        elif self.s.state == 'BOTTOM':
            if elbow <= self.bottom_elbow:
                self.hold += 1
            if elbow > self.bottom_elbow and self.hold >= self.hold_frames:
                self.s.state = 'ASCENT'
        elif self.s.state == 'ASCENT':
            if elbow >= self.top_elbow:
                self.s.state = 'TOP'
                cycle_ok = True
                if self._cycle_start_frame is not None:
                    cycle_ok = (self._frame_idx - self._cycle_start_frame) >= self.min_cycle_frames
                gap_ok = (self._frame_idx - self._last_rep_frame) >= self.min_rep_gap_frames
                if cycle_ok and gap_ok:
                    self.s.reps += 1
                    self._last_rep_frame = self._frame_idx
                self._cycle_start_frame = None
                self._armed_count = 0

        armed_display = (self._armed_count >= self.arm_frames)
        self.s.extras.clear()
        self.s.extras.update({
            'elbow_angle': round(float(elbow),2),
            'trunk_angle': round(float(trunk),2),
            'knee_angle_min': round(float(knee_min),2),
            'ready': True,
            'armed': armed_display
        })
        return self.snapshot()
