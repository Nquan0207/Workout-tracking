from typing import Dict, List, Optional
import cv2, json, numpy as np, time

from ..pose.mediapipe_pose import MediaPipePoseEstimator
from ..smoothing.pose_smoother import PoseSmoother
from ..metrics.angles import knee_angle, elbow_angle, trunk_angle
from ..rules.engine import evaluate_rules
from ..counters.squat import SquatCounter
from ..counters.pushup import PushupCounter
from ..counters.bicep_curl import BicepCurlCounter
from ..counters.pullup import PullupCounter
from ..exercises.specs import SPECS
from ..utils.naming import MEDIAPIPE_LANDMARKS
from ..utils.voice import VoiceAnnouncer
from .overlay import draw_pose, draw_hud

class Analyzer:
    def __init__(self, exercise:str='squat', fps:float=30.0, overlay=False, voice=False, voice_name=None):
        self.exercise = exercise
        self.pose = MediaPipePoseEstimator()
        self.smoother = PoseSmoother(freq=fps, min_cutoff=0.7, beta=0.02)
        self.overlay = overlay
        self.voice = VoiceAnnouncer(enabled=voice, voice_name=voice_name)
        spec = SPECS[exercise]
        if exercise == 'squat':
            self.counter = SquatCounter(top_deg=spec['thresholds']['top_knee_min'],
                                        bottom_deg=spec['thresholds']['bottom_knee_max'],
                                        hold_frames=spec['thresholds']['hold_frames'])
        elif exercise == 'pushup':
            self.counter = PushupCounter(top_elbow=spec['thresholds']['top_elbow_min'],
                                         bottom_elbow=spec['thresholds']['bottom_elbow_max'],
                                         hold_frames=spec['thresholds']['hold_frames'],
                                         min_trunk_angle=spec['thresholds'].get('min_trunk_angle', 0),
                                         min_knee_angle=spec['thresholds'].get('min_knee_angle', 0),
                                         ready_frames=spec['thresholds'].get('ready_frames', 1),
                                         arm_frames=spec['thresholds'].get('arm_frames', 1),
                                         min_rep_gap_frames=spec['thresholds'].get('min_rep_gap_frames', 10),
                                         min_cycle_frames=spec['thresholds'].get('min_cycle_frames', 8))
        elif exercise == 'bicep_curl':
            self.counter = BicepCurlCounter(top_elbow_max=spec['thresholds']['top_elbow_max'],
                                            bottom_elbow_min=spec['thresholds']['bottom_elbow_min'],
                                            hold_frames=spec['thresholds']['hold_frames'],
                                            max_trunk_angle=spec['thresholds'].get('max_trunk_angle', 25))
        elif exercise == 'pullup':
            self.counter = PullupCounter(top_elbow_max=spec['thresholds']['top_elbow_max'],
                                         bottom_elbow_min=spec['thresholds']['bottom_elbow_min'],
                                         hold_frames=spec['thresholds']['hold_frames'])
        else:
            raise ValueError('Unsupported exercise')
        self.rules = spec['form_rules']
        self.rule_labels = {rule['id']: rule.get('label', rule['id']) for rule in self.rules}
        # Advisory-only feedback state (decoupled from rep counting).
        self._feedback_bad_streak = {}
        self._feedback_good_streak = {}
        self._feedback_active = set()
        self._feedback_bad_frames = 3
        self._feedback_good_frames = 3

    def _angles(self, kps:Dict[str, tuple]):
        out = {}
        # knee / elbow min across sides for ROM metrics
        lk, rk = knee_angle('left', kps), knee_angle('right', kps)
        le, re = elbow_angle('left', kps), elbow_angle('right', kps)
        out['left_knee'], out['right_knee'] = lk, rk
        out['left_elbow'], out['right_elbow'] = le, re
        out['knee_angle_min'] = min(lk, rk)
        out['elbow_angle_min'] = min(le, re)
        out['trunk_angle'] = trunk_angle(kps)
        return out

    def _hud_origin(self, kps:Dict[str, tuple], frame_shape:tuple):
        H, W = frame_shape[:2]
        points = [(kp[0], kp[1]) for kp in kps.values() if len(kp) > 3 and kp[3] > 0.4]
        if not points:
            return (10, 30)
        xs, ys = zip(*points)
        x = int(np.clip(min(xs) - 40, 10, max(10, W - 220)))
        y = int(np.clip(min(ys) - 40, 30, max(30, H - 30)))
        return (x, y)

    def _announce_rep_if_needed(self, prev_reps:int, current_reps:int):
        if current_reps > prev_reps:
            for rep in range(prev_reps + 1, current_reps + 1):
                self.voice.announce_rep(rep)

    def _form_assessment(self, rule_flags:Dict[str, bool]):
        if not self.rules:
            return {
                'form_ok': None,
                'form_status': 'unchecked',
                'violations': [],
            }
        failed = [rid for rid, ok in rule_flags.items() if not ok]
        return {
            'form_ok': len(failed) == 0,
            'form_status': 'correct' if not failed else 'incorrect',
            'violations': failed,
        }

    def _feedback_assessment(self, counter_snapshot:Dict, rule_flags:Dict[str, bool]):
        # Keep feedback independent from counting logic:
        # - "observing" when user is not in a valid exercise-ready pose.
        # - temporal hysteresis to avoid frame-level flicker and keep CPU cheap (O(num_rules)).
        if not self.rules:
            return {'form_ok': None, 'form_status': 'unchecked', 'violations': []}

        ready = bool(counter_snapshot.get('ready', True))
        if not ready:
            return {'form_ok': None, 'form_status': 'observing', 'violations': sorted(self._feedback_active)}

        for rid, ok in rule_flags.items():
            if ok:
                self._feedback_bad_streak[rid] = 0
                self._feedback_good_streak[rid] = self._feedback_good_streak.get(rid, 0) + 1
                if self._feedback_good_streak[rid] >= self._feedback_good_frames:
                    self._feedback_active.discard(rid)
            else:
                self._feedback_good_streak[rid] = 0
                self._feedback_bad_streak[rid] = self._feedback_bad_streak.get(rid, 0) + 1
                if self._feedback_bad_streak[rid] >= self._feedback_bad_frames:
                    self._feedback_active.add(rid)

        if self._feedback_active:
            return {
                'form_ok': False,
                'form_status': 'needs_improvement',
                'violations': sorted(self._feedback_active),
            }
        return {'form_ok': True, 'form_status': 'good', 'violations': []}

    def _overlay_lines(self, counter_snapshot:Dict, form:Dict):
        txt = [f"{self.exercise} | Reps: {counter_snapshot['reps']} State: {counter_snapshot['state']}"]
        if form['form_ok'] is None:
            txt.append('Form: Unchecked')
            return txt
        txt.append(f"Form: {'OK' if form['form_ok'] else 'Wrong'}")
        if form['violations']:
            txt.append('Issues: ' + ', '.join(self.rule_labels.get(rid, rid) for rid in form['violations']))
        return txt

    def run_on_video(self, in_path:str, out_overlay_path:Optional[str]=None, json_out:Optional[str]=None):
        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open {in_path}')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = None
        if self.overlay and out_overlay_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_overlay_path, fourcc, fps, (W,H))
        timeline = []
        fidx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pose = self.pose.infer(frame, timestamp_ms=int(round(fidx * 1000.0 / fps)))
            if pose is None:
                if writer is not None:
                    draw_hud(frame, [f'No person detected', f'Reps: {self.counter.s.reps}'])
                    writer.write(frame)
                fidx += 1
                continue
            kps = self.smoother.smooth(pose.keypoints)
            angles = self._angles(kps)
            prev_reps = int(self.counter.s.reps)
            counter_snapshot = self.counter.update(kps, angles)
            self._announce_rep_if_needed(prev_reps, int(counter_snapshot['reps']))
            metrics = {
                'knee_angle_min': angles['knee_angle_min'],
                'elbow_angle_min': angles['elbow_angle_min'],
                'trunk_angle': angles['trunk_angle'],
            }
            rule_flags = evaluate_rules(
                self.rules,
                metrics,
                context={'state': counter_snapshot.get('state')},
            )
            form = self._feedback_assessment(counter_snapshot, rule_flags)
            timeline.append({
                'frame': fidx,
                'reps': counter_snapshot['reps'],
                'state': counter_snapshot['state'],
                'metrics': {k: round(float(v),2) for k,v in metrics.items()},
                'rules': rule_flags,
                'form': form,
            })
            if writer is not None:
                txt = self._overlay_lines(counter_snapshot, form)
                frame = draw_pose(frame, kps)
                frame = draw_hud(frame, txt, org=(10, 30))
                writer.write(frame)
            fidx += 1
        cap.release()
        if writer is not None:
            writer.release()
        summary = {
            'exercise': self.exercise,
            'total_reps': int(self.counter.s.reps),
            'frames': len(timeline),
            'violations_rate': {rid: float(1 - sum(1 for t in timeline if t['rules'].get(rid, True)) / max(1,len(timeline))) for rid in {r['id'] for r in self.rules}}
        }
        result = {'summary': summary, 'timeline': timeline}
        if json_out:
            with open(json_out,'w') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    def run_live(self, src:int=0, json_out:Optional[str]=None, window_name:str='Workout Monitor', out_overlay_path:Optional[str]=None):
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open camera index {src}')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps < 15 or fps > 120:
            fps = 30.0
        cap.set(cv2.CAP_PROP_FPS, fps)
        self.smoother = PoseSmoother(freq=fps, min_cutoff=0.7, beta=0.02)
        writer = None
        if out_overlay_path and self.overlay:
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_overlay_path, fourcc, fps, (W,H))
        timeline = []
        countdown_phases = [('3', 1.0), ('2', 1.0), ('1', 1.0), ('Ready', 1.0), ('Go!', 0.5)]
        countdown_idx = 0
        # Track countdown in frames so the recorded clip keeps real-time pacing
        countdown_frame_counts = [max(1, int(round(duration * fps))) for _, duration in countdown_phases]
        countdown_frames_left = countdown_frame_counts[0] if countdown_frame_counts else 0
        countdown_done = len(countdown_phases) == 0
        frame_period = 1.0 / fps
        live_start_time = None

        def throttle(start_time):
            elapsed = time.time() - start_time
            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)

        def live_timestamp_ms():
            nonlocal live_start_time
            now = time.monotonic()
            if live_start_time is None:
                live_start_time = now
            return int(round((now - live_start_time) * 1000.0))

        while True:
            loop_start = time.time()
            ok, frame = cap.read()
            if not ok:
                break
            display = frame.copy()
            if not countdown_done:
                if self.overlay:
                    H, W = display.shape[:2]
                    text = countdown_phases[countdown_idx][0]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 2.5
                    thickness_bg = 8
                    thickness_fg = 4
                    ((tw, th), baseline) = cv2.getTextSize(text, font, scale, thickness_fg)
                    org = (int(W/2 - tw/2), int(H/2 + th/2))
                    cv2.putText(display, text, org, font, scale, (0,0,0), thickness_bg, cv2.LINE_AA)
                    cv2.putText(display, text, org, font, scale, (0,255,0), thickness_fg, cv2.LINE_AA)
                    cv2.imshow(window_name, display)
                    if writer is not None:
                        writer.write(display)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q')):
                        countdown_done = True
                        break
                if countdown_frame_counts:
                    countdown_frames_left -= 1
                    if countdown_frames_left <= 0:
                        countdown_idx += 1
                        if countdown_idx >= len(countdown_phases):
                            countdown_done = True
                        else:
                            countdown_frames_left = countdown_frame_counts[countdown_idx]
                else:
                    countdown_done = True
                # Skip processing the final countdown frame; next loop picks up realtime
                throttle(loop_start)
                continue
            pose = self.pose.infer(frame, timestamp_ms=live_timestamp_ms())
            if pose is None:
                if self.overlay:
                    draw_hud(display, [f'No person detected', f'Reps: {self.counter.s.reps}'])
                    cv2.imshow(window_name, display)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q')):
                        break
                throttle(loop_start)
                continue
            kps = self.smoother.smooth(pose.keypoints)
            angles = self._angles(kps)
            prev_reps = int(self.counter.s.reps)
            counter_snapshot = self.counter.update(kps, angles)
            self._announce_rep_if_needed(prev_reps, int(counter_snapshot['reps']))
            metrics = {
                'knee_angle_min': angles['knee_angle_min'],
                'elbow_angle_min': angles['elbow_angle_min'],
                'trunk_angle': angles['trunk_angle'],
            }
            rule_flags = evaluate_rules(
                self.rules,
                metrics,
                context={'state': counter_snapshot.get('state')},
            )
            form = self._feedback_assessment(counter_snapshot, rule_flags)
            timeline.append({
                'frame': len(timeline),
                'reps': counter_snapshot['reps'],
                'state': counter_snapshot['state'],
                'metrics': {k: round(float(v),2) for k,v in metrics.items()},
                'rules': rule_flags,
                'form': form,
            })
            if self.overlay:
                txt = self._overlay_lines(counter_snapshot, form)
                display = draw_pose(display, kps)
                display = draw_hud(display, txt, org=(10, 30))
                cv2.imshow(window_name, display)
                if writer is not None:
                    writer.write(display)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
            elif writer is not None:
                writer.write(display)
            throttle(loop_start)
        cap.release()
        if self.overlay:
            cv2.destroyWindow(window_name)
        if writer is not None:
            writer.release()
        summary = {
            'exercise': self.exercise,
            'total_reps': int(self.counter.s.reps),
            'frames': len(timeline),
            'violations_rate': {rid: float(1 - sum(1 for t in timeline if t['rules'].get(rid, True)) / max(1,len(timeline))) for rid in {r['id'] for r in self.rules}}
        }
        result = {'summary': summary, 'timeline': timeline}
        if json_out:
            with open(json_out,'w') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        return result
