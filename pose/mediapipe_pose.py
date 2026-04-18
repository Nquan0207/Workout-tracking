from dataclasses import dataclass
from pathlib import Path
import os
from typing import Dict, Tuple, Optional
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from ..utils.naming import MEDIAPIPE_LANDMARKS

MODEL_ENV_VAR = 'MEDIAPIPE_POSE_MODEL_PATH'
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / 'assets' / 'pose_landmarker_full.task'
DEFAULT_FRAME_MS = 33

@dataclass
class PoseResult:
    keypoints: Dict[str, Tuple[float, float, float, float]]  # x,y,z,visibility (pixel coords)
    score: float

class MediaPipePoseEstimator:
    def __init__(self, static_image_mode=False, model_complexity=1, enable_segmentation=False,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self._timestamp_ms = 0
        self._model_path = self._resolve_model_path()
        # The Tasks API selects accuracy through the bundled model asset.
        _ = model_complexity
        running_mode = VisionTaskRunningMode.IMAGE if static_image_mode else VisionTaskRunningMode.VIDEO
        self.model = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self._model_path)),
                running_mode=running_mode,
                num_poses=1,
                min_pose_detection_confidence=min_detection_confidence,
                min_pose_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                output_segmentation_masks=enable_segmentation,
            )
        )

    def infer(self, frame_bgr, timestamp_ms: Optional[int] = None) -> Optional[PoseResult]:
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self.static_image_mode:
            out = self.model.detect(image)
        else:
            if timestamp_ms is None:
                timestamp_ms = self._timestamp_ms
                self._timestamp_ms += DEFAULT_FRAME_MS
            else:
                timestamp_ms = int(timestamp_ms)
                if timestamp_ms <= self._timestamp_ms:
                    timestamp_ms = self._timestamp_ms + 1
                self._timestamp_ms = timestamp_ms
            out = self.model.detect_for_video(image, timestamp_ms)
        if not out.pose_landmarks:
            return None
        lm = out.pose_landmarks[0]
        keypoints = {}
        vis_scores = []
        for i, name in enumerate(MEDIAPIPE_LANDMARKS):
            l = lm[i]
            x, y = float(l.x * w), float(l.y * h)
            z = float(l.z)
            visibility = getattr(l, 'visibility', None)
            presence = getattr(l, 'presence', None)
            v = float(visibility if visibility is not None else presence if presence is not None else 1.0)
            keypoints[name] = (x, y, z, v)
            vis_scores.append(v)
        score = float(np.mean(vis_scores)) if vis_scores else 0.0
        return PoseResult(keypoints=keypoints, score=score)

    def close(self):
        if hasattr(self.model, 'close'):
            self.model.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _resolve_model_path(self) -> Path:
        model_path = os.environ.get(MODEL_ENV_VAR)
        if model_path:
            path = Path(model_path).expanduser().resolve()
        else:
            path = DEFAULT_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f'Pose model not found at {path}. Set {MODEL_ENV_VAR} to a valid MediaPipe pose '
                'landmarker .task file or add the bundled model to ai/assets/pose_landmarker_full.task.'
            )
        return path
