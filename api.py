from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional

import cv2
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .exercises.specs import SPECS

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "api_artifacts"
UI_DIR = BASE_DIR / "ui"
DB_PATH = ARTIFACTS_DIR / "server_db.json"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
UI_DIR.mkdir(parents=True, exist_ok=True)

Exercise = Literal["squat", "pushup", "bicep_curl", "pullup"]
logger = logging.getLogger("workout.api")


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


_load_dotenv(BASE_DIR / ".env")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "").strip()
MAX_WS_FRAME_BYTES = _env_int("MAX_WS_FRAME_BYTES", 2_000_000)
WS_LOG_EVERY_N_FRAMES = max(1, _env_int("WS_LOG_EVERY_N_FRAMES", 30))


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=200)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    username: str


class ExerciseConfigItem(BaseModel):
    exercise: Exercise
    thresholds: Dict[str, float | int]
    form_rules: List[Dict]


class ExerciseConfigResponse(BaseModel):
    version: str
    updated_at: str
    items: List[ExerciseConfigItem]


class SessionSummary(BaseModel):
    total_reps: int = 0
    duration_ms: int = 0
    avg_fps: float = 0.0


class SessionPayload(BaseModel):
    client_session_id: str = Field(min_length=1, max_length=120)
    exercise: Exercise
    started_at: str
    ended_at: str
    config_version: str
    summary: SessionSummary
    timeline: Optional[List[Dict]] = None
    kpis: Optional[Dict] = None


class SessionUpsertResponse(BaseModel):
    status: str
    cursor: int
    session_id: str


class SessionSyncItem(BaseModel):
    cursor: int
    updated_at: str
    payload: SessionPayload


class SessionSyncResponse(BaseModel):
    next_cursor: int
    items: List[SessionSyncItem]


class ServerStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._tokens: Dict[str, Dict[str, str]] = {}
        if not self.path.exists():
            self._write({"cursor": 0, "sessions": {}})

    def _read(self) -> Dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"cursor": 0, "sessions": {}}

    def _write(self, data: Dict) -> None:
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def login(self, username: str) -> LoginResponse:
        user_norm = username.strip().lower()
        user_id = hashlib.sha1(user_norm.encode("utf-8")).hexdigest()[:16]
        token = secrets.token_urlsafe(32)
        self._tokens[token] = {"user_id": user_id, "username": username}
        return LoginResponse(access_token=token, user_id=user_id, username=username)

    def authenticate(self, token: str) -> Dict[str, str]:
        info = self._tokens.get(token)
        if not info:
            raise HTTPException(status_code=401, detail="invalid or expired token")
        return info

    def upsert_session(self, user_id: str, payload: SessionPayload) -> SessionUpsertResponse:
        with self._lock:
            db = self._read()
            db.setdefault("sessions", {})
            db["cursor"] = int(db.get("cursor", 0)) + 1
            cursor = db["cursor"]
            key = f"{user_id}:{payload.client_session_id}"
            db["sessions"][key] = {
                "user_id": user_id,
                "cursor": cursor,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "payload": payload.model_dump(),
            }
            self._write(db)
        return SessionUpsertResponse(status="ok", cursor=cursor, session_id=payload.client_session_id)

    def sync_sessions(self, user_id: str, since: int) -> SessionSyncResponse:
        db = self._read()
        sessions = db.get("sessions", {})
        items: List[SessionSyncItem] = []
        max_cursor = int(since)
        for record in sessions.values():
            if record.get("user_id") != user_id:
                continue
            cur = int(record.get("cursor", 0))
            if cur <= since:
                continue
            payload = SessionPayload.model_validate(record["payload"])
            items.append(
                SessionSyncItem(
                    cursor=cur,
                    updated_at=record.get("updated_at", datetime.now(timezone.utc).isoformat()),
                    payload=payload,
                )
            )
            if cur > max_cursor:
                max_cursor = cur
        items.sort(key=lambda x: x.cursor)
        return SessionSyncResponse(next_cursor=max_cursor, items=items)


store = ServerStore(DB_PATH)
CONFIG_VERSION = hashlib.sha1(json.dumps(SPECS, sort_keys=True).encode("utf-8")).hexdigest()[:12]
CONFIG_UPDATED_AT = datetime.now(timezone.utc).isoformat()

app = FastAPI(title="Workout Monitor Mobile Backend", version="2.0.0")
CORS_ALLOWED_ORIGINS_RAW = os.getenv("CORS_ALLOWED_ORIGINS", "*").strip()
if CORS_ALLOWED_ORIGINS_RAW == "*":
    cors_allow_origins = ["*"]
    cors_allow_credentials = False
else:
    cors_allow_origins = [o.strip() for o in CORS_ALLOWED_ORIGINS_RAW.split(",") if o.strip()]
    cors_allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home() -> FileResponse:
    index_path = UI_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="ui/index.html not found")
    return FileResponse(index_path)


@app.get("/live_test.html")
def live_test() -> FileResponse:
    live_test_path = UI_DIR / "live_test.html"
    if not live_test_path.exists():
        raise HTTPException(status_code=404, detail="ui/live_test.html not found")
    return FileResponse(live_test_path)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "ts": int(time.time())}


@app.post("/auth/login", response_model=LoginResponse)
def login(body: LoginRequest) -> LoginResponse:
    # MVP auth: accept non-empty credentials and issue short-lived in-memory token.
    # This keeps backend scope minimal while app AI loop runs fully on-device.
    return store.login(body.username)


def _is_internal_api_key_valid(raw_key: str) -> bool:
    if not INTERNAL_API_KEY:
        return False
    return secrets.compare_digest(raw_key.strip(), INTERNAL_API_KEY)


def current_user(
    authorization: str = Header(default=""),
    x_internal_key: str = Header(default="", alias="X-Internal-Key"),
) -> Dict[str, str]:
    if _is_internal_api_key_valid(x_internal_key):
        return {"user_id": "internal_service", "username": "internal_service"}
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="missing bearer token")
    return store.authenticate(token)


def _authorize_ws(websocket: WebSocket) -> Dict[str, str]:
    # Browser WebSocket cannot set arbitrary headers, so allow query-param auth
    # for local testing (token/internal_key) in addition to header-based auth.
    if _is_internal_api_key_valid(websocket.headers.get("x-internal-key", "")):
        return {"user_id": "internal_service", "username": "internal_service"}
    if _is_internal_api_key_valid(websocket.query_params.get("internal_key", "")):
        return {"user_id": "internal_service", "username": "internal_service"}
    query_token = (websocket.query_params.get("token", "") or "").strip()
    if query_token:
        return store.authenticate(query_token)
    auth = websocket.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = auth.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="missing bearer token")
    return store.authenticate(token)


@app.websocket("/v1/live/ws")
async def live_ws(
    websocket: WebSocket,
    exercise: Exercise = Query("pushup"),
    overlay: bool = Query(True),
    landmarks: bool = Query(True),
) -> None:
    ws_start = time.perf_counter()
    try:
        user = _authorize_ws(websocket)
    except HTTPException:
        await websocket.close(code=4401)
        return

    await websocket.accept()
    logger.info(
        "live_ws connected exercise=%s user=%s overlay=%s landmarks=%s",
        exercise,
        user.get("user_id"),
        overlay,
        landmarks,
    )
    try:
        from .pipeline.analyze import Analyzer
        from .pipeline.overlay import draw_hud, draw_pose
        from .rules.engine import evaluate_rules
    except Exception as exc:
        await websocket.send_json({"error": f"live_ws_init_failed: {exc}"})
        await websocket.close(code=1011)
        return

    analyzer = Analyzer(exercise=exercise, overlay=overlay, voice=False)
    frame_idx = 0
    fps = 30.0

    def _elapsed_ms(start: float) -> int:
        return int((time.perf_counter() - start) * 1000)

    try:
        while True:
            frame_started_at = time.perf_counter()
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            payload = msg.get("bytes")
            if not payload:
                continue
            if len(payload) > MAX_WS_FRAME_BYTES:
                await websocket.send_json(
                    {
                        "error": "frame_too_large",
                        "max_ws_frame_bytes": MAX_WS_FRAME_BYTES,
                    }
                )
                continue

            decode_started_at = time.perf_counter()
            arr = np.frombuffer(payload, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            decode_ms = _elapsed_ms(decode_started_at)
            if frame is None:
                await websocket.send_json({"error": "invalid_frame"})
                continue

            display = frame.copy()
            pose_started_at = time.perf_counter()
            pose = analyzer.pose.infer(frame, timestamp_ms=int(round(frame_idx * 1000.0 / fps)))
            pose_inference_ms = _elapsed_ms(pose_started_at)
            frame_idx += 1

            if pose is None:
                overlay_encode_ms = 0
                if overlay:
                    draw_hud(display, ["No person detected", f"Reps: {analyzer.counter.s.reps}"])
                    overlay_started_at = time.perf_counter()
                    ok, encoded = cv2.imencode(".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
                    overlay_encode_ms = _elapsed_ms(overlay_started_at)
                    overlay_jpeg = (
                        f"data:image/jpeg;base64,{base64.b64encode(encoded.tobytes()).decode('ascii')}" if ok else None
                    )
                else:
                    overlay_jpeg = None
                processing_ms = _elapsed_ms(frame_started_at)
                await websocket.send_json(
                    {
                        "exercise": exercise,
                        "reps": int(analyzer.counter.s.reps),
                        "state": analyzer.counter.s.state,
                        "form_status": "observing",
                        "violations": [],
                        "metrics": {},
                        "perf": {
                            "decode_ms": decode_ms,
                            "pose_inference_ms": pose_inference_ms,
                            "overlay_encode_ms": overlay_encode_ms,
                            "server_processing_ms": processing_ms,
                        },
                        "overlay_jpeg": overlay_jpeg,
                    }
                )
                if frame_idx % WS_LOG_EVERY_N_FRAMES == 0:
                    logger.info(
                        "live_ws perf exercise=%s reps=%s state=%s decode_ms=%s pose_inference_ms=%s overlay_encode_ms=%s server_processing_ms=%s",
                        exercise,
                        analyzer.counter.s.reps,
                        analyzer.counter.s.state,
                        decode_ms,
                        pose_inference_ms,
                        overlay_encode_ms,
                        processing_ms,
                    )
                continue

            analysis_started_at = time.perf_counter()
            kps = analyzer.smoother.smooth(pose.keypoints)
            angles = analyzer._angles(kps)
            counter_snapshot = analyzer.counter.update(kps, angles)
            metrics = {
                "knee_angle_min": float(angles["knee_angle_min"]),
                "elbow_angle_min": float(angles["elbow_angle_min"]),
                "trunk_angle": float(angles["trunk_angle"]),
            }
            rule_flags = evaluate_rules(
                analyzer.rules,
                metrics,
                context={"state": counter_snapshot.get("state")},
            )
            form = analyzer._feedback_assessment(counter_snapshot, rule_flags)
            analysis_ms = _elapsed_ms(analysis_started_at)

            overlay_encode_ms = 0
            if overlay:
                text_lines = analyzer._overlay_lines(counter_snapshot, form)
                if landmarks:
                    display = draw_pose(display, kps)
                display = draw_hud(display, text_lines, org=(10, 30))
                overlay_started_at = time.perf_counter()
                ok, encoded = cv2.imencode(".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
                overlay_encode_ms = _elapsed_ms(overlay_started_at)
                overlay_jpeg = (
                    f"data:image/jpeg;base64,{base64.b64encode(encoded.tobytes()).decode('ascii')}" if ok else None
                )
            else:
                overlay_jpeg = None

            processing_ms = _elapsed_ms(frame_started_at)
            await websocket.send_json(
                {
                    "exercise": exercise,
                    "reps": int(counter_snapshot["reps"]),
                    "state": counter_snapshot["state"],
                    "form_status": form["form_status"],
                    "violations": form["violations"],
                    "metrics": {k: round(float(v), 2) for k, v in metrics.items()},
                    "perf": {
                        "decode_ms": decode_ms,
                        "pose_inference_ms": pose_inference_ms,
                        "analysis_ms": analysis_ms,
                        "overlay_encode_ms": overlay_encode_ms,
                        "server_processing_ms": processing_ms,
                    },
                    "overlay_jpeg": overlay_jpeg,
                }
            )
            if frame_idx % WS_LOG_EVERY_N_FRAMES == 0:
                logger.info(
                    "live_ws perf exercise=%s reps=%s state=%s decode_ms=%s pose_inference_ms=%s analysis_ms=%s overlay_encode_ms=%s server_processing_ms=%s",
                    exercise,
                    counter_snapshot["reps"],
                    counter_snapshot["state"],
                    decode_ms,
                    pose_inference_ms,
                    analysis_ms,
                    overlay_encode_ms,
                    processing_ms,
                )
    except WebSocketDisconnect:
        logger.info("live_ws disconnected exercise=%s user=%s", exercise, user.get("user_id"))
    finally:
        session_ms = int((time.perf_counter() - ws_start) * 1000)
        logger.info("live_ws closed exercise=%s user=%s duration_ms=%s", exercise, user.get("user_id"), session_ms)
        try:
            analyzer.pose.close()
        except Exception:
            pass


@app.get("/v1/exercise-config", response_model=ExerciseConfigResponse)
def get_exercise_config(_: Dict[str, str] = Depends(current_user)) -> ExerciseConfigResponse:
    items = [
        ExerciseConfigItem(
            exercise=ex_name,
            thresholds=spec.get("thresholds", {}),
            form_rules=spec.get("form_rules", []),
        )
        for ex_name, spec in SPECS.items()
    ]
    return ExerciseConfigResponse(version=CONFIG_VERSION, updated_at=CONFIG_UPDATED_AT, items=items)


@app.post("/v1/workout-sessions", response_model=SessionUpsertResponse)
def upsert_workout_session(
    body: SessionPayload,
    user: Dict[str, str] = Depends(current_user),
) -> SessionUpsertResponse:
    return store.upsert_session(user_id=user["user_id"], payload=body)


@app.get("/v1/workout-sessions/sync", response_model=SessionSyncResponse)
def sync_workout_sessions(
    since: int = 0,
    user: Dict[str, str] = Depends(current_user),
) -> SessionSyncResponse:
    if since < 0:
        raise HTTPException(status_code=400, detail="since must be >= 0")
    return store.sync_sessions(user_id=user["user_id"], since=since)
