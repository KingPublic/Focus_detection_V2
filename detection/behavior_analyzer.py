# =============================================================================
# detection/behavior_analyzer.py — Classroom Monitor
# Rule-Based Behavior Analyzer (sama dengan System 1, tanpa SECOND_PERSON)
# =============================================================================

from typing import Dict, Set
import config.settings as cfg


BEHAVIOR_RULES = [
    {
        "id":    "LOOKING_LEFT",
        "check_fn": lambda hp_yaw, hp_pitch, hp_ok, ear, eye_closed, gaze_x: (
            hp_ok and hp_yaw > cfg.YAW_THRESHOLD_RIGHT
        ),
    },
    {
        "id":    "LOOKING_RIGHT",
        "check_fn": lambda hp_yaw, hp_pitch, hp_ok, ear, eye_closed, gaze_x: (
            hp_ok and hp_yaw < cfg.YAW_THRESHOLD_LEFT
        ),
    },
    {
        "id":    "LOOKING_DOWN",
        "check_fn": lambda hp_yaw, hp_pitch, hp_ok, ear, eye_closed, gaze_x: (
            hp_ok and hp_pitch > cfg.PITCH_DOWN_THRESHOLD
        ),
    },
    {
        "id":    "LOOKING_UP",
        "check_fn": lambda hp_yaw, hp_pitch, hp_ok, ear, eye_closed, gaze_x: (
            hp_ok and hp_pitch < cfg.PITCH_UP_THRESHOLD
        ),
    },
    {
        "id":    "EYES_CLOSED",
        "check_fn": lambda hp_yaw, hp_pitch, hp_ok, ear, eye_closed, gaze_x: (
            eye_closed
        ),
    },
    {
        "id":    "EYE_LOOK_LEFT",
        "check_fn": lambda hp_yaw, hp_pitch, hp_ok, ear, eye_closed, gaze_x: (
            gaze_x < cfg.GAZE_LEFT_THRESHOLD
            and (not hp_ok or abs(hp_yaw) < 15.0)
            and not eye_closed
        ),
    },
    {
        "id":    "EYE_LOOK_RIGHT",
        "check_fn": lambda hp_yaw, hp_pitch, hp_ok, ear, eye_closed, gaze_x: (
            gaze_x > cfg.GAZE_RIGHT_THRESHOLD
            and (not hp_ok or abs(hp_yaw) < 15.0)
            and not eye_closed
        ),
    },
]

WARN_THRESH = {
    "LOOKING_LEFT":   cfg.DURATION_WARN_LOOK_AWAY,
    "LOOKING_RIGHT":  cfg.DURATION_WARN_LOOK_AWAY,
    "LOOKING_DOWN":   cfg.DURATION_WARN_LOOK_DOWN,
    "LOOKING_UP":     cfg.DURATION_WARN_LOOK_AWAY,
    "FACE_ABSENT":    cfg.DURATION_WARN_FACE_ABSENT,
    "EYES_CLOSED":    cfg.DURATION_WARN_EYES_CLOSED,
    "EYE_LOOK_LEFT":  cfg.DURATION_WARN_GAZE,
    "EYE_LOOK_RIGHT": cfg.DURATION_WARN_GAZE,
}
CRIT_THRESH = {
    "LOOKING_LEFT":   cfg.DURATION_CRIT_LOOK_AWAY,
    "LOOKING_RIGHT":  cfg.DURATION_CRIT_LOOK_AWAY,
    "LOOKING_DOWN":   cfg.DURATION_CRIT_LOOK_DOWN,
    "LOOKING_UP":     cfg.DURATION_CRIT_LOOK_AWAY,
    "FACE_ABSENT":    cfg.DURATION_CRIT_FACE_ABSENT,
    "EYES_CLOSED":    cfg.DURATION_CRIT_EYES_CLOSED,
    "EYE_LOOK_LEFT":  cfg.DURATION_CRIT_GAZE,
    "EYE_LOOK_RIGHT": cfg.DURATION_CRIT_GAZE,
}


def evaluate_behaviors(hp_yaw: float, hp_pitch: float, hp_ok: bool,
                        ear: float, eye_closed: bool,
                        gaze_x: float) -> Set[str]:
    """Evaluasi semua rule dan kembalikan set behavior aktif."""
    active = set()
    for rule in BEHAVIOR_RULES:
        try:
            if rule["check_fn"](hp_yaw, hp_pitch, hp_ok, ear, eye_closed, gaze_x):
                active.add(rule["id"])
        except Exception:
            pass
    return active


def compute_severity(active_behaviors: Set[str],
                     durations: Dict[str, float]) -> str:
    """Kembalikan 'OK' | 'WARNING' | 'CRITICAL'."""
    severity = "OK"
    for beh in active_behaviors:
        dur    = durations.get(beh, 0.0)
        crit_t = CRIT_THRESH.get(beh, 999)
        warn_t = WARN_THRESH.get(beh, 999)
        if dur >= crit_t:
            return "CRITICAL"
        elif dur >= warn_t:
            severity = "WARNING"
    return severity