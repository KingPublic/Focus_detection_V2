# =============================================================================
# tracking/student_state.py
# Per-Student State Container
#
# Setiap mahasiswa yang terdeteksi memiliki instance StudentState sendiri
# yang menyimpan:
#   - ID unik (Student_01, Student_02, dst)
#   - Bounding box posisi wajah di frame
#   - Instance TemporalTracker, ScoreCalculator, SessionLogger sendiri
#   - History behavior dan durasi
#   - Status kehadiran (aktif / absent)
# =============================================================================

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Set, Dict, Optional, Tuple

import config.settings as cfg


# ── Inline minimal versions (hindari circular import) ──────────────────────

class _TemporalTracker:
    """Versi ringkas TemporalTracker untuk per-student."""
    GRACE = 0.6

    def __init__(self):
        self._timers: Dict[str, Dict] = {}

    def update(self, active: Set[str]) -> Dict[str, float]:
        now = time.monotonic()
        durations = {}
        for beh in active:
            if beh not in self._timers:
                self._timers[beh] = {"start": now, "active": True, "last": now}
            t = self._timers[beh]
            if not t["active"]:
                t["start"] = now
                t["active"] = True
            t["last"] = now
            durations[beh] = now - t["start"]
        for beh, t in self._timers.items():
            if beh not in active and t["active"]:
                if now - t["last"] > self.GRACE:
                    t["active"] = False
                    t["start"]  = 0.0
                else:
                    durations[beh] = now - t["start"]
        return durations

    def reset(self):
        self._timers.clear()


class _ScoreCalc:
    """Versi ringkas ScoreCalculator untuk per-student."""
    def __init__(self):
        self.focus      = 100.0
        self.suspicious = 0.0

    def update(self, n_behaviors: int, severity: str):
        if n_behaviors == 0:
            self.focus      = min(100.0, self.focus + 0.5)
            self.suspicious = max(0.0,   self.suspicious - 0.2)
        else:
            m = 2.0 if severity == "CRITICAL" else 1.0
            self.focus      = max(0.0,   self.focus - 1.5 * n_behaviors * m)
            self.suspicious = min(100.0, self.suspicious + 4.0 * n_behaviors * m)


class _SessionLog:
    """Overall Session Score — hanya turun, tidak pernah naik."""
    GRADE = [(90,"A"),(75,"B"),(60,"C"),(45,"D"),(0,"F")]

    def __init__(self):
        self._start     = time.monotonic()
        self._score     = 100.0
        self._f_ok      = 0
        self._f_warn    = 0
        self._f_crit    = 0
        self._ev_warn   = 0
        self._ev_crit   = 0
        self._prev_sev  = "OK"

    def update(self, severity: str):
        if severity == "OK":
            self._f_ok += 1
        elif severity == "WARNING":
            self._f_warn += 1
            self._score = max(0.0, self._score - cfg.PENALTY_WARNING)
        else:
            self._f_crit += 1
            self._score = max(0.0, self._score - cfg.PENALTY_CRITICAL)
        if severity == "WARNING"  and self._prev_sev != "WARNING":  self._ev_warn += 1
        if severity == "CRITICAL" and self._prev_sev != "CRITICAL": self._ev_crit += 1
        self._prev_sev = severity

    @property
    def score(self) -> float:
        return self._score

    @property
    def grade(self) -> str:
        for thr, g in self.GRADE:
            if self._score >= thr:
                return g
        return "F"

    @property
    def duration_str(self) -> str:
        e = time.monotonic() - self._start
        return f"{int(e//3600):02d}:{int((e%3600)//60):02d}:{int(e%60):02d}"

    @property
    def pct_ok(self)   -> float:
        n = max(self._f_ok + self._f_warn + self._f_crit, 1)
        return self._f_ok / n * 100

    @property
    def pct_warn(self) -> float:
        n = max(self._f_ok + self._f_warn + self._f_crit, 1)
        return self._f_warn / n * 100

    @property
    def pct_crit(self) -> float:
        n = max(self._f_ok + self._f_warn + self._f_crit, 1)
        return self._f_crit / n * 100

    @property
    def warn_events(self) -> int: return self._ev_warn

    @property
    def crit_events(self) -> int: return self._ev_crit


# ── StudentState ────────────────────────────────────────────────────────────

class StudentState:
    """
    Menyimpan semua state untuk satu mahasiswa yang diidentifikasi.

    Setiap mahasiswa punya:
        - ID string: "Student_01", "Student_02", dst
        - Bounding box wajah terakhir
        - Instance tracker sendiri (temporal, score, session)
        - Behavior aktif dan durasinya
        - Status: ACTIVE | ABSENT
    """

    def __init__(self, student_id: int):
        self.id_num:   int   = student_id
        self.label:    str   = f"Student_{student_id:02d}"
        self.short_lbl: str  = f"S{student_id:02d}"

        # Bounding box: (x, y, w, h) piksel
        self.bbox: Tuple[int,int,int,int] = (0, 0, 0, 0)

        # Status kehadiran
        self.is_active:   bool  = True
        self.last_seen:   float = time.monotonic()

        # Trackers per-student
        self.temporal   = _TemporalTracker()
        self.score_calc = _ScoreCalc()
        self.session    = _SessionLog()

        # State terkini (diupdate setiap frame)
        self.active_behaviors: Set[str]       = set()
        self.durations:        Dict[str,float]= {}
        self.severity:         str            = "OK"
        self.head_pitch:       float          = 0.0
        self.head_yaw:         float          = 0.0
        self.ear_avg:          float          = 0.30
        self.eye_closed:       bool           = False

    def mark_seen(self, bbox: Tuple[int,int,int,int]):
        """Update kehadiran dan bbox saat wajah terdeteksi."""
        self.bbox        = bbox
        self.is_active   = True
        self.last_seen   = time.monotonic()

    def mark_absent(self):
        """Tandai student sebagai tidak terdeteksi."""
        self.is_active = False
        self.active_behaviors = {"FACE_ABSENT"}

    def check_timeout(self) -> bool:
        """Kembalikan True jika student sudah timeout (perlu mark_absent)."""
        return (time.monotonic() - self.last_seen) > cfg.ABSENT_TIMEOUT_SEC

    def update_behaviors(self, behaviors: Set[str], durations: Dict[str,float],
                         severity: str, pitch: float, yaw: float,
                         ear: float, eye_closed: bool):
        """Update semua state behavior untuk frame ini."""
        self.active_behaviors = behaviors
        self.durations        = durations
        self.severity         = severity
        self.head_pitch       = pitch
        self.head_yaw         = yaw
        self.ear_avg          = ear
        self.eye_closed       = eye_closed

        self.score_calc.update(len(behaviors), severity)
        self.session.update(severity)

    # ── Convenience properties ──────────────────────────────────────────

    @property
    def overall_score(self) -> float:
        return self.session.score

    @property
    def grade(self) -> str:
        return self.session.grade

    @property
    def focus_score(self) -> float:
        return self.score_calc.focus

    @property
    def suspicious_score(self) -> float:
        return self.score_calc.suspicious

    @property
    def bbox_center(self) -> Tuple[int,int]:
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)