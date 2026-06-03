# =============================================================================
# tracking/student_tracker.py — Zone-Based 2-Student Tracker
#
# Pendekatan: ZONE-BASED (bukan IoU re-identification)
#
# Frame dibagi dua zona vertikal:
#   ┌─────────┬─────────┐
#   │  ZONE 1 │  ZONE 2 │
#   │  S01    │  S02    │
#   │  (kiri) │(kanan)  │
#   └─────────┴─────────┘
#
# Rules:
#   - Wajah dengan bbox center x < split_x  → selalu S01
#   - Wajah dengan bbox center x >= split_x → selalu S02
#   - Jika tidak ada wajah di zona → student ABSENT
#   - Jika kembali ke zona → student yang SAMA (persistent by design)
#   - Tidak perlu re-identification — zona = identitas
#
# Keunggulan vs IoU tracking:
#   - Tidak crash saat student pergi/kembali
#   - ID 100% persistent selama tidak pindah zona
#   - Sangat stabil dan mudah dijelaskan di paper
#   - Tidak terpengaruh pergerakan kepala
# =============================================================================

import time
from typing import List, Tuple, Optional
import config.settings as cfg
from tracking.student_state import StudentState


class StudentTracker:
    """
    Zone-Based 2-Student Tracker.

    Hanya dua slot: S01 (kiri) dan S02 (kanan).
    Slot ditentukan oleh posisi horizontal bbox center, bukan IoU matching.
    """

    def __init__(self, frame_width: int = 1280):
        self._fw       = frame_width
        self._split_x  = int(frame_width * cfg.SPLIT_RATIO)

        # Dua slot tetap — tidak pernah dihapus
        self._s01 = StudentState(1)   # Zona kiri
        self._s02 = StudentState(2)   # Zona kanan

        self._s01.is_active = False
        self._s02.is_active = False

        print(f"[ZoneTracker] split_x={self._split_x}px  "
              f"(S01: 0-{self._split_x}  |  S02: {self._split_x}-{frame_width})")

    # ------------------------------------------------------------------ #

    def update(self,
               bboxes: List[Tuple[int,int,int,int]]
               ) -> List[Optional[StudentState]]:
        """
        Cocokkan setiap bbox ke zona (kiri/kanan).

        Returns:
            List[StudentState] sejajar dengan input bboxes.
            result[i] = StudentState (S01 atau S02) untuk bboxes[i].

        Logika assignment:
            1. Hitung bbox center x untuk setiap wajah
            2. center_x < split_x  → S01
            3. center_x >= split_x → S02
            4. Jika dua wajah masuk zona yang sama → ambil yang lebih besar
               (heuristik: wajah lebih besar = lebih dekat ke kamera = lebih valid)
        """
        now = time.monotonic()

        # Reset kehadiran frame ini
        s01_candidate: Optional[Tuple[int, Tuple]] = None  # (area, bbox)
        s02_candidate: Optional[Tuple[int, Tuple]] = None

        result: List[Optional[StudentState]] = [None] * len(bboxes)

        # ── Tentukan zona tiap bbox ─────────────────────────────────────
        for i, bbox in enumerate(bboxes):
            x, y, w, h = bbox
            center_x   = x + w // 2
            area       = w * h

            if center_x < self._split_x:
                # Zona kiri → kandidat S01
                if s01_candidate is None or area > s01_candidate[0]:
                    s01_candidate = (area, bbox, i)
            else:
                # Zona kanan → kandidat S02
                if s02_candidate is None or area > s02_candidate[0]:
                    s02_candidate = (area, bbox, i)

        # ── Update S01 ─────────────────────────────────────────────────
        if s01_candidate is not None:
            _, bbox, idx  = s01_candidate
            self._s01.mark_seen(bbox)
            result[idx]   = self._s01
        else:
            # Tidak ada wajah di zona kiri
            if (now - self._s01.last_seen) > cfg.ABSENT_TIMEOUT_SEC:
                self._s01.is_active = False

        # ── Update S02 ─────────────────────────────────────────────────
        if s02_candidate is not None:
            _, bbox, idx  = s02_candidate
            self._s02.mark_seen(bbox)
            result[idx]   = self._s02
        else:
            if (now - self._s02.last_seen) > cfg.ABSENT_TIMEOUT_SEC:
                self._s02.is_active = False

        return result

    # ------------------------------------------------------------------ #

    def get_all_students(self) -> List[StudentState]:
        """Kembalikan kedua student (aktif maupun absent)."""
        return [self._s01, self._s02]

    def get_split_x(self) -> int:
        return self._split_x

    def get_all_ever(self) -> List[StudentState]:
        return [self._s01, self._s02]

    def reset(self):
        """Reset skor dan timer kedua student, ID tetap."""
        from tracking.student_state import _TemporalTracker, _ScoreCalc, _SessionLog
        for s in [self._s01, self._s02]:
            s.temporal   = _TemporalTracker()
            s.score_calc = _ScoreCalc()
            s.session    = _SessionLog()
            s.active_behaviors = set()
            s.durations        = {}
            s.severity         = "OK"
            s.is_active        = False
        print("[ZoneTracker] Reset — scores cleared, IDs preserved.")