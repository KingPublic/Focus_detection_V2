# =============================================================================
# tracking/student_tracker.py  v3
#
# FIX KRITIS:
#   update() sekarang mengembalikan List[StudentState] yang DIJAMIN sejajar
#   dengan input bboxes — result[i] selalu sesuai dengan bboxes[i].
#
#   Sebelumnya: tracker return list dengan urutan arbitrary → zip(faces,students)
#   salah pasang → behavior student A ke-assign ke student B.
#
#   Sekarang: update() return Dict[int, StudentState] dimapping ke bbox index,
#   sehingga main.py bisa akses student yang tepat untuk tiap face_data.
# =============================================================================

import numpy as np
import time
from typing import Dict, List, Tuple, Optional

import config.settings as cfg
from tracking.student_state import StudentState


def compute_iou(bbox_a: Tuple, bbox_b: Tuple) -> float:
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b
    ix1 = max(ax, bx);       iy1 = max(ay, by)
    ix2 = min(ax+aw, bx+bw); iy2 = min(ay+ah, by+bh)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0


class StudentTracker:
    """
    Persistent student tracker dengan IoU matching.

    update() mengembalikan list StudentState yang SEJAJAR dengan input bboxes:
        result[i] = StudentState untuk bboxes[i]

    Ini menjamin zip(faces, matched_students) selalu benar.
    """

    def __init__(self):
        self._students: Dict[int, StudentState] = {}  # id → state
        self._history:  Dict[int, StudentState] = {}
        self._next_id:  int = 1
        print("[StudentTracker] v3 initialized — aligned output guaranteed.")

    def update(self,
               bboxes: List[Tuple[int,int,int,int]]
               ) -> List[Optional[StudentState]]:
        """
        Cocokkan bbox baru dengan student yang dikenal.

        Returns:
            List[StudentState] dengan panjang = len(bboxes).
            result[i] adalah StudentState untuk bboxes[i].
            Tidak ada None — setiap bbox pasti dapat StudentState
            (lama atau baru).
        """
        # ── Timeout check ──────────────────────────────────────────────
        for sid in list(self._students.keys()):
            s = self._students[sid]
            if s.check_timeout():
                s.mark_absent()
                self._history[sid] = s
                del self._students[sid]

        # ── Hasil: satu StudentState per bbox input ─────────────────────
        result: List[Optional[StudentState]] = [None] * len(bboxes)

        if not bboxes:
            return result

        active_ids = list(self._students.keys())
        n_active   = len(active_ids)
        n_new      = len(bboxes)

        matched_student_ids: set = set()   # student id yang sudah di-match
        matched_bbox_idx:    set = set()   # indeks bbox yang sudah di-match

        if n_active > 0:
            # Bangun IoU matrix [n_active × n_new]
            iou_mat = np.zeros((n_active, n_new), dtype=np.float32)
            for i, sid in enumerate(active_ids):
                for j, bbox in enumerate(bboxes):
                    iou_mat[i, j] = compute_iou(self._students[sid].bbox, bbox)

            # Greedy matching dengan tie-break jarak center
            while True:
                if iou_mat.max() < cfg.IOU_MATCH_THRESHOLD:
                    break

                max_val    = iou_mat.max()
                candidates = np.argwhere(iou_mat >= max_val - 0.01)

                # Tie-break: pilih pasangan dengan jarak center terpendek
                best_i, best_j = candidates[0]
                best_dist = float('inf')
                for ci, cj in candidates:
                    sid_c = active_ids[ci]
                    cx_s, cy_s = self._students[sid_c].bbox_center
                    bx, by, bw, bh = bboxes[cj]
                    dist = (cx_s - (bx + bw//2))**2 + (cy_s - (by + bh//2))**2
                    if dist < best_dist:
                        best_dist = dist
                        best_i, best_j = ci, cj

                sid = active_ids[best_i]
                self._students[sid].mark_seen(bboxes[best_j])
                matched_student_ids.add(sid)
                matched_bbox_idx.add(best_j)

                # ── KUNCI FIX: catat student di posisi bbox yang tepat ──
                result[best_j] = self._students[sid]

                iou_mat[best_i, :] = 0
                iou_mat[:, best_j] = 0

        # ── Bbox yang tidak match → student baru ───────────────────────
        for j in range(n_new):
            if j not in matched_bbox_idx:
                if self._next_id <= cfg.MAX_STUDENTS:
                    ns = StudentState(self._next_id)
                    ns.mark_seen(bboxes[j])
                    self._students[self._next_id] = ns
                    result[j] = ns
                    print(f"[Tracker] New: {ns.label}")
                    self._next_id += 1

        # ── Safety: pastikan tidak ada None di result ───────────────────
        for j in range(n_new):
            if result[j] is None:
                # Fallback: assign student baru
                ns = StudentState(self._next_id)
                ns.mark_seen(bboxes[j])
                self._students[self._next_id] = ns
                result[j] = ns
                self._next_id += 1

        return result

    def get_all_students(self) -> List[StudentState]:
        return list(self._students.values())

    def get_history(self) -> List[StudentState]:
        return list(self._history.values())

    def get_all_ever(self) -> List[StudentState]:
        combined = {**self._history, **self._students}
        return sorted(combined.values(), key=lambda s: s.id_num)

    def reset(self):
        self._students.clear()
        self._history.clear()
        self._next_id = 1
        print("[StudentTracker] Reset.")