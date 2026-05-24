# =============================================================================
# tracking/student_tracker.py
# Persistent Student Tracker menggunakan IoU Matching
#
# Algoritma:
#   Setiap frame menghasilkan N bounding box wajah dari MediaPipe.
#   Tracker mencocokkan setiap bbox baru dengan student yang sudah dikenali
#   menggunakan IoU (Intersection over Union).
#
#   IoU = Area(A ∩ B) / Area(A ∪ B)
#
#   Jika IoU >= threshold → bbox baru = update student lama (ID tetap)
#   Jika IoU <  threshold → tidak ada match → student baru → ID baru
#
#   Prinsip Hungarian Assignment:
#     Gunakan greedy matching (cukup untuk kelas dengan wajah tidak saling
#     tumpang tindih). Untuk akurasi lebih tinggi bisa upgrade ke
#     scipy.optimize.linear_sum_assignment (Hungarian algorithm).
#
# Referensi:
#   Bewley et al., "Simple Online and Realtime Tracking (SORT)," ICIP, 2016.
#   (Versi ini disederhanakan: tanpa Kalman filter, hanya IoU matching)
# =============================================================================

import numpy as np
import time
from typing import Dict, List, Tuple, Optional

import config.settings as cfg
from tracking.student_state import StudentState


def _bbox_to_xyxy(bbox: Tuple[int,int,int,int]) -> Tuple[int,int,int,int]:
    """Konversi (x, y, w, h) → (x1, y1, x2, y2)."""
    x, y, w, h = bbox
    return x, y, x + w, y + h


def compute_iou(bbox_a: Tuple, bbox_b: Tuple) -> float:
    """
    Hitung IoU antara dua bounding box format (x,y,w,h).

    IoU = Intersection Area / Union Area

    Returns:
        float dalam [0.0, 1.0]
    """
    ax1, ay1, ax2, ay2 = _bbox_to_xyxy(bbox_a)
    bx1, by1, bx2, by2 = _bbox_to_xyxy(bbox_b)

    # Intersection
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter   = inter_w * inter_h

    if inter == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


class StudentTracker:
    """
    Mengelola identitas semua mahasiswa yang terdeteksi dalam satu sesi.

    Fungsi utama:
        update(bboxes) → List[StudentState]
            Menerima list bounding box frame ini, mencocokkan dengan
            student yang dikenal, membuat student baru jika perlu,
            dan mengembalikan list StudentState aktif.

    ID Assignment:
        - ID diberikan secara berurutan (1, 2, 3, ...)
        - ID tidak pernah di-recycle dalam satu sesi
        - Student yang timeout dipertahankan di _history untuk laporan akhir
    """

    def __init__(self):
        self._active:  Dict[int, StudentState] = {}  # id → StudentState (di frame)
        self._history: Dict[int, StudentState] = {}  # id → StudentState (sudah absent)
        self._next_id: int = 1
        print("[StudentTracker] Initialized.")

    # ------------------------------------------------------------------ #

    def update(self, bboxes: List[Tuple[int,int,int,int]]) -> List[StudentState]:
        """
        Update tracker dengan bbox-bbox baru dari frame ini.

        Langkah:
            1. Hitung IoU matrix: active_students × new_bboxes
            2. Greedy match: tiap bbox baru → student dengan IoU tertinggi
            3. Unmatched bbox → student baru
            4. Unmatched student → timeout check → mark absent

        Args:
            bboxes: List (x, y, w, h) dari MediaPipe frame ini

        Returns:
            List StudentState yang aktif saat ini (termasuk yang baru)
        """
        # ── Cek timeout student yang tidak terlihat ────────────────────
        for sid, student in list(self._active.items()):
            if student.check_timeout():
                student.mark_absent()
                self._history[sid] = student
                del self._active[sid]

        if not bboxes:
            return list(self._active.values())

        # ── Bangun IoU matrix ──────────────────────────────────────────
        active_ids    = list(self._active.keys())
        n_active      = len(active_ids)
        n_new         = len(bboxes)

        matched_students: set = set()   # id student yang sudah dicocokkan
        matched_bboxes:   set = set()   # indeks bbox yang sudah dicocokkan

        if n_active > 0:
            # iou_matrix[i][j] = IoU antara student i dan bbox baru j
            iou_matrix = np.zeros((n_active, n_new), dtype=np.float32)
            for i, sid in enumerate(active_ids):
                for j, bbox in enumerate(bboxes):
                    iou_matrix[i][j] = compute_iou(
                        self._active[sid].bbox, bbox
                    )

            # Greedy matching: ambil pasangan IoU tertinggi berulang
            while True:
                if iou_matrix.max() < cfg.IOU_MATCH_THRESHOLD:
                    break
                i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                sid   = active_ids[i]

                # Match: update student lama dengan bbox baru
                self._active[sid].mark_seen(bboxes[j])
                matched_students.add(sid)
                matched_bboxes.add(j)

                # Hapus dari matrix agar tidak dicocokkan lagi
                iou_matrix[i, :] = 0
                iou_matrix[:, j] = 0

        # ── Bbox baru yang tidak cocok → student baru ─────────────────
        for j, bbox in enumerate(bboxes):
            if j not in matched_bboxes:
                if self._next_id <= cfg.MAX_STUDENTS:
                    new_student = StudentState(self._next_id)
                    new_student.mark_seen(bbox)
                    self._active[self._next_id] = new_student
                    print(f"[Tracker] New student: {new_student.label}")
                    self._next_id += 1

        return list(self._active.values())

    def get_all_students(self) -> List[StudentState]:
        """Kembalikan semua student aktif saat ini."""
        return list(self._active.values())

    def get_history(self) -> List[StudentState]:
        """Kembalikan student yang sudah tidak aktif (untuk laporan)."""
        return list(self._history.values())

    def get_all_ever(self) -> List[StudentState]:
        """Semua student yang pernah terdeteksi (aktif + history)."""
        combined = {**self._history, **self._active}
        return sorted(combined.values(), key=lambda s: s.id_num)

    def reset(self):
        self._active.clear()
        self._history.clear()
        self._next_id = 1
        print("[StudentTracker] Reset.")