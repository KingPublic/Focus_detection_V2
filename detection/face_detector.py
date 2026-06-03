# =============================================================================
# detection/face_detector.py — Classroom Monitor v2
# Perbaikan:
#   1. NMS (Non-Maximum Suppression) — hapus bbox overlap berlebihan
#      sehingga 1 wajah tidak bisa jadi 2 deteksi
#   2. Minimum face size filter — hapus deteksi terlalu kecil (bukan wajah)
#   3. Confidence threshold dinaikkan (di settings.py)
# =============================================================================

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, List, Tuple
import config.settings as cfg

from mediapipe.python.solutions.face_mesh    import FaceMesh
from mediapipe.python.solutions.drawing_utils  import draw_landmarks
from mediapipe.python.solutions.drawing_styles import get_default_face_mesh_tesselation_style
from mediapipe.python.solutions import face_mesh as _mp_face_mesh_module


@dataclass
class FaceData:
    landmarks_px:   np.ndarray
    landmarks_norm: np.ndarray
    face_rect:      Tuple[int,int,int,int]
    iris_left_px:   Optional[np.ndarray]
    iris_right_px:  Optional[np.ndarray]


def _nms_bboxes(faces: List[FaceData],
                overlap_thresh: float) -> List[FaceData]:
    """
    Non-Maximum Suppression sederhana untuk FaceData list.

    Menghapus deteksi yang bbox-nya terlalu overlap (IoU > threshold)
    dengan deteksi lain yang sudah ada. Ini mencegah 1 wajah fisik
    dideteksi sebagai 2 student berbeda.

    Algoritma: greedy — simpan deteksi terbesar dulu, hapus yang overlap.
    """
    if len(faces) <= 1:
        return faces

    # Urutkan berdasarkan area bbox (terbesar dulu)
    faces_sorted = sorted(
        faces,
        key=lambda f: f.face_rect[2] * f.face_rect[3],
        reverse=True
    )

    kept = []
    suppressed = set()

    for i, fa in enumerate(faces_sorted):
        if i in suppressed:
            continue
        kept.append(fa)
        ax, ay, aw, ah = fa.face_rect

        for j in range(i + 1, len(faces_sorted)):
            if j in suppressed:
                continue
            bx, by, bw, bh = faces_sorted[j].face_rect

            # Hitung IoU
            ix1 = max(ax, bx);  iy1 = max(ay, by)
            ix2 = min(ax+aw, bx+bw); iy2 = min(ay+ah, by+bh)
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            if inter == 0:
                continue
            union = aw*ah + bw*bh - inter
            iou   = inter / union if union > 0 else 0.0

            if iou > overlap_thresh:
                suppressed.add(j)

    return kept


class FaceDetector:
    """
    MediaPipe Face Mesh wrapper dengan NMS dan size filter.
    """
    IRIS_LEFT_IDX  = [468, 469, 470, 471, 472]
    IRIS_RIGHT_IDX = [473, 474, 475, 476, 477]

    def __init__(self):
        self._face_mesh = FaceMesh(
            max_num_faces=cfg.MAX_NUM_FACES,
            refine_landmarks=cfg.REFINE_LANDMARKS,
            min_detection_confidence=cfg.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=cfg.MIN_TRACKING_CONFIDENCE,
        )
        print(f"[FaceDetector] conf={cfg.MIN_DETECTION_CONFIDENCE} "
              f"min_size={cfg.MIN_FACE_SIZE_PX}px NMS={cfg.NMS_OVERLAP_THRESHOLD}")

    def process(self, bgr_frame: np.ndarray) -> Tuple[bool, List[FaceData]]:
        h, w = bgr_frame.shape[:2]
        rgb  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._face_mesh.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            return False, []

        faces: List[FaceData] = []
        for face_landmarks in results.multi_face_landmarks:
            lm = face_landmarks.landmark

            landmarks_norm = np.array(
                [[p.x, p.y, p.z] for p in lm], dtype=np.float32)
            landmarks_px = (landmarks_norm[:, :2] * np.array([w, h])).astype(np.int32)

            xs, ys = landmarks_px[:, 0], landmarks_px[:, 1]
            x1 = max(0, xs.min() - 10)
            y1 = max(0, ys.min() - 10)
            x2 = min(w, xs.max() + 10)
            y2 = min(h, ys.max() + 10)
            bw, bh = x2 - x1, y2 - y1

            # ── Filter: ukuran minimum ─────────────────────────────────
            if bw < cfg.MIN_FACE_SIZE_PX or bh < cfg.MIN_FACE_SIZE_PX:
                continue   # Terlalu kecil — kemungkinan bukan wajah

            face_rect = (x1, y1, bw, bh)

            iris_left = iris_right = None
            if len(lm) > 476:
                iris_left  = np.array([[lm[i].x*w, lm[i].y*h]
                                        for i in self.IRIS_LEFT_IDX],  dtype=np.float32)
                iris_right = np.array([[lm[i].x*w, lm[i].y*h]
                                        for i in self.IRIS_RIGHT_IDX], dtype=np.float32)

            faces.append(FaceData(
                landmarks_px=landmarks_px,
                landmarks_norm=landmarks_norm,
                face_rect=face_rect,
                iris_left_px=iris_left,
                iris_right_px=iris_right,
            ))

        # ── NMS: hapus deteksi yang overlap berlebihan ─────────────────
        faces = _nms_bboxes(faces, cfg.NMS_OVERLAP_THRESHOLD)

        return len(faces) > 0, faces

    def draw_landmarks(self, frame, face_data, draw_tesselation=False):
        if draw_tesselation:
            res = self._face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for mp_lm in (res.multi_face_landmarks or []):
                draw_landmarks(frame, mp_lm,
                    _mp_face_mesh_module.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=get_default_face_mesh_tesselation_style())
        else:
            for pt in face_data.landmarks_px[::5]:
                cv2.circle(frame, tuple(pt), 1, (80, 200, 80), -1)
        return frame

    def release(self):
        self._face_mesh.close()
        print("[FaceDetector] Released.")