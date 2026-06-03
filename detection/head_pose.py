# =============================================================================
# detection/head_pose.py — Classroom Monitor v2
#
# PERBAIKAN UTAMA: Adaptive Camera Matrix per wajah
#
# Masalah sebelumnya:
#   Camera matrix tunggal menggunakan principal point di tengah frame.
#   Wajah di tepi frame (mahasiswa di pojok kelas) mengalami distorsi
#   perspektif yang membuat sudut yaw/pitch salah secara sistematis.
#
# Solusi — Adaptive Camera Matrix:
#   Untuk setiap wajah, hitung camera matrix lokal:
#
#   1. Principal point (cx, cy) = pusat bounding box wajah
#      (bukan pusat frame)
#
#   2. Focal length adaptif berdasarkan jarak inter-okular (IOD):
#      Jarak antar sudut mata kiri-kanan rata-rata = 65mm di dunia nyata.
#      Jika pixel IOD = d, maka f_estimated = d × (frame_width / avg_iod_px)
#      Ini mengkompensasi perbedaan jarak kamera ke mahasiswa.
#
#   Hasil: head pose akurat baik untuk mahasiswa dekat maupun jauh,
#          dan untuk wajah di tepi frame.
#
# Referensi:
#   Zhang et al., "Head Pose Estimation in Videos," IEEE T-PAMI, 2020.
# =============================================================================

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from detection.face_detector import FaceData


@dataclass
class HeadPoseResult:
    pitch:   float
    yaw:     float
    roll:    float
    rvec:    np.ndarray
    tvec:    np.ndarray
    success: bool


class HeadPoseEstimator:
    """
    Head pose estimator dengan adaptive camera matrix per wajah.

    Untuk setiap wajah yang dianalisis, principal point dan focal length
    disesuaikan berdasarkan posisi dan ukuran wajah di frame.
    """

    # 6 titik model 3D wajah generik (mm)
    FACE_3D = np.array([
        [  0.0,    0.0,    0.0],   # Nose tip     #4
        [  0.0,  -63.6,  -12.5],   # Chin         #152
        [-43.3,   32.7,  -26.0],   # Left eye     #33
        [ 43.3,   32.7,  -26.0],   # Right eye    #263
        [-28.9,  -28.9,  -24.1],   # Left mouth   #61
        [ 28.9,  -28.9,  -24.1],   # Right mouth  #291
    ], dtype=np.float64)

    LM_IDX = [4, 152, 33, 263, 61, 291]

    # Jarak inter-okular rata-rata di dunia nyata (mm)
    REAL_IOD_MM = 65.0

    # Rata-rata IOD dalam pixel untuk webcam 1280px dari jarak 0.6m
    # Digunakan sebagai referensi kalibrasi focal length
    REF_IOD_PX  = 120.0
    REF_FOCAL   = 1280.0

    def __init__(self, frame_width: int, frame_height: int):
        self._fw   = frame_width
        self._fh   = frame_height
        self._dist = np.zeros((4, 1), dtype=np.float64)
        print(f"[HeadPose] Adaptive matrix mode — {frame_width}x{frame_height}")

    def _build_adaptive_matrix(self, face_data: FaceData) -> np.ndarray:
        """
        Hitung camera matrix adaptif untuk satu wajah.

        Principal point = pusat bounding box wajah.
        Focal length    = estimasi berdasarkan pixel IOD wajah.

        Mengapa ini lebih akurat:
          - Wajah di pojok frame: principal point frame center sangat meleset
          - Wajah jauh (kecil): IOD kecil → focal length perlu disesuaikan
        """
        x, y, w, h = face_data.face_rect

        # Principal point = pusat wajah (bukan pusat frame)
        cx = float(x + w / 2)
        cy = float(y + h / 2)

        # Estimasi focal length dari IOD (jarak pixel antar sudut mata)
        lm = face_data.landmarks_px
        left_corner  = lm[33].astype(float)   # sudut mata kiri
        right_corner = lm[263].astype(float)  # sudut mata kanan
        iod_px = float(np.linalg.norm(right_corner - left_corner))

        if iod_px > 10:
            # f proporsional dengan IOD pixel
            # Semakin jauh mahasiswa → IOD kecil → f lebih kecil (kompensasi)
            focal = (iod_px / self.REF_IOD_PX) * self.REF_FOCAL
            # Clamp agar tidak terlalu ekstrem
            focal = float(np.clip(focal, self._fw * 0.5, self._fw * 2.0))
        else:
            focal = float(self._fw)

        return np.array([
            [focal,   0,  cx],
            [0,   focal,  cy],
            [0,       0,   1],
        ], dtype=np.float64)

    def estimate(self, face_data: FaceData) -> HeadPoseResult:
        """Estimasi pose kepala dengan adaptive camera matrix."""
        lm_px  = face_data.landmarks_px
        pts_2d = np.array([lm_px[i] for i in self.LM_IDX], dtype=np.float64)

        # Camera matrix adaptif untuk wajah ini
        cam_mat = self._build_adaptive_matrix(face_data)

        success, rvec, tvec = cv2.solvePnP(
            self.FACE_3D, pts_2d, cam_mat, self._dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return HeadPoseResult(0, 0, 0, np.zeros(3), np.zeros(3), False)

        rot_mat, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = self._euler(rot_mat)

        # Koreksi arah (konvensi intuitif + flip compensation)
        pitch = -pitch
        yaw   = -yaw

        # Normalisasi flipped solution
        if pitch > 90.0:
            pitch =  180.0 - pitch
            yaw   = -yaw
            roll  =  roll - 180.0 if roll > 0 else roll + 180.0
        elif pitch < -90.0:
            pitch = -180.0 - pitch
            yaw   = -yaw
            roll  =  roll - 180.0 if roll > 0 else roll + 180.0

        return HeadPoseResult(pitch=pitch, yaw=yaw, roll=roll,
                              rvec=rvec, tvec=tvec, success=True)

    @staticmethod
    def _euler(R: np.ndarray) -> Tuple[float, float, float]:
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        if sy > 1e-6:
            pitch = np.arctan2( R[2,1],  R[2,2])
            yaw   = np.arctan2(-R[2,0],  sy)
            roll  = np.arctan2( R[1,0],  R[0,0])
        else:
            pitch = np.arctan2(-R[1,2], R[1,1])
            yaw   = np.arctan2(-R[2,0], sy)
            roll  = 0.0
        return (float(np.degrees(pitch)),
                float(np.degrees(yaw)),
                float(np.degrees(roll)))

    def draw_pose_axes(self, frame, result, face_data, axis_length=50.0):
        if not result.success:
            return frame
        lm      = face_data.landmarks_px
        cam_mat = self._build_adaptive_matrix(face_data)
        axis_3d = np.float32([
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, axis_length],
        ])
        axis_2d, _ = cv2.projectPoints(
            axis_3d, result.rvec, result.tvec, cam_mat, np.zeros((4,1)))
        axis_2d = axis_2d.reshape(-1, 2).astype(int)
        origin  = tuple(lm[4])
        cv2.arrowedLine(frame, origin, tuple(axis_2d[0]), (0,0,220),  2, tipLength=0.2)
        cv2.arrowedLine(frame, origin, tuple(axis_2d[1]), (0,200,0),  2, tipLength=0.2)
        cv2.arrowedLine(frame, origin, tuple(axis_2d[2]), (220,0,0),  2, tipLength=0.2)
        return frame