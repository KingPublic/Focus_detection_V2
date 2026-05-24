# =============================================================================
# alert/classroom_visual.py
# Classroom Visual Renderer
#
# Layout:
#   ┌────────────────────────────────────────────┬──────────────┐
#   │                                            │   SIDEBAR    │
#   │   CAMERA FRAME                             │              │
#   │                                            │  S01  A 98   │
#   │   [S01]──[Grade:A 98.2]                   │  S02  C 64   │
#   │   [S02]──[Grade:C 64.1]                   │  S03  B 81   │
#   │                                            │  ...         │
#   │                                            │              │
#   └────────────────────────────────────────────┴──────────────┘
#
# Di atas setiap wajah:
#   ┌─ S01 ─────────────┐
#   │ Grade: A  98.2/100 │
#   │ [*] Look Left 3.1s │   ← hanya tampil jika ada behavior
#   └────────────────────┘
# =============================================================================

import cv2
import numpy as np
from typing import List, Dict
import config.settings as cfg
from tracking.student_state import StudentState


class ClassroomVisualRenderer:
    """
    Renderer visual untuk kelas — multi-student overlay.

    Tiap wajah mendapat:
      - Bounding box berwarna sesuai severity
      - Badge di atas kepala: ID | Grade | Score
      - Label behavior aktif (jika ada)

    Sidebar kanan:
      - Tabel ringkas semua student
      - Class summary (rata-rata score, jumlah suspicious)
    """

    SIDEBAR_W = cfg.SIDEBAR_WIDTH

    def __init__(self, frame_w: int, frame_h: int):
        self._fw = frame_w
        self._fh = frame_h
        print(f"[ClassroomVisual] Initialized {frame_w}x{frame_h}")

    # ================================================================== #
    #  Main Entry
    # ================================================================== #

    def render(self, frame: np.ndarray,
               students: List[StudentState],
               fps: float) -> np.ndarray:
        """
        Render frame lengkap dengan overlay per-student dan sidebar.

        Returns:
            Canvas = sidebar (kanan) + frame (kiri)
        """
        out = frame.copy()

        # Gambar overlay per student di atas frame
        for student in students:
            self._draw_student_overlay(out, student)

        # Header frame
        self._draw_frame_header(out, students, fps)

        # Build sidebar
        sidebar = self._build_sidebar(students)

        return np.hstack([out, sidebar])

    # ================================================================== #
    #  Per-Student Overlay (di atas frame kamera)
    # ================================================================== #

    def _draw_student_overlay(self, frame: np.ndarray,
                               student: StudentState):
        """Gambar bbox + badge info di atas kepala satu student."""
        if not student.is_active:
            return

        x, y, w, h = student.bbox
        sev = student.severity

        # ── Warna berdasarkan severity ─────────────────────────────────
        box_color = {
            "OK":       cfg.COLOR_OK,
            "WARNING":  cfg.COLOR_WARN,
            "CRITICAL": cfg.COLOR_CRITICAL,
        }.get(sev, cfg.COLOR_WHITE)

        thickness = 3 if sev == "CRITICAL" else 2

        # ── Bounding box wajah ─────────────────────────────────────────
        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, thickness)

        # ── Badge di atas wajah ────────────────────────────────────────
        grade_col = cfg.GRADE_COLORS.get(student.grade, cfg.COLOR_WHITE)
        badge_lines = self._make_badge_lines(student)

        badge_h = 18 * len(badge_lines) + 8
        badge_w = max(len(line) * 8 + 10 for line in badge_lines)
        badge_w = max(badge_w, w)  # minimal selebar bbox

        bx1 = x
        by1 = max(0, y - badge_h - 4)
        bx2 = bx1 + badge_w
        by2 = by1 + badge_h

        # Background badge semi-transparan
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (15, 15, 25), -1)
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), box_color, 1)
        frame[:] = cv2.addWeighted(overlay, 0.80, frame, 0.20, 0)

        # Teks badge
        ty = by1 + 14
        for i, line in enumerate(badge_lines):
            col = grade_col if i == 0 else (
                cfg.COLOR_CRITICAL if sev == "CRITICAL" and i > 0 else
                cfg.COLOR_WARN     if sev == "WARNING"  and i > 0 else
                cfg.COLOR_WHITE
            )
            cv2.putText(frame, line, (bx1 + 4, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        col, 1, cv2.LINE_AA)
            ty += 18

    def _make_badge_lines(self, student: StudentState) -> List[str]:
        """Buat baris-baris teks untuk badge di atas wajah."""
        lines = []

        # Baris 1: ID + Grade + Score
        lines.append(
            f"{student.short_lbl}  {student.grade}  "
            f"{student.overall_score:.1f}/100"
        )

        # Baris 2+: behavior aktif yang sudah melewati warning
        for beh in sorted(student.active_behaviors):
            dur   = student.durations.get(beh, 0.0)
            label = cfg.BEHAVIOR_LABELS.get(beh, beh)
            if dur >= 1.0:   # hanya tampilkan jika sudah > 1 detik
                lines.append(f"[!] {label} {dur:.1f}s")

        return lines

    # ================================================================== #
    #  Frame Header
    # ================================================================== #

    def _draw_frame_header(self, frame: np.ndarray,
                           students: List[StudentState],
                           fps: float):
        """Bar tipis di bagian atas frame — summary singkat."""
        n_active   = sum(1 for s in students if s.is_active)
        n_critical = sum(1 for s in students if s.severity == "CRITICAL")
        n_warning  = sum(1 for s in students if s.severity == "WARNING")

        bar_color = (0, 30, 220)  if n_critical > 0 else \
                    (0, 140, 200) if n_warning  > 0 else \
                    (20, 80, 20)

        cv2.rectangle(frame, (0, 0), (self._fw, 30), bar_color, -1)

        txt = (f"CLASSROOM MONITOR  |  "
               f"Active: {n_active}  "
               f"Warning: {n_warning}  "
               f"Critical: {n_critical}  |  "
               f"FPS: {fps:.1f}")
        cv2.putText(frame, txt, (10, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    cfg.COLOR_WHITE, 1, cv2.LINE_AA)

    # ================================================================== #
    #  Sidebar — Daftar Semua Student
    # ================================================================== #

    def _build_sidebar(self, students: List[StudentState]) -> np.ndarray:
        """Sidebar kanan berisi tabel semua student + class summary."""
        sb = np.full((self._fh, self.SIDEBAR_W, 3),
                     cfg.COLOR_SIDEBAR, dtype=np.uint8)
        y  = 18

        # ── Header sidebar ─────────────────────────────────────────────
        y = self._txt(sb, "STUDENT ROSTER", (10, y),
                      scale=0.58, color=cfg.COLOR_WHITE, bold=True)
        y = self._txt(sb, f"Total detected: {len(students)}", (10, y),
                      scale=0.40, color=cfg.COLOR_GRAY)
        y += 4
        self._hline(sb, y); y += 8

        # ── Class Summary ──────────────────────────────────────────────
        y = self._draw_class_summary(sb, y, students)
        self._hline(sb, y); y += 8

        # ── Kolom header tabel ─────────────────────────────────────────
        y = self._txt(sb, " ID    Score  Gr  Status     Behaviors", (6, y),
                      scale=0.36, color=cfg.COLOR_GRAY)
        self._hline(sb, y, color=(50, 50, 50)); y += 4

        # ── Satu baris per student ─────────────────────────────────────
        # Hitung berapa baris yang muat
        row_h      = 40
        max_rows   = (self._fh - y - 20) // row_h

        # Urutkan: CRITICAL dulu, lalu WARNING, lalu OK, lalu Absent
        order = {"CRITICAL": 0, "WARNING": 1, "OK": 2}
        sorted_students = sorted(
            students,
            key=lambda s: (0 if not s.is_active else order.get(s.severity, 2),
                           s.id_num)
        )

        for i, student in enumerate(sorted_students[:max_rows]):
            y = self._draw_student_row(sb, y, student, row_h)

        # Indikator jika ada student yang tidak tampil
        if len(sorted_students) > max_rows:
            extra = len(sorted_students) - max_rows
            self._txt(sb, f"  ... +{extra} more students",
                      (10, self._fh - 25),
                      scale=0.38, color=cfg.COLOR_GRAY)

        # Footer
        cv2.putText(sb, "IEEE Classroom Monitor",
                    (self.SIDEBAR_W // 2 - 82, self._fh - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    cfg.COLOR_GRAY, 1, cv2.LINE_AA)
        return sb

    def _draw_class_summary(self, sb: np.ndarray, y: int,
                             students: List[StudentState]) -> int:
        """Ringkasan statistik kelas."""
        active = [s for s in students if s.is_active]
        if not active:
            return self._txt(sb, "  No students detected", (10, y),
                             scale=0.42, color=cfg.COLOR_GRAY)

        scores     = [s.overall_score for s in active]
        avg_score  = sum(scores) / len(scores)
        min_score  = min(scores)
        n_crit     = sum(1 for s in active if s.severity == "CRITICAL")
        n_warn     = sum(1 for s in active if s.severity == "WARNING")

        # Warna avg score
        avg_col = (cfg.COLOR_OK       if avg_score >= 75 else
                   cfg.COLOR_WARN     if avg_score >= 50 else
                   cfg.COLOR_CRITICAL)

        y = self._txt(sb, "CLASS SUMMARY", (10, y),
                      scale=0.46, color=cfg.COLOR_WHITE, bold=True)
        y = self._txt(sb, f"  Avg Score : {avg_score:.1f}  "
                          f"Min: {min_score:.1f}",
                      (10, y), scale=0.40, color=avg_col)
        y = self._txt(sb, f"  Active: {len(active)}  "
                          f"Warn: {n_warn}  Crit: {n_crit}",
                      (10, y), scale=0.40, color=cfg.COLOR_WHITE)

        # Distribusi grade
        grade_counts: Dict[str,int] = {}
        for s in active:
            grade_counts[s.grade] = grade_counts.get(s.grade, 0) + 1
        grade_str = "  Grades: " + "  ".join(
            f"{g}:{c}" for g, c in sorted(grade_counts.items())
        )
        y = self._txt(sb, grade_str, (10, y), scale=0.38, color=cfg.COLOR_GRAY)
        y += 4
        return y

    def _draw_student_row(self, sb: np.ndarray, y: int,
                          student: StudentState, row_h: int) -> int:
        """Gambar satu baris info student di sidebar."""
        sev = student.severity if student.is_active else "ABSENT"

        row_color = {
            "OK":       cfg.COLOR_OK,
            "WARNING":  cfg.COLOR_WARN,
            "CRITICAL": cfg.COLOR_CRITICAL,
            "ABSENT":   cfg.COLOR_ABSENT,
        }.get(sev, cfg.COLOR_WHITE)

        # Background baris tipis
        if sev == "CRITICAL":
            cv2.rectangle(sb, (4, y - 2),
                          (self.SIDEBAR_W - 4, y + row_h - 4),
                          (0, 10, 60), -1)
        elif sev == "WARNING":
            cv2.rectangle(sb, (4, y - 2),
                          (self.SIDEBAR_W - 4, y + row_h - 4),
                          (0, 50, 60), -1)

        grade_col = cfg.GRADE_COLORS.get(student.grade, cfg.COLOR_WHITE)

        # Baris 1: ID | Score | Grade | Status
        line1 = (f" {student.short_lbl}  "
                 f"{student.overall_score:5.1f}  "
                 f" {student.grade}   "
                 f"{'ABSENT' if not student.is_active else sev}")
        self._txt(sb, line1, (6, y + 13),
                  scale=0.40, color=row_color)

        # Baris 2: behavior aktif (compact)
        if student.active_behaviors and student.is_active:
            behs = []
            for beh in sorted(student.active_behaviors):
                dur = student.durations.get(beh, 0.0)
                lbl = cfg.BEHAVIOR_LABELS.get(beh, beh)
                if dur >= 1.0:
                    behs.append(f"{lbl[:8]} {dur:.0f}s")
            if behs:
                beh_str = "  " + " | ".join(behs[:2])   # max 2 behavior
                self._txt(sb, beh_str, (6, y + 28),
                          scale=0.34, color=cfg.COLOR_WARN
                          if sev == "WARNING" else cfg.COLOR_CRITICAL)

        # Garis pemisah tipis
        cv2.line(sb, (10, y + row_h - 2),
                 (self.SIDEBAR_W - 10, y + row_h - 2),
                 (45, 45, 55), 1)

        return y + row_h

    # ================================================================== #
    #  Helpers
    # ================================================================== #

    def _txt(self, img, text, pos, scale=0.42,
             color=(255,255,255), bold=False) -> int:
        t = 2 if bold else 1
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, t, cv2.LINE_AA)
        return pos[1] + int(scale * 36) + 4

    def _hline(self, img, y, color=(55, 55, 65)):
        cv2.line(img, (6, y), (self.SIDEBAR_W - 6, y), color, 1)