#!/usr/bin/env python3
# =============================================================================
# main.py — Classroom Monitor (System 2)
# Multi-Student Real-Time Focus & Behavior Monitoring
#
# Perbedaan dengan System 1:
#   - Deteksi unlimited mahasiswa sekaligus
#   - Setiap mahasiswa punya ID persistent (IoU tracking)
#   - Grade & score ditampilkan langsung di atas kepala
#   - Sidebar berisi roster semua mahasiswa
#   - Semua behavior System 1 berlaku per-mahasiswa
#
# Flow per frame:
#   Webcam → Frame
#     │
#     ├─ FaceDetector → N wajah (bbox + 468 landmark masing-masing)
#     │
#     ├─ StudentTracker (IoU matching)
#     │     → assign / update Student ID untuk tiap wajah
#     │
#     ├─ Per student (paralel):
#     │     ├─ HeadPoseEstimator → pitch, yaw, roll
#     │     ├─ EyeTracker        → EAR, iris gaze
#     │     ├─ evaluate_behaviors() → set behavior aktif
#     │     ├─ TemporalTracker   → durasi tiap behavior
#     │     ├─ compute_severity() → OK/WARNING/CRITICAL
#     │     └─ StudentState.update_behaviors()
#     │           → ScoreCalculator + SessionLogger update
#     │
#     ├─ ClassroomVisualRenderer
#     │     → bbox + badge per student + sidebar roster
#     │
#     └─ AudioAlert (jika ada student CRITICAL)
# =============================================================================

import sys
import time
import argparse
import cv2
import numpy as np

import config.settings as cfg
from detection.face_detector     import FaceDetector
from detection.head_pose          import HeadPoseEstimator, HeadPoseResult
from detection.eye_tracker        import EyeTrackResult
from detection.behavior_analyzer  import evaluate_behaviors, compute_severity
from tracking.student_tracker     import StudentTracker
from alert.classroom_visual       import ClassroomVisualRenderer
from alert.audio_alert            import AudioAlert
from utils.fps_counter            import FPSCounter


def parse_args():
    p = argparse.ArgumentParser(description="Classroom Monitor — System 2")
    p.add_argument("--camera",    type=int, default=cfg.CAMERA_INDEX)
    p.add_argument("--width",     type=int, default=cfg.FRAME_WIDTH)
    p.add_argument("--height",    type=int, default=cfg.FRAME_HEIGHT)
    p.add_argument("--no-audio",  action="store_true")
    p.add_argument("--show-lm",   action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  CLASSROOM MONITOR — System 2")
    print("  Multi-Student Rule-Based CV Monitoring")
    print("=" * 60)
    print(f"  Camera       : {args.camera}")
    print(f"  Max faces    : {cfg.MAX_NUM_FACES}")
    print(f"  Audio        : {'OFF' if args.no_audio else 'ON'}")
    print(f"  Press Q/ESC to quit | R to reset | S for screenshot")
    print("=" * 60)

    # ── Buka kamera ───────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Kamera {args.camera} tidak dapat dibuka.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS,          cfg.TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Camera] {frame_w}x{frame_h}")

    # ── Inisialisasi modul ────────────────────────────────────────────
    face_detector  = FaceDetector()
    head_pose_est  = HeadPoseEstimator(frame_w, frame_h)
    student_tracker= StudentTracker()
    visual_render  = ClassroomVisualRenderer(frame_w, frame_h)
    audio_alert    = AudioAlert()
    fps_counter    = FPSCounter(window_size=30)

    # EyeTracker: satu instance per student tidak efisien untuk classroom
    # (EAR counter per-student), jadi kita hitung EAR stateless per frame
    from detection.eye_tracker import EyeTracker
    _eye_trackers = {}   # student_id → EyeTracker instance

    print("[System] All modules initialized.\n")

    # ── Dummy results untuk saat wajah tidak terdeteksi ───────────────
    _dummy_hp = HeadPoseResult(0, 0, 0,
                               np.zeros(3), np.zeros(3), False)

    prev_had_critical = False

    # ================================================================ #
    #  MAIN LOOP
    # ================================================================ #
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        fps_counter.tick()

        # ── STEP 1: Deteksi semua wajah ────────────────────────────────
        face_present, faces = face_detector.process(frame)

        # ── STEP 2: Student Tracker (IoU matching → persistent ID) ─────
        bboxes   = [f.face_rect for f in faces] if face_present else []
        students = student_tracker.update(bboxes)

        # ── STEP 3: Per-student analysis ───────────────────────────────
        for face_data, student in zip(faces, students):
            # Pastikan student punya EyeTracker instance sendiri
            if student.id_num not in _eye_trackers:
                _eye_trackers[student.id_num] = EyeTracker()

            et = _eye_trackers[student.id_num]

            # Head Pose
            hp_result = head_pose_est.estimate(face_data)

            # Eye / EAR
            eye_result = et.analyze(face_data)

            # Behavior Rules
            active_behaviors = evaluate_behaviors(
                hp_yaw   = hp_result.yaw   if hp_result.success else 0.0,
                hp_pitch = hp_result.pitch if hp_result.success else 0.0,
                hp_ok    = hp_result.success,
                ear      = eye_result.ear_avg,
                eye_closed = eye_result.eye_closed,
                gaze_x   = eye_result.gaze_x,
            )

            # Temporal Analysis
            durations = student.temporal.update(active_behaviors)

            # Severity
            severity = compute_severity(active_behaviors, durations)

            # Update state student (juga update score & session)
            student.update_behaviors(
                behaviors  = active_behaviors,
                durations  = durations,
                severity   = severity,
                pitch      = hp_result.pitch if hp_result.success else 0.0,
                yaw        = hp_result.yaw   if hp_result.success else 0.0,
                ear        = eye_result.ear_avg,
                eye_closed = eye_result.eye_closed,
            )

            # Debug pose axes (opsional)
            if cfg.SHOW_POSE_AXES and hp_result.success:
                head_pose_est.draw_pose_axes(frame, hp_result, face_data,
                                             axis_length=40.0)
            if args.show_lm:
                face_detector.draw_landmarks(frame, face_data)

        # ── STEP 4: Tandai student yang tidak terdeteksi frame ini ─────
        detected_ids = {s.id_num for s in students}
        for s in student_tracker.get_all_students():
            if s.id_num not in detected_ids:
                # update FACE_ABSENT
                durations = s.temporal.update({"FACE_ABSENT"})
                severity  = compute_severity({"FACE_ABSENT"}, durations)
                s.update_behaviors({"FACE_ABSENT"}, durations, severity,
                                   0, 0, 0.30, False)

        # ── STEP 5: Visual Render ──────────────────────────────────────
        all_students = student_tracker.get_all_students()
        canvas = visual_render.render(frame, all_students, fps_counter.fps)

        # ── STEP 6: Audio Alert ────────────────────────────────────────
        if not args.no_audio:
            has_critical = any(s.severity == "CRITICAL" for s in all_students)
            has_warning  = any(s.severity == "WARNING"  for s in all_students)

            if has_critical:
                audio_alert.alert_critical(cooldown=cfg.AUDIO_COOLDOWN_CRIT)
            elif has_warning:
                audio_alert.alert_warning(cooldown=cfg.AUDIO_COOLDOWN_WARN)
            elif prev_had_critical:
                audio_alert.alert_clear()

            prev_had_critical = has_critical

        # ── STEP 7: Display ────────────────────────────────────────────
        cv2.imshow("Classroom Monitor", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            print("[System] Quit.")
            break
        elif key == ord('r'):
            student_tracker.reset()
            _eye_trackers.clear()
            print("[System] Session reset — all students cleared.")
        elif key == ord('s'):
            fname = f"classroom_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(fname, canvas)
            print(f"[System] Screenshot: {fname}")

    # ── Cleanup ───────────────────────────────────────────────────────
    cap.release()
    face_detector.release()
    cv2.destroyAllWindows()

    # ── Laporan akhir sesi ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL SESSION REPORT")
    print("=" * 60)
    all_ever = student_tracker.get_all_ever()
    if all_ever:
        print(f"  {'Student':<14} {'Score':>7}  {'Grade'}  "
              f"{'Warn':>5}  {'Crit':>5}")
        print("  " + "-" * 46)
        for s in all_ever:
            print(f"  {s.label:<14} "
                  f"{s.overall_score:>6.1f}   "
                  f"  {s.grade}     "
                  f"{s.session.warn_events:>4}   "
                  f"{s.session.crit_events:>4}")
    else:
        print("  No students detected in this session.")
    print("=" * 60)


if __name__ == "__main__":
    main()