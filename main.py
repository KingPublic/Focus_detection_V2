#!/usr/bin/env python3
# =============================================================================
# main.py — Classroom Monitor v3
#
# FIX KRITIS v3:
#   1. zip(faces, students) diganti dengan indeks eksplisit setelah NMS
#      → tiap face_data dijamin ke-assign ke student yang benar
#   2. Sinkronisasi faces ↔ tracker: bboxes yang masuk ke tracker
#      diambil DARI faces hasil NMS (bukan dari hasil mediapipe langsung)
#      sehingga urutan selalu konsisten
#   3. Error handling di setiap step per-student → crash terlokalisir,
#      tidak menghentikan seluruh program
# =============================================================================

import sys
import time
import argparse
import cv2
import numpy as np

import config.settings as cfg
from detection.face_detector     import FaceDetector
from detection.head_pose          import HeadPoseEstimator, HeadPoseResult
from detection.eye_tracker        import EyeTracker
from detection.behavior_analyzer  import evaluate_behaviors, compute_severity
from tracking.student_tracker     import StudentTracker
from alert.classroom_visual       import ClassroomVisualRenderer
from alert.audio_alert            import AudioAlert
from utils.fps_counter            import FPSCounter


def parse_args():
    p = argparse.ArgumentParser(description="Classroom Monitor v3")
    p.add_argument("--camera",   type=int, default=cfg.CAMERA_INDEX)
    p.add_argument("--width",    type=int, default=cfg.FRAME_WIDTH)
    p.add_argument("--height",   type=int, default=cfg.FRAME_HEIGHT)
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--show-lm",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("  CLASSROOM MONITOR v3 — Multi-Student")
    print(f"  Camera: {args.camera} | MaxFaces: {cfg.MAX_NUM_FACES}")
    print("  Q/ESC = quit | R = reset | S = screenshot")
    print("=" * 60)

    # ── Kamera ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Kamera {args.camera} tidak bisa dibuka."); sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS,          cfg.TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Camera] {frame_w}x{frame_h}")

    # ── Modul ─────────────────────────────────────────────────────────
    face_detector   = FaceDetector()
    head_pose_est   = HeadPoseEstimator(frame_w, frame_h)
    student_tracker = StudentTracker()
    visual_render   = ClassroomVisualRenderer(frame_w, frame_h)
    audio_alert     = AudioAlert()
    fps_counter     = FPSCounter(window_size=30)

    # EyeTracker per student (dict: student_id → EyeTracker)
    eye_trackers: dict = {}

    prev_had_critical = False
    print("[System] Ready.\n")

    # ================================================================= #
    #  MAIN LOOP
    # ================================================================= #
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        frame = cv2.flip(frame, 1)
        fps_counter.tick()

        # ── STEP 1: Deteksi wajah (dengan NMS di dalamnya) ─────────────
        face_present, faces = face_detector.process(frame)

        # ── STEP 2: Kirim bbox ke tracker — SEJAJAR dengan faces ───────
        # KUNCI FIX: bboxes diambil dari faces hasil NMS, bukan raw mediapipe
        # Sehingga faces[i] ↔ bboxes[i] ↔ matched_students[i] SELALU SAMA
        if face_present and faces:
            bboxes          = [f.face_rect for f in faces]
            matched_students = student_tracker.update(bboxes)
            # matched_students[i] = StudentState untuk faces[i]
        else:
            matched_students = []

        # ── STEP 3: Analisis per student ───────────────────────────────
        # Iterasi dengan indeks eksplisit — BUKAN zip langsung
        for i in range(len(faces)):
            face_data = faces[i]
            student   = matched_students[i]

            if student is None:
                continue

            try:
                # EyeTracker per student
                if student.id_num not in eye_trackers:
                    eye_trackers[student.id_num] = EyeTracker()
                et = eye_trackers[student.id_num]

                # Head Pose
                hp = head_pose_est.estimate(face_data)

                # Eye / EAR
                eye = et.analyze(face_data)

                # Behavior Rules
                behaviors = evaluate_behaviors(
                    hp_yaw    = hp.yaw    if hp.success else 0.0,
                    hp_pitch  = hp.pitch  if hp.success else 0.0,
                    hp_ok     = hp.success,
                    ear       = eye.ear_avg,
                    eye_closed= eye.eye_closed,
                    gaze_x    = eye.gaze_x,
                )

                # Temporal
                durations = student.temporal.update(behaviors)

                # Severity
                severity = compute_severity(behaviors, durations)

                # Update state student ini (score, session, dll)
                student.update_behaviors(
                    behaviors  = behaviors,
                    durations  = durations,
                    severity   = severity,
                    pitch      = hp.pitch if hp.success else 0.0,
                    yaw        = hp.yaw   if hp.success else 0.0,
                    ear        = eye.ear_avg,
                    eye_closed = eye.eye_closed,
                )

                if cfg.SHOW_POSE_AXES and hp.success:
                    head_pose_est.draw_pose_axes(frame, hp, face_data, 40.0)

            except Exception as e:
                # Error per-student tidak crash seluruh program
                print(f"[WARN] Student {student.label} error: {e}")
                continue

        # ── STEP 4: Update student yang tidak terdeteksi ───────────────
        detected_ids = {matched_students[i].id_num
                        for i in range(len(matched_students))
                        if matched_students[i] is not None}

        for s in student_tracker.get_all_students():
            if s.id_num not in detected_ids:
                durations = s.temporal.update({"FACE_ABSENT"})
                severity  = compute_severity({"FACE_ABSENT"}, durations)
                s.update_behaviors({"FACE_ABSENT"}, durations, severity,
                                   0, 0, 0.30, False)

        # ── STEP 5: Render ─────────────────────────────────────────────
        all_students = student_tracker.get_all_students()
        canvas = visual_render.render(frame, all_students, fps_counter.fps)

        # ── STEP 6: Audio ──────────────────────────────────────────────
        if not args.no_audio:
            has_crit = any(s.severity == "CRITICAL" for s in all_students)
            has_warn = any(s.severity == "WARNING"  for s in all_students)
            if has_crit:
                audio_alert.alert_critical(cooldown=cfg.AUDIO_COOLDOWN_CRIT)
            elif has_warn:
                audio_alert.alert_warning(cooldown=cfg.AUDIO_COOLDOWN_WARN)
            elif prev_had_critical:
                audio_alert.alert_clear()
            prev_had_critical = has_crit

        # ── STEP 7: Tampilkan ──────────────────────────────────────────
        cv2.imshow("Classroom Monitor", canvas)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            print("[System] Quit.")
            break
        elif key == ord('r'):
            student_tracker.reset()
            eye_trackers.clear()
            print("[System] Reset — semua student dihapus.")
        elif key == ord('s'):
            fn = f"classroom_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(fn, canvas)
            print(f"[System] Screenshot: {fn}")

    # ── Cleanup ───────────────────────────────────────────────────────
    cap.release()
    face_detector.release()
    cv2.destroyAllWindows()

    # ── Laporan akhir ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL SESSION REPORT")
    print("=" * 60)
    all_ever = student_tracker.get_all_ever()
    if all_ever:
        print(f"  {'Student':<14} {'Score':>7}  {'Grade'}  {'Warn':>5}  {'Crit':>5}")
        print("  " + "-" * 46)
        for s in all_ever:
            print(f"  {s.label:<14} {s.overall_score:>6.1f}     {s.grade}"
                  f"     {s.session.warn_events:>4}   {s.session.crit_events:>4}")
    print("=" * 60)


if __name__ == "__main__":
    main()