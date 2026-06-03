#!/usr/bin/env python3
# =============================================================================
# main.py — Classroom Monitor  Zone-Based Edition
#
# Tracking: Zone-Based 2-Student
#   Kiri  = Student_01 | Kanan = Student_02
#   Persistent: pergi dan kembali tetap ID yang sama
# =============================================================================

import sys
import time
import argparse
import cv2
import numpy as np

import config.settings as cfg
from detection.face_detector    import FaceDetector
from detection.head_pose         import HeadPoseEstimator
from detection.eye_tracker       import EyeTracker
from detection.behavior_analyzer import evaluate_behaviors, compute_severity
from tracking.student_tracker    import StudentTracker
from alert.classroom_visual      import ClassroomVisualRenderer
from alert.audio_alert           import AudioAlert
from utils.fps_counter           import FPSCounter


def parse_args():
    p = argparse.ArgumentParser(description="Classroom Monitor — Zone-Based")
    p.add_argument("--camera",   type=int, default=cfg.CAMERA_INDEX)
    p.add_argument("--width",    type=int, default=cfg.FRAME_WIDTH)
    p.add_argument("--height",   type=int, default=cfg.FRAME_HEIGHT)
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--split",    type=float, default=cfg.SPLIT_RATIO,
                   help="Rasio split kiri/kanan (default 0.5 = tengah)")
    return p.parse_args()


def main():
    args = parse_args()

    # Override split ratio dari argument jika ada
    cfg.SPLIT_RATIO = args.split

    print("=" * 60)
    print("  CLASSROOM MONITOR — Zone-Based 2 Student")
    print(f"  Split: {args.split*100:.0f}% | Camera: {args.camera}")
    print("  Q/ESC=quit | R=reset scores | S=screenshot")
    print("  Tip: jalankan dengan --split 0.4 jika S01 terlalu sempit")
    print("=" * 60)

    # ── Kamera ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Kamera {args.camera} tidak bisa dibuka.")
        sys.exit(1)

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
    student_tracker = StudentTracker(frame_width=frame_w)
    visual_render   = ClassroomVisualRenderer(frame_w, frame_h)
    audio_alert     = AudioAlert()
    fps_counter     = FPSCounter(window_size=30)

    # EyeTracker per student (S01=id 1, S02=id 2)
    eye_trackers = {1: EyeTracker(), 2: EyeTracker()}

    prev_had_critical = False
    split_x = student_tracker.get_split_x()

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

        # ── Garis zona pemisah (visual guide) ─────────────────────────
        if cfg.SHOW_SPLIT_LINE:
            cv2.line(frame, (split_x, 0), (split_x, frame_h),
                     (80, 80, 80), 1, cv2.LINE_AA)
            cv2.putText(frame, "S01", (split_x // 2 - 15, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (80, 80, 80), 1, cv2.LINE_AA)
            cv2.putText(frame, "S02", (split_x + (frame_w - split_x) // 2 - 15, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (80, 80, 80), 1, cv2.LINE_AA)

        # ── STEP 1: Deteksi wajah ──────────────────────────────────────
        face_present, faces = face_detector.process(frame)

        # ── STEP 2: Zone assignment ───────────────────────────────────
        if face_present and faces:
            bboxes           = [f.face_rect for f in faces]
            matched_students = student_tracker.update(bboxes)
        else:
            matched_students = []
            student_tracker.update([])   # trigger absent check

        # ── STEP 3: Analisis per wajah yang terdeteksi ─────────────────
        processed_ids = set()

        for i in range(len(faces)):
            student = matched_students[i] if i < len(matched_students) else None
            if student is None:
                continue
            if student.id_num in processed_ids:
                continue   # Cegah duplikat (jika 2 wajah masuk zona sama)
            processed_ids.add(student.id_num)

            face_data = faces[i]

            try:
                et  = eye_trackers[student.id_num]
                hp  = head_pose_est.estimate(face_data)
                eye = et.analyze(face_data)

                behaviors = evaluate_behaviors(
                    hp_yaw    = hp.yaw    if hp.success else 0.0,
                    hp_pitch  = hp.pitch  if hp.success else 0.0,
                    hp_ok     = hp.success,
                    ear       = eye.ear_avg,
                    eye_closed= eye.eye_closed,
                    gaze_x    = eye.gaze_x,
                )

                durations = student.temporal.update(behaviors)
                severity  = compute_severity(behaviors, durations)

                student.update_behaviors(
                    behaviors = behaviors,
                    durations = durations,
                    severity  = severity,
                    pitch     = hp.pitch if hp.success else 0.0,
                    yaw       = hp.yaw   if hp.success else 0.0,
                    ear       = eye.ear_avg,
                    eye_closed= eye.eye_closed,
                )

            except Exception as e:
                print(f"[WARN] {student.label}: {e}")
                continue

        # ── STEP 4: Update student yang tidak terdeteksi frame ini ─────
        for s in student_tracker.get_all_students():
            if s.id_num not in processed_ids and not s.is_active:
                durations = s.temporal.update({"FACE_ABSENT"})
                severity  = compute_severity({"FACE_ABSENT"}, durations)
                s.update_behaviors({"FACE_ABSENT"}, durations, severity,
                                   0, 0, 0.30, False)

        # ── STEP 5: Render ─────────────────────────────────────────────
        all_students = student_tracker.get_all_students()
        try:
            canvas = visual_render.render(frame, all_students, fps_counter.fps)
        except Exception as e:
            print(f"[ERROR] Render gagal: {e}")
            import traceback; traceback.print_exc()
            canvas = frame   # fallback: tampilkan frame mentah

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

        # ── STEP 7: Display ────────────────────────────────────────────
        cv2.imshow("Classroom Monitor", canvas)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            break
        elif key == ord('r'):
            student_tracker.reset()
            eye_trackers = {1: EyeTracker(), 2: EyeTracker()}
            print("[System] Scores reset — zones tetap.")
        elif key == ord('s'):
            fn = f"classroom_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(fn, canvas)
            print(f"[Screenshot] {fn}")

    # ── Cleanup ───────────────────────────────────────────────────────
    cap.release()
    face_detector.release()
    cv2.destroyAllWindows()

    # Laporan akhir
    print("\n" + "=" * 60)
    print("  FINAL REPORT")
    print("=" * 60)
    for s in student_tracker.get_all_ever():
        status = "Active" if s.is_active else "Absent"
        print(f"  {s.label}  Score:{s.overall_score:.1f}  "
              f"Grade:{s.grade}  "
              f"Warn:{s.session.warn_events}  "
              f"Crit:{s.session.crit_events}  [{status}]")
    print("=" * 60)


if __name__ == "__main__":
    main()