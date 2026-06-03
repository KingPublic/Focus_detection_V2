import os, sys

folder = sys.argv[1] if len(sys.argv) > 1 else "."

checks = {
    "detection/face_detector.py":   "MIN_FACE_SIZE_PX",
    "detection/head_pose.py":       "_build_adaptive_matrix",
    "tracking/student_tracker.py":  "ZoneTracker",
    "main.py":                      "Zone-Based",
    "config/settings.py":           "SPLIT_RATIO",
    "alert/classroom_visual.py":    "ClassroomVisualRenderer",
    "tracking/student_state.py":    "StudentState",
    "detection/behavior_analyzer.py": "evaluate_behaviors",
}

print(f"\nVersion check: {folder}\n")
all_ok = True
for path, keyword in checks.items():
    full = os.path.join(folder, path)
    if not os.path.exists(full):
        print(f"  MISSING  {path}")
        all_ok = False
        continue
    content = open(full).read()
    ok = keyword in content
    status = "OK  " if ok else "OLD "
    if not ok: all_ok = False
    print(f"  {status}   {path}  (need: '{keyword}')")

print()
if all_ok:
    print("Semua file sudah versi terbaru.")
else:
    print("File bertanda OLD perlu diganti dengan versi terbaru.")