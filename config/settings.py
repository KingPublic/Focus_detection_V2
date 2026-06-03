# =============================================================================
# config/settings.py — Classroom Monitor (System 2)
# =============================================================================

# --- CAMERA ---
CAMERA_INDEX    = 0
FRAME_WIDTH     = 1280
FRAME_HEIGHT    = 720
TARGET_FPS      = 30

# --- MEDIAPIPE ---
# Untuk kelas, set tinggi. FPS akan turun jika terlalu banyak wajah.
# Rekomendasi: 10 untuk kelas kecil, 15 untuk kelas besar.
MAX_NUM_FACES               = 15
MIN_DETECTION_CONFIDENCE    = 0.72  # Dinaikkan: filter objek bukan wajah
MIN_TRACKING_CONFIDENCE     = 0.65  # Dinaikkan: tracking lebih stabil
REFINE_LANDMARKS            = True

# --- ZONE-BASED TRACKING (2 Students) ---
# Frame dibagi dua secara vertikal:
#   Zona KIRI  (x < SPLIT_RATIO * frame_width) = Student_01
#   Zona KANAN (x >= SPLIT_RATIO * frame_width) = Student_02
SPLIT_RATIO           = 0.50   # 0.50 = tepat tengah, geser jika perlu
SHOW_SPLIT_LINE       = True   # Tampilkan garis pemisah zona di frame
NMS_OVERLAP_THRESHOLD = 0.40   # NMS: hapus bbox overlap berlebihan
MIN_FACE_SIZE_PX      = 55     # Minimal ukuran wajah (px)
ABSENT_TIMEOUT_SEC    = 3.0    # Detik sampai student dianggap absent
MAX_STUDENTS          = 2      # Maksimal 2 student (kiri + kanan)

# --- HEAD POSE THRESHOLDS (derajat) ---
YAW_THRESHOLD_LEFT    = -20.0
YAW_THRESHOLD_RIGHT   =  20.0
PITCH_DOWN_THRESHOLD  =  15.0
PITCH_UP_THRESHOLD    = -18.0

# --- EAR ---
EAR_CLOSED_THRESHOLD  = 0.20
EAR_CONSEC_FRAMES     = 3

# --- GAZE (IRIS) ---
GAZE_LEFT_THRESHOLD   = -0.35
GAZE_RIGHT_THRESHOLD  =  0.35

# --- TEMPORAL THRESHOLDS (detik) ---
DURATION_WARN_LOOK_AWAY   = 3.0
DURATION_WARN_LOOK_DOWN   = 4.0
DURATION_WARN_FACE_ABSENT = 2.0
DURATION_WARN_EYES_CLOSED = 3.0
DURATION_WARN_GAZE        = 2.5

DURATION_CRIT_LOOK_AWAY   = 5.0
DURATION_CRIT_LOOK_DOWN   = 6.0
DURATION_CRIT_FACE_ABSENT = 4.0
DURATION_CRIT_EYES_CLOSED = 5.0
DURATION_CRIT_GAZE        = 5.0

# --- SESSION SCORE PENALTY PER FRAME ---
PENALTY_WARNING  = 0.008
PENALTY_CRITICAL = 0.025

# --- AUDIO ---
AUDIO_COOLDOWN_WARN = 6.0
AUDIO_COOLDOWN_CRIT = 4.0

# --- DISPLAY ---
SHOW_POSE_AXES  = False    # Matikan di kelas — terlalu ramai
SIDEBAR_WIDTH   = 340

# --- WARNA (BGR) ---
COLOR_OK        = (0,   210,   0)
COLOR_WARN      = (0,   200, 255)
COLOR_CRITICAL  = (0,    30, 220)
COLOR_WHITE     = (255, 255, 255)
COLOR_GRAY      = (150, 150, 150)
COLOR_SIDEBAR   = (20,   20,  30)
COLOR_ABSENT    = (100, 100, 100)

# Warna grade di atas kepala
GRADE_COLORS = {
    "A": (0,   220,   0),
    "B": (0,   200, 140),
    "C": (0,   200, 255),
    "D": (0,   140, 255),
    "F": (0,    30, 220),
}

# --- BEHAVIOR LABELS ---
BEHAVIOR_LABELS = {
    "LOOKING_LEFT":   "Look Left",
    "LOOKING_RIGHT":  "Look Right",
    "LOOKING_DOWN":   "Look Down",
    "LOOKING_UP":     "Look Up",
    "FACE_ABSENT":    "Face Absent",
    "EYES_CLOSED":    "Eyes Closed",
    "EYE_LOOK_LEFT":  "Gaze Left",
    "EYE_LOOK_RIGHT": "Gaze Right",
}