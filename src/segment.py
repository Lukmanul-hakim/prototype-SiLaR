# src/segment.py
import cv2
import numpy as np

# ======================================================
# PARAMETER FINAL (SUDAH SEIMBANG)
# ======================================================
START_TH = 0.025
END_TH = 0.012

OBSERVE_FRAMES = 12

MIN_FRAMES = 20
MAX_FRAMES = 400        # gesture panjang aman

STATIC_HOLD_FRAMES = 40     # statis ~1.3 detik
POST_MOTION_HOLD = 30       # dinamis: harus BENAR-BENAR berhenti

STABLE_WINDOW = 20
STABLE_RATIO = 0.75

MIN_HAND_AREA = 0.02

PALM_IDX = [0, 5, 9, 13, 17]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

# ======================================================
# UTIL
# ======================================================
def normalize_pose(lm):
    lm = lm.copy()
    lm -= lm[0]
    return lm

def motion_score(prev, curr):
    if prev is None or curr is None:
        return 0.0
    p = normalize_pose(prev)
    c = normalize_pose(curr)
    return np.mean(np.linalg.norm(c[PALM_IDX] - p[PALM_IDX], axis=1))

def valid_hand(lm):
    if lm is None:
        return False
    pts = lm.reshape(-1, 2)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    area = (x_max - x_min) * (y_max - y_min)
    return area >= MIN_HAND_AREA

def draw_hand(frame, hand, color):
    h, w = frame.shape[:2]
    for x, y in hand:
        cv2.circle(frame, (int(x*w), int(y*h)), 4, color, -1)
    for i, j in HAND_CONNECTIONS:
        x1, y1 = hand[i]
        x2, y2 = hand[j]
        cv2.line(frame,
                 (int(x1*w), int(y1*h)),
                 (int(x2*w), int(y2*h)),
                 color, 2)

def draw_keypoints(frame, lm):
    if lm is None:
        return
    if np.any(lm[:21]):
        draw_hand(frame, lm[:21], (0,255,0))
    if np.any(lm[21:]):
        draw_hand(frame, lm[21:], (255,0,0))

def draw_ui(frame, state, label_text, stable_ratio=None):
    cv2.putText(frame, f"STATE: {state}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    if stable_ratio is not None:
        cv2.putText(frame, f"STABLE_RATIO: {stable_ratio:.2f}",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    cv2.putText(frame, f"LABEL: {label_text}",
                (10, frame.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    cv2.putText(frame,
                "Masukkan tangan | Gerakan = dinamis | Tahan = statis | ENTER = simpan | ESC = keluar",
                (10, frame.shape[0]-50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

def keyboard_input(key, current):
    if 65 <= key <= 90 or 97 <= key <= 122:
        return current + chr(key).upper()
    if key == 8:
        return current[:-1]
    return current

# ======================================================
# AUTO RECORD (FINAL)
# ======================================================
def auto_record(cam, extractor, label_state):
    state = "WAIT"

    buffer = []
    observe_counter = 0

    static_hold = 0
    post_motion_hold = 0
    stable_window = []

    has_motion = False
    prev_lm = None

    while True:
        frame, _ = cam.read()
        if frame is None:
            return None

        lm = extractor(frame)
        motion = motion_score(prev_lm, lm)
        prev_lm = lm

        disp = frame.copy()

        if valid_hand(lm):
            draw_keypoints(disp, lm)
        else:
            lm = None

        # ================= UI (SELALU DIGAMBAR) =================
        draw_ui(
            disp,
            state,
            label_state["text"],
            stable_ratio=None if state != "RECORD"
            else sum(stable_window)/max(1,len(stable_window))
        )

        # ================= LOGIC =================
        if state == "WAIT":
            if lm is not None:
                observe_counter = 1
                state = "OBSERVE"

        elif state == "OBSERVE":
            if lm is None:
                state = "WAIT"
            else:
                observe_counter += 1
                if motion > START_TH:
                    state = "RECORD"
                    buffer.clear()
                    has_motion = True
                    static_hold = 0
                    post_motion_hold = 0
                    stable_window.clear()
                elif observe_counter >= OBSERVE_FRAMES:
                    state = "RECORD"
                    buffer.clear()
                    has_motion = False
                    static_hold = 0
                    post_motion_hold = 0
                    stable_window.clear()

        elif state == "RECORD":
            if lm is None:
                state = "WAIT"
                continue

            buffer.append(lm)

            if motion > START_TH:
                has_motion = True
                static_hold = 0
                post_motion_hold = 0
            elif motion < END_TH:
                static_hold += 1
                if has_motion:
                    post_motion_hold += 1

            stable_window.append(motion < END_TH)
            if len(stable_window) > STABLE_WINDOW:
                stable_window.pop(0)

            # === SAVE ===
            if has_motion:
                if post_motion_hold >= POST_MOTION_HOLD and len(buffer) >= MIN_FRAMES:
                    return buffer
            else:
                if static_hold >= STATIC_HOLD_FRAMES and len(buffer) >= MIN_FRAMES:
                    return buffer

            if len(buffer) >= MAX_FRAMES:
                return buffer

        cv2.imshow("Teach Mode", disp)
        key = cv2.waitKey(1) & 0xFF

        label_state["text"] = keyboard_input(key, label_state["text"])
        if key == 13 and label_state["text"]:
            label_state["confirmed"] = label_state["text"]

        if key == 27:
            return None
