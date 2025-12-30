import cv2
import numpy as np

# ======================================================
# PARAMETER
# ======================================================
START_TH = 0.025
END_TH = 0.012

OBSERVE_FRAMES = 12

MIN_FRAMES = 20
MAX_FRAMES = 400

STATIC_HOLD_FRAMES = 40
POST_MOTION_HOLD = 30

STABLE_WINDOW = 20
MIN_HAND_AREA = 0.02

MAX_MISSING_FRAMES = 5   # ⬅ toleransi landmark hilang (penting untuk Z)

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

def draw_ui(frame, state, label_text):
    cv2.putText(frame, f"STATE: {state}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"LABEL: {label_text}",
                (10, frame.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

def keyboard_input(key, current):
    if 65 <= key <= 90 or 97 <= key <= 122:
        return current + chr(key).upper()
    if key == 8:
        return current[:-1]
    return current

# ======================================================
# AUTO RECORD (TRAJECTORY FIXED)
# ======================================================
def auto_record(cam, extractor, label_state):
    state = "WAIT"

    buffer = []
    observe_counter = 0

    static_hold = 0
    post_motion_hold = 0
    has_motion = False

    prev_lm = None
    missing_frames = 0

    while True:
        frame, _ = cam.read()
        if frame is None:
            return None

        lm_raw = extractor(frame)
        motion = motion_score(prev_lm, lm_raw)

        # ===== HANDLE LANDMARK HILANG =====
        if valid_hand(lm_raw):
            lm = lm_raw
            missing_frames = 0
        else:
            missing_frames += 1
            if missing_frames <= MAX_MISSING_FRAMES:
                lm = prev_lm
            else:
                lm = None

        prev_lm = lm
        disp = frame.copy()
        draw_keypoints(disp, lm)
        draw_ui(disp, state, label_state["text"])

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

                    # ⬅⬅⬅ INI KUNCI: AMBIL FRAME AWAL GERAK
                    buffer.append(prev_lm)
                    buffer.append(lm)

                    has_motion = True
                    static_hold = 0
                    post_motion_hold = 0

                elif observe_counter >= OBSERVE_FRAMES:
                    state = "RECORD"
                    buffer.clear()
                    buffer.append(lm)
                    has_motion = False
                    static_hold = 0
                    post_motion_hold = 0

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

            # ===== SAVE RULE =====
            if has_motion:
                if post_motion_hold >= POST_MOTION_HOLD and len(buffer) >= MIN_FRAMES:
                    # ⬅ CEK ADA MOTION NYATA
                    motions = [
                        motion_score(buffer[i-1], buffer[i])
                        for i in range(1, len(buffer))
                    ]
                    if np.max(motions) < START_TH:
                        continue  # TOLAK: ini pose statis
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
