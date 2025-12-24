# src/teach.py
import numpy as np
import mediapipe as mp

from src.capture import Camera
from src.segment import auto_record
from src.prototype_db import save_prototype

# =========================
# MEDIAPIPE (2 HANDS, AS-IS)
# =========================
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def extract_landmarks(frame):
    """
    NOTE:
    - Frame sudah mirror untuk tampilan (di capture.py)
    - Landmark JANGAN di-mirror lagi
    - Handedness dibiarkan apa adanya
    """
    rgb = frame[:, :, ::-1]
    res = mp_hands.process(rgb)

    left = np.zeros((21, 2), dtype=np.float32)
    right = np.zeros((21, 2), dtype=np.float32)

    if not res.multi_hand_landmarks or not res.multi_handedness:
        return None

    for lm, info in zip(res.multi_hand_landmarks, res.multi_handedness):
        label = info.classification[0].label  # 'Left' / 'Right'
        pts = np.array([[p.x, p.y] for p in lm.landmark], dtype=np.float32)

        if label == "Left":
            left = pts
        elif label == "Right":
            right = pts

    return np.vstack([left, right])  # (42, 2)

# =========================
# TEACH MODE
# =========================
def run():
    cam = Camera()
    print("=== TEACH MODE (NO DOUBLE MIRROR) ===")
    print("Ketik label di layar | ENTER konfirmasi | Q keluar")

    label_state = {"text": "", "confirmed": None}

    while True:
        seq = auto_record(cam, extract_landmarks, label_state)

        if seq is None:
            break

        if label_state["confirmed"] is None:
            continue

        seq = np.stack(seq, axis=0)
        save_prototype(label_state["confirmed"], seq)

        print(f"✓ Gesture '{label_state['confirmed']}' tersimpan ({len(seq)} frames)")

    cam.release()
