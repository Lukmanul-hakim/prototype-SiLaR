# src/feature.py
import numpy as np
import mediapipe as mp
import cv2

# =========================
# MEDIAPIPE HANDS
# =========================
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def extract_features(frame):
    """
    Extract hand landmarks from frame.
    Aman untuk frame BGR maupun grayscale.
    Return: (42, 2) landmark atau None
    """
    if frame is None:
        return None

    # ===== PASTIKAN FRAME 3 CHANNEL =====
    if len(frame.shape) == 2:
        # grayscale → BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mp_hands.process(rgb)

    if not res.multi_hand_landmarks or not res.multi_handedness:
        return None

    left = np.zeros((21, 2), dtype=np.float32)
    right = np.zeros((21, 2), dtype=np.float32)

    for lm, info in zip(res.multi_hand_landmarks, res.multi_handedness):
        label = info.classification[0].label  # 'Left' / 'Right'
        pts = np.array([[p.x, p.y] for p in lm.landmark], dtype=np.float32)

        if label == "Left":
            left = pts
        elif label == "Right":
            right = pts

    return np.vstack([left, right])  # (42, 2)
