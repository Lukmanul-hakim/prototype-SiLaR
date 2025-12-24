# src/recognize.py
import cv2
import numpy as np
import joblib
import time

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from src.capture import Camera
from src.feature import extract_features
from src.segment import auto_record

# =========================
# PATH MODEL
# =========================
MODEL_PATH = "data/models/gesture_model.joblib"


# =========================
# GESTURE RECOGNIZER (DTW)
# =========================
class GestureRecognizer:
    def __init__(self, model_path=MODEL_PATH):
        payload = joblib.load(model_path)
        self.dataset = payload["dataset"]
        self.labels = payload["labels"]

    def _distance(self, seq1, seq2):
        s1 = seq1.reshape(seq1.shape[0], -1)
        s2 = seq2.reshape(seq2.shape[0], -1)
        dist, _ = fastdtw(s1, s2, dist=euclidean)
        return dist / max(len(s1), len(s2))

    def predict(self, seq):
        best_label = None
        best_dist = float("inf")

        for label, samples in self.dataset.items():
            for proto in samples:
                d = self._distance(seq, proto)
                if d < best_dist:
                    best_dist = d
                    best_label = label

        return best_label, best_dist


# =========================
# REALTIME RECOGNIZE LOOP
# =========================
def run():
    print("\n=== RECOGNIZE MODE (REALTIME) ===")
    print("Gerakkan isyarat di depan kamera")
    print("ESC = keluar | SPACE = commit kata | BACKSPACE = hapus kata\n")

    cam = Camera()
    recognizer = GestureRecognizer()

    last_emit_time = 0
    EMIT_DELAY = 0.8  # detik

    word = ""
    sentence = ""

    try:
        while True:
            # Blocking sampai 1 gesture selesai
            seq = auto_record(cam, extract_features, {"text": "", "confirmed": None})
            if seq is None:
                break

            label, dist = recognizer.predict(np.array(seq))

            now = time.time()
            if now - last_emit_time > EMIT_DELAY:
                word += label
                last_emit_time = now

            frame, _ = cam.read()
            if frame is None:
                break

            overlay = frame.copy()

            cv2.putText(
                overlay,
                f"Detected: {label} (dist={dist:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                overlay,
                f"Word: {word}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                overlay,
                f"Sentence: {sentence}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 200, 200),
                2,
            )
            cv2.putText(
                overlay,
                "[SPACE]=commit | [BACKSPACE]=clear | [ESC]=quit",
                (10, overlay.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )

            cv2.imshow("Sign Language Realtime", overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                sentence += word + " "
                word = ""
            elif key == 8:  # BACKSPACE
                word = ""

    finally:
        cam.release()
        cv2.destroyAllWindows()
