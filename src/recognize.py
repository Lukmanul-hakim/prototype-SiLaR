# src/recognize.py
import cv2
import numpy as np
import joblib
import time
import warnings

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from src.capture import Camera
from src.feature import extract_features

# =========================
# OPTIONAL: suppress noisy warnings (protobuf/mediapipe)
# =========================
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", message=".*Feedback manager.*")
warnings.filterwarnings("ignore")

# =========================
# PATH MODEL
# =========================
MODEL_PATH = "data/models/gesture_model.joblib"

def downsample_seq(seq_3d: np.ndarray, target_len: int = 12) -> np.ndarray:
    T = seq_3d.shape[0]
    if T <= target_len:
        return seq_3d
    idx = np.linspace(0, T - 1, target_len).astype(int)
    return seq_3d[idx]

# =========================
# GESTURE RECOGNIZER (DTW) - optimized
# =========================
class GestureRecognizer:
    def __init__(self, model_path=MODEL_PATH):
        payload = joblib.load(model_path)
        raw_dataset = payload["dataset"]

        self.protos = []      # (label, flat_seq)
        self.proto_mean = []  # mean vector per proto

        for label, samples in raw_dataset.items():
            for seq in samples:
                arr = np.asarray(seq, dtype=np.float32).reshape(seq.shape[0], -1)  # (T,84)
                self.protos.append((label, arr))
                self.proto_mean.append(arr.mean(axis=0))

        self.proto_mean = np.vstack(self.proto_mean).astype(np.float32)  # (N,84)

    @staticmethod
    def _distance_flat(s1_2d: np.ndarray, s2_2d: np.ndarray) -> float:
        dist, _ = fastdtw(s1_2d, s2_2d, dist=euclidean)
        return dist / max(len(s1_2d), len(s2_2d))

    def predict(self, seq_3d: np.ndarray, topk: int = 30):
        s1 = np.asarray(seq_3d, dtype=np.float32).reshape(seq_3d.shape[0], -1)  # (T,84)

        # ---- Stage 1: cheap filter by mean-vector distance
        q = s1.mean(axis=0)  # (84,)
        d0 = np.linalg.norm(self.proto_mean - q[None, :], axis=1)  # (N,)
        idx = np.argpartition(d0, min(topk, len(d0)-1))[:topk]

        # ---- Stage 2: DTW only on topk
        best_label = None
        best_dist = float("inf")
        for i in idx:
            label, s2 = self.protos[i]
            d = self._distance_flat(s1, s2)
            if d < best_dist:
                best_dist = d
                best_label = label

        return best_label, best_dist

# =========================
# REALTIME RECOGNIZE LOOP (1 window)
# =========================
def run():
    print("\n=== RECOGNIZE MODE (REALTIME - 1 WINDOW) ===")
    print("ESC = keluar | SPACE = commit kata | BACKSPACE = hapus kata\n")

    cam = Camera()
    recognizer = GestureRecognizer()

    # =========================
    # TUNING (ubah sesuai device)
    # =========================
    WINDOW_SIZE = 10          # 18-25 biasanya lebih ringan daripada 30
    PREDICT_HZ = 2            # prediksi per detik (3-6 recommended)
    UNKNOWN_TH = 0.55         # kalau dist > ini -> UNKNOWN (kalibrasi)
    STREAK_N = 3              # harus muncul N kali berturut untuk "fix" label

    # Buffer untuk sliding window
    seq_buf = []

    # Output text
    word = ""
    sentence = ""

    # Stabilizer state
    streak_label = None
    streak_count = 0

    # Last shown
    last_label = "-"
    last_dist = 0.0

    # Timing for throttling prediction
    predict_interval = 1.0 / max(PREDICT_HZ, 1)
    last_predict_time = 0.0

    # FPS tracker
    t_prev = time.time()
    fps = 0.0

    try:
        while True:
            frame, _ = cam.read()
            if frame is None:
                break

            # FPS update
            t_now = time.time()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Extract landmark features
            lm = extract_features(frame)  # expected (42,2) or None
            if lm is not None:
                seq_buf.append(lm)
                if len(seq_buf) > WINDOW_SIZE:
                    seq_buf.pop(0)

            # Predict (time-throttled)
            if len(seq_buf) == WINDOW_SIZE and (t_now - last_predict_time) >= predict_interval:
                last_predict_time = t_now

                seq = np.array(seq_buf, dtype=np.float32)  # (T,42,2)
                # label, dist = recognizer.predict(seq)
                label, dist = recognizer.predict(seq, topk=30)

                # Unknown reject
                if dist is None or dist > UNKNOWN_TH:
                    last_label = "UNKNOWN"
                    last_dist = float(dist) if dist is not None else 0.0
                    streak_label = None
                    streak_count = 0
                    # optional: kosongkan word saat unknown
                    # word = ""
                else:
                    # Stabilizer: require STREAK_N consecutive same labels
                    if label == streak_label:
                        streak_count += 1
                    else:
                        streak_label = label
                        streak_count = 1

                    if streak_count >= STREAK_N:
                        last_label = label
                        last_dist = float(dist)
                        word = label  # label dianggap "kata", bukan ditempel terus

            overlay = frame.copy()

            cv2.putText(
                overlay,
                f"Detected: {last_label} (dist={last_dist:.2f})",
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
                f"FPS: {fps:.1f} | Win:{WINDOW_SIZE} | Hz:{PREDICT_HZ} | TH:{UNKNOWN_TH}",
                (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
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
                if word and word != "UNKNOWN":
                    sentence += word + " "
                word = ""
                streak_label = None
                streak_count = 0
            elif key == 8:  # BACKSPACE
                word = ""
                streak_label = None
                streak_count = 0

    finally:
        cam.release()
        cv2.destroyAllWindows()
