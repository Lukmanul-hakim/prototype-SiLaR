# src/recognize_realtime.py
import cv2
import numpy as np

from src.capture import Camera
from src.teach import extract_landmarks
from src.segment import auto_record
from src.recognize import GestureRecognizer
from src.text_buffer import TextBuffer


def main():
    cam = Camera()
    recognizer = GestureRecognizer()
    text_buf = TextBuffer(cooldown=1.0)

    print("=== REALTIME RECOGNITION ===")
    print("SPACE = finalize word | BACKSPACE = clear | Q = quit")

    while True:
        seq = auto_record(cam, extract_landmarks, {"text": "", "confirmed": None})
        if seq is None:
            break

        seq = np.stack(seq, axis=0)
        label, dist = recognizer.predict(seq)

        if label:
            accepted = text_buf.add(label)
            if accepted:
                print(f"✔ {label} (dist={dist:.2f})")

        while True:
            frame, _ = cam.read()
            if frame is None:
                break

            cv2.putText(frame, f"WORD: {text_buf.word}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, f"SENTENCE: {text_buf.sentence}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

            cv2.imshow("Sign Language Realtime", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                text_buf.space()
            elif key == 8:
                text_buf.clear()
            elif key == ord('q'):
                cam.release()
                cv2.destroyAllWindows()
                return
            else:
                break


if __name__ == "__main__":
    main()
