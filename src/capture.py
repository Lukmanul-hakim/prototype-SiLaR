# src/capture.py
import cv2
import time

class Camera:
    def __init__(self, index=1, width=640, height=480):
        # =========================
        # PAKSA BACKEND (PENTING)
        # =========================
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError("Kamera tidak bisa dibuka")

        # =========================
        # SET RESOLUSI AMAN
        # =========================
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # =========================
        # BUFFER FLUSH (ANTI NOISE)
        # =========================
        for _ in range(5):
            self.cap.read()

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        # =========================
        # FIX MIRROR (UX)
        # =========================
        frame = cv2.flip(frame, 1)

        return frame, time.time()

    def release(self):
        if self.cap:
            self.cap.release()
