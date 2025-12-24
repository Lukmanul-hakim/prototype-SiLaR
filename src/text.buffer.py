# src/text_buffer.py
import time


class TextBuffer:
    def __init__(self, cooldown=1.0):
        self.word = ""
        self.sentence = ""
        self.last_emit_time = 0.0
        self.cooldown = cooldown

    def add(self, token):
        now = time.time()
        if now - self.last_emit_time < self.cooldown:
            return False

        self.word += token
        self.last_emit_time = now
        return True

    def space(self):
        if self.word:
            self.sentence += self.word + " "
            self.word = ""

    def clear(self):
        self.word = ""
        self.sentence = ""
