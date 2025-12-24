# src/prototype_db.py
import os
import numpy as np

BASE = "data/prototypes"

def save_prototype(label, seq):
    path = os.path.join(BASE, label)
    os.makedirs(path, exist_ok=True)
    idx = len(os.listdir(path)) + 1
    np.save(os.path.join(path, f"{label}_{idx}.npy"), seq)

def load_prototypes():
    protos = []
    if not os.path.exists(BASE):
        return protos

    for label in os.listdir(BASE):
        for f in os.listdir(os.path.join(BASE, label)):
            seq = np.load(os.path.join(BASE, label, f))
            protos.append((label, seq))
    return protos
