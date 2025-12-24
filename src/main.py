# src/main.py

from src import teach
from src import recognize

print("1 = TEACH")
print("2 = RECOGNIZE")

mode = input("Pilih mode: ").strip()

if mode == "1":
    teach.run()
elif mode == "2":
    recognize.run()
else:
    print("Mode tidak valid")
