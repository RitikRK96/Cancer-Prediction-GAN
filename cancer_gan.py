import os

os.system("python scripts/preprocess.py")
os.system("python scripts/train.py")
os.system("python scripts/generate.py")
os.system("python scripts/evaluate.py")

print("✅ End-to-end execution complete!")
