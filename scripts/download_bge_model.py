# AI/scripts/download_bge_model.py
import os
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "bge-m3")
MODEL_NAME = "BAAI/bge-m3"

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"downloading model: ({MODEL_NAME} -> {MODEL_DIR})")
model = SentenceTransformer(MODEL_NAME)
model.save(MODEL_DIR)
print("completed")
