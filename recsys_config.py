# AI/recsys_config.py
import os

# 이 파일(recsys_config.py)이 있는 폴더가 AI 폴더라고 가정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# bge-m3 SentenceTransformer 저장 경로
MODEL_PATH = os.environ.get(
    "RECSYS_MODEL_PATH",
    os.path.join(BASE_DIR, "models", "bge-m3")
)

# 레시피 메타 CSV
RECIPE_META_PATH = os.environ.get(
    "RECSYS_META_PATH",
    os.path.join(BASE_DIR, "data", "recipe_meta.csv")
)

# 레시피 임베딩 NPY
RECIPE_EMB_PATH = os.environ.get(
    "RECSYS_EMB_PATH",
    os.path.join(BASE_DIR, "data", "recipe_embeddings.npy")
)

# 점수 가중치 (환경 변수로 오버라이드 가능)
W_SIM = float(os.environ.get("RECSYS_W_SIM", 0.6))            # 임베딩 유사도
W_COVER_OWNED = float(os.environ.get("RECSYS_W_OWNED", 0.2))  # 보유 재료 커버율
W_COVER_SELECTED = float(os.environ.get("RECSYS_W_SELECTED", 0.2))  # 소진 희망 재료 커버율
