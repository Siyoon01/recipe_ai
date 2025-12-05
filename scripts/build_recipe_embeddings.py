# AI/scripts/build_recipe_embeddings.py
import os
import ast
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models", "bge-m3")

RAW_CSV = os.path.join(DATA_DIR, "recipes_raw.csv")  # Won이 준비하는 원본 CSV
META_CSV = os.path.join(DATA_DIR, "recipe_meta.csv")
EMB_NPY = os.path.join(DATA_DIR, "recipe_embeddings.npy")

COL_ID = "recipeId"
COL_TITLE = "recipe_title_refined"
COL_CATEGORY = "type_2"
COL_HASHTAGS = "hashtags"
COL_DESC = "description_post"


def clean_hashtags(ht):
    if pd.isna(ht):
        return ""
    try:
        parsed = ast.literal_eval(ht) if isinstance(ht, str) else ht
        if isinstance(parsed, list):
            parsed = [str(x) for x in parsed if x]
            return ", ".join(parsed)
        return str(parsed)
    except Exception:
        return str(ht)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("CSV 로드:", RAW_CSV)
    df = pd.read_csv(RAW_CSV, encoding="utf-8-sig")
    print("shape:", df.shape)
    # recipeId 부여 (1부터)
    df[COL_ID] = range(1, len(df) + 1)

    # 최소 컬럼 체크
    for col in [COL_TITLE, COL_CATEGORY, COL_HASHTAGS, COL_DESC]:
        if col not in df.columns:
            raise RuntimeError(f"필수 컬럼 누락: {col}")

    df["hashtags_clean"] = df[COL_HASHTAGS].apply(clean_hashtags)

    # 임베딩용 텍스트 생성
    df["embedding_text"] = (
        "레시피 이름: " + df[COL_TITLE].fillna("") + "\n"
        "카테고리: " + df[COL_CATEGORY].fillna("") + "\n"
        "해시태그: " + df["hashtags_clean"].fillna("") + "\n"
        "설명: " + df[COL_DESC].fillna("")
    )

    # 메타만 따로 저장
    meta_cols = [
        COL_ID,
        COL_TITLE,
        COL_CATEGORY,
        "hashtags_clean",
        COL_DESC,
        "embedding_text",
    ]
    df_meta = df[meta_cols].copy()
    df_meta.to_csv(META_CSV, index=False, encoding="utf-8-sig")
    print("메타 저장 완료:", META_CSV)

    # 모델 로드
    print("모델 로드:", MODEL_DIR)
    model = SentenceTransformer(MODEL_DIR)

    texts = df_meta["embedding_text"].tolist()
    print("임베딩 계산 중... (N =", len(texts), ")")

    embs = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")

    print("임베딩 shape:", embs.shape)
    np.save(EMB_NPY, embs)
    print("임베딩 저장 완료:", EMB_NPY)


if __name__ == "__main__":
    main()
