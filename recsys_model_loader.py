# AI/recsys_loader.py
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from recsys_config import MODEL_PATH, RECIPE_META_PATH, RECIPE_EMB_PATH

_model = None
_recipe_meta = None
_recipe_embs = None
_recipe_index_map = None  # recipeId -> index


def get_model():
    """SentenceTransformer(bge-m3) 전역 1회 로드"""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_PATH)
    return _model


def get_recipe_data():
    """
    recipe_meta(DataFrame),
    recipe_embs(np.ndarray),
    recipe_index_map(dict: recipeId -> row index) 반환
    """
    global _recipe_meta, _recipe_embs, _recipe_index_map

    if _recipe_meta is None or _recipe_embs is None or _recipe_index_map is None:
        df = pd.read_csv(RECIPE_META_PATH, encoding="utf-8-sig")
        embs = np.load(RECIPE_EMB_PATH)

        if len(df) != embs.shape[0]:
            raise RuntimeError(
                f"recipe_meta row 수({len(df)})와 임베딩 수({embs.shape[0]})가 다릅니다."
            )

        if "recipeId" not in df.columns:
            raise RuntimeError("recipe_meta.csv에 'recipeId' 컬럼이 필요합니다.")

        index_map = {
            int(rid): idx for idx, rid in enumerate(df["recipeId"].tolist())
        }

        _recipe_meta = df
        _recipe_embs = embs
        _recipe_index_map = index_map

    return _recipe_meta, _recipe_embs, _recipe_index_map


def encode_query(query_text: str) -> np.ndarray:
    """
    검색 쿼리 텍스트를 임베딩으로 변환.
    build_recipe_embeddings에서 사용한 embedding_text 포맷과 최대한 비슷하게 감싼다.
    """
    model = get_model()

    full_query = (
        "레시피 이름 또는 설명: " + (query_text or "") + "\n"
        "카테고리: \n"
        "주요 재료: \n"
        "해시태그: "
    )

    emb = model.encode([full_query], normalize_embeddings=True)[0]
    return emb.astype("float32")
