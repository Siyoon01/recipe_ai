# AI/recsys_scorer.py
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import util

from recsys_config import W_SIM, W_COVER_OWNED, W_COVER_SELECTED


def _compute_selected_coverages(
    candidate_ingre_ids: List[int],
    selected_ids: List[int],
) -> Tuple[float, float]:
    """
    coverage_recipe   = (후보 레시피 재료 ∩ 소진 희망 재료) / 후보 레시피 재료 개수
    coverage_selected = (후보 레시피 재료 ∩ 소진 희망 재료) / 소진 희망 재료 개수
    """
    cand = set(int(x) for x in (candidate_ingre_ids or []))
    selected = set(int(x) for x in (selected_ids or []))

    if not cand or not selected:
        # 레시피 재료가 없거나, 소진 희망 재료가 없으면 커버율은 0
        return 0.0, 0.0

    inter = cand & selected

    # 레시피 입장에서의 커버율: 레시피 재료 중 main(소진 희망) 재료 비율
    coverage_recipe = len(inter) / len(cand) if cand else 0.0

    # 사용자가 선택한 재료 입장에서의 커버율: 선택 재료 중 이 레시피가 커버하는 비율
    coverage_selected = len(inter) / len(selected) if selected else 0.0

    return float(coverage_recipe), float(coverage_selected)


def score_candidates(
    query_emb: np.ndarray,
    candidates: List[Dict[str, Any]],
    recipe_embs: np.ndarray,
    recipe_index_map: Dict[int, int],
    ownedIngredientIds: List[int],      # keeping for interface reason (check)
    selectedIngredientIds: List[int],
    requireMain: bool,
) -> List[Dict[str, float]]:

    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)

    recs: List[Dict[str, float]] = []

    for c in candidates:
        recipe_id = int(c.get("recipeId"))
        cand_ingre_ids = c.get("ingredientIds") or []

        # recipeId(서버는 1부터) -> 임베딩 인덱스(0부터) 매핑
        idx = recipe_index_map.get(recipe_id)
        if idx is None:
            # 임베딩이 없는 레시피는 스킵
            continue

        doc_emb = recipe_embs[idx].reshape(1, -1)
        sim = float(util.cos_sim(query_emb, doc_emb)[0][0])

        # --- 커버율: 전부 selectedIngredientIds 기준 ---
        coverage_recipe, coverage_selected = _compute_selected_coverages(
            cand_ingre_ids,
            selectedIngredientIds,
        )

        # 기본 점수 공식
        # W_COVER_OWNED  : 레시피 내에서 main 재료 비율(coverage_recipe) 가중치
        # W_COVER_SELECTED: 사용자가 선택한 재료가 얼마나 커버되는지(coverage_selected) 가중치
        base_score = (
            W_SIM * sim
            + W_COVER_OWNED * coverage_recipe
            + W_COVER_SELECTED * coverage_selected
        )

        # requireMain = true 이고 selected가 비어있지 않으면
        # 후보 레시피에 selected 중 하나도 안 들어있으면 제외
        if requireMain and selectedIngredientIds:
            cand_set = set(int(x) for x in cand_ingre_ids)
            sel_set = set(int(x) for x in selectedIngredientIds)
            if len(cand_set & sel_set) == 0:
                continue

        recs.append(
            {
                "recipeId": recipe_id,
                "score": round(float(base_score), 4),
                "sim": round(sim, 4),
                "coverage_recipe": round(coverage_recipe, 4),
                "coverage_selected": round(coverage_selected, 4),
            }
        )

    # 점수 기준 내림차순 정렬
    recs.sort(key=lambda x: x["score"], reverse=True)
    return recs
