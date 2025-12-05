# AI/recsys_scorer.py
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import util

from recsys_config import W_SIM, W_COVER_OWNED, W_COVER_SELECTED


def _compute_coverage(
    candidate_ingre_ids: List[int],
    owned_ids: List[int],
    selected_ids: List[int],
) -> Tuple[float, float]:
    """
    coverage_owned    = (후보 재료 ∩ 보유 재료) / 후보 재료 개수
    coverage_selected = (후보 재료 ∩ 소진 희망 재료) / 소진 희망 재료 개수
    """
    cand = set(int(x) for x in (candidate_ingre_ids or []))
    owned = set(int(x) for x in (owned_ids or []))
    selected = set(int(x) for x in (selected_ids or []))

    if not cand:
        return 0.0, 0.0

    inter_owned = cand & owned
    coverage_owned = len(inter_owned) / len(cand)

    if not selected:
        coverage_selected = 0.0
    else:
        inter_selected = cand & selected
        coverage_selected = len(inter_selected) / len(selected)

    return float(coverage_owned), float(coverage_selected)


def score_candidates(
    query_emb: np.ndarray,
    candidates: List[Dict[str, Any]],
    recipe_embs: np.ndarray,
    recipe_index_map: Dict[int, int],
    ownedIngredientIds: List[int],
    selectedIngredientIds: List[int],
    requireMain: bool,
) -> List[Dict[str, float]]:

    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)

    recs = []

    for c in candidates:
        recipe_id = int(c.get("recipeId"))
        cand_ingre_ids = c.get("ingredientIds") or []

        idx = recipe_index_map.get(recipe_id)
        if idx is None:
            # 임베딩이 없는 레시피는 스킵
            continue

        doc_emb = recipe_embs[idx].reshape(1, -1)
        sim = float(util.cos_sim(query_emb, doc_emb)[0][0])

        coverage_owned, coverage_selected = _compute_coverage(
            cand_ingre_ids,
            ownedIngredientIds,
            selectedIngredientIds,
        )

        # 기본 점수 공식
        base_score = (
            W_SIM * sim
            + W_COVER_OWNED * coverage_owned
            + W_COVER_SELECTED * coverage_selected
        )

        # requireMain = true 이고 selected가 있는데
        # 후보 레시피에 하나도 안 들어있으면 제외
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
                "coverage_owned": round(coverage_owned, 4),
                "coverage_selected": round(coverage_selected, 4),
            }
        )

    recs.sort(key=lambda x: x["score"], reverse=True)
    return recs
