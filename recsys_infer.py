# AI/recsys_infer.py
import sys
import json
import traceback

from recsys_loader import get_recipe_data, encode_query
from recsys_scorer import score_candidates


def read_stdin_json():
    raw = sys.stdin.read()
    if not raw:
        raise ValueError("stdin으로부터 입력이 없습니다.")
    return json.loads(raw)


def make_error_response(message: str, status_code: int = 500):
    return {
        "success": False,
        "message": message,
        "statusCode": status_code,
        "resultCode": status_code,   # 필요하면 서버에서 사용
        "recommendations": None,
    }


def main():
    try:
        req = read_stdin_json()

        user_id = req.get("userId")
        owned_ids = req.get("ownedIngredientIds") or []
        query = req.get("query") or {}
        query_text = query.get("queryText", "")
        selected_ids = query.get("selectedIngredientIds") or []
        require_main = bool(req.get("requireMain", False))
        candidates = req.get("candidates") or []

        # 필수 파라미터 체크
        if user_id is None:
            resp = make_error_response("필수 파라미터(userId)가 누락되었습니다.", 400)
            sys.stdout.write(json.dumps(resp, ensure_ascii=False))
            return

        if not isinstance(candidates, list) or len(candidates) == 0:
            resp = make_error_response("필수 파라미터(candidates)가 누락되었습니다.", 400)
            sys.stdout.write(json.dumps(resp, ensure_ascii=False))
            return

        # 데이터/모델 로드
        recipe_meta, recipe_embs, recipe_index_map = get_recipe_data()
        # recipe_meta는 지금은 안 쓰지만 추후 feature 확장용으로 남겨둠
        _ = recipe_meta

        # 쿼리 임베딩
        query_emb = encode_query(query_text)

        # 점수 계산
        scored = score_candidates(
            query_emb=query_emb,
            candidates=candidates,
            recipe_embs=recipe_embs,
            recipe_index_map=recipe_index_map,
            ownedIngredientIds=owned_ids,
            selectedIngredientIds=selected_ids,
            requireMain=require_main,
        )

        # 상위 N개만 응답
        TOP_K = 20
        recommendations = [
            {"recipeId": r["recipeId"], "score": r["score"]}
            for r in scored[:TOP_K]
        ]

        resp = {
            "success": True,
            "message": "최적 레시피가 성공적으로 추출되었습니다.",
            "statusCode": 200,
            "resultCode": 200,
            "recommendations": recommendations,
        }

        sys.stdout.write(json.dumps(resp, ensure_ascii=False))

    except json.JSONDecodeError:
        resp = make_error_response("요청 JSON을 파싱할 수 없습니다.", 400)
        sys.stdout.write(json.dumps(resp, ensure_ascii=False))
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        resp = make_error_response(
            f"레시피 예측 중 AI 모델에서 오류가 발생했습니다: {str(e)}", 500
        )
        sys.stdout.write(json.dumps(resp, ensure_ascii=False))


if __name__ == "__main__":
    main()
