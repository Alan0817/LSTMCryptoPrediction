"""Deterministic reciprocal-rank fusion."""


def reciprocal_rank_fusion(rankings, rrf_k=60):
    scores = {}
    by_id = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            by_id[item.chunk_id] = item
            scores[item.chunk_id] = scores.get(item.chunk_id, 0) + 1 / (rrf_k + rank)

    ordered_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [(by_id[chunk_id], scores[chunk_id]) for chunk_id in ordered_ids]
