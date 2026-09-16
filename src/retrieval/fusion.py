"""Deterministic reciprocal-rank fusion."""
def reciprocal_rank_fusion(rankings, rrf_k=60):
    scores={}; by_id={}
    for ranking in rankings:
        for rank,item in enumerate(ranking,1):
            by_id[item.chunk_id]=item; scores[item.chunk_id]=scores.get(item.chunk_id,0)+1/(rrf_k+rank)
    return [(by_id[i],scores[i]) for i in sorted(scores,key=lambda i:(-scores[i],i))]
