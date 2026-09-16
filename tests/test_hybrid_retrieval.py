from documents.types import FinancialDocumentChunk
from retrieval.bm25 import BM25Retriever, tokenize
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.types import RetrievalResult

def c(i,text,ticker='MSTR'):
 return FinancialDocumentChunk(i,'d'+i,'Co',ticker,'0000000001','0000000001-26-000001','10-K','2026-01-01',None,'SEC','url','PART I ITEM 1A','Risk',0,text)
def r(i,rank): return RetrievalResult(rank,1,i,'d'+i,'Co','MSTR','10-K','2026-01-01',None,'S','S','t','SEC','url')
class Fake:
 def __init__(self,items): self.items=items
 def search(self,*args,top_k=5,**kwargs): return self.items[:top_k]
class Reverse(Reranker):
 def rerank(self,q,candidates,top_k): return [(x,10-rank) for rank,x in enumerate(reversed(candidates),1)][:top_k]

def test_bm25_tokenization_ranking_filters_and_provenance():
 assert tokenize('Bitcoin-CUSTODY!')==['bitcoin','custody']
 b=BM25Retriever([c('b','supply chain'),c('a','bitcoin custody risk'),c('z','bitcoin',ticker='AAPL')])
 out=b.search('Bitcoin custody',ticker='MSTR',top_k=5)
 assert [x.chunk_id for x in out]==['a','b'] and out[0].source_url=='url'
 assert b.search('bitcoin',ticker='NVDA')==[]

def test_rrf_and_hybrid_reranking_are_deterministic():
 fused=reciprocal_rank_fusion([[r('a',1),r('b',2)],[r('b',1),r('c',2)]])
 assert [x[0].chunk_id for x in fused]==['b','a','c']
 h=HybridRetriever(Fake([r('a',1),r('b',2)]),Fake([r('b',1),r('c',2)]),candidate_k=2)
 assert [x.chunk_id for x in h.search('q',top_k=3)]==['b','a','c']
 h=HybridRetriever(Fake([r('a',1),r('b',2)]),Fake([r('b',1),r('c',2)]),reranker=Reverse(),candidate_k=3)
 assert len(h.search('q',top_k=2))==2 and h.last_diagnostics[0]['chunk_id']=='b'
