"""Small deterministic BM25 retriever for local SEC chunks."""
import math
import re
from collections import Counter
from datetime import date
from documents.types import FinancialDocumentChunk
from .types import RetrievalResult

def tokenize(text): return re.findall(r"[a-z0-9]+", text.lower())

class BM25Retriever:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks=tuple(sorted(chunks,key=lambda c:c.chunk_id)); self.k1=k1; self.b=b
        self.docs=[tokenize(c.text) for c in self.chunks]; self.tf=[Counter(d) for d in self.docs]
        self.lengths=[len(d) for d in self.docs]; self.avgdl=sum(self.lengths)/len(self.lengths) if self.lengths else 0
        df=Counter(t for d in self.docs for t in set(d)); n=len(self.docs)
        self.idf={t:math.log(1+(n-f+.5)/(f+.5)) for t,f in df.items()}
    def search(self,query,top_k=5,ticker=None,document_type=None,section=None,filing_date_from=None,filing_date_to=None):
        if not isinstance(query,str) or not query.strip(): raise ValueError('query must be a non-empty string.')
        if not isinstance(top_k,int) or isinstance(top_k,bool) or top_k<=0: raise ValueError('top_k must be a positive integer.')
        if document_type is not None and document_type not in {'10-K','10-Q'}: raise ValueError("document_type must be '10-K' or '10-Q'.")
        for value in (filing_date_from,filing_date_to):
            if value is not None: date.fromisoformat(value)
        terms=tokenize(query); candidates=[]
        for i,c in enumerate(self.chunks):
            if ticker and c.ticker!=ticker.upper() or document_type and c.document_type!=document_type or section and c.section!=section or filing_date_from and c.filing_date<filing_date_from or filing_date_to and c.filing_date>filing_date_to: continue
            score=sum(self.idf.get(t,0)*self.tf[i][t]*(self.k1+1)/(self.tf[i][t]+self.k1*(1-self.b+self.b*self.lengths[i]/self.avgdl)) for t in terms if self.tf[i][t])
            candidates.append((score,c))
        candidates.sort(key=lambda x:(-x[0],x[1].chunk_id))
        return [RetrievalResult(rank,float(score),c.chunk_id,c.document_id,c.ticker,c.company,c.document_type,c.filing_date,c.period_end,c.section,c.section_title,c.text,c.source,c.source_url) for rank,(score,c) in enumerate(candidates[:top_k],1)]
