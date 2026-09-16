import pytest
from documents.types import FinancialDocumentChunk
from retrieval.factory import DEFAULT_BACKEND, build_document_retriever
from retrieval.planning import document_plan_metrics

def chunk(): return FinancialDocumentChunk('c','d','Co','MSTR','0000000001','0000000001-26-000001','10-K','2026-01-01',None,'SEC','url','S','S',0,'bitcoin custody')
def test_bm25_factory_is_lazy_for_dense_models():
 r=build_document_retriever('bm25',chunks=[chunk()])
 assert r.search('bitcoin')[0].chunk_id=='c' and DEFAULT_BACKEND=='hybrid'
def test_invalid_backend_and_empty_corpus_fail():
 with pytest.raises(ValueError): build_document_retriever('nope',chunks=[chunk()])
 with pytest.raises(RuntimeError): build_document_retriever('bm25',chunks=[])
def test_planning_metrics_detect_refinement_repeat_and_budget():
 base={'event':'tool_requested','name':'search_financial_documents'}
 assert document_plan_metrics([base|{'arguments':{'query':'custody'}}])['one_shot']
 refined=document_plan_metrics([base|{'arguments':{'query':'custody'}},base|{'arguments':{'query':'private keys','ticker':'MSTR'}}])
 assert refined['materially_changed_query'] and refined['materially_changed_filters'] and not refined['exceeded_document_budget']
 repeated=document_plan_metrics([base|{'arguments':{'query':'x'}},base|{'arguments':{'query':'x'}},base|{'arguments':{'query':'x'}}])
 assert repeated['identical_repeated_retrieval'] and repeated['exceeded_document_budget']
