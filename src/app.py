"""Small composition helper for the configured financial-analysis application."""
from agent.financial_agent import FinancialAnalysisAgent
from llm.client import LLMClient
from retrieval.factory import build_document_retriever
from tools.registry import ToolRegistry
def build_financial_analysis_agent(provider=None, retrieval_backend=None):
    retriever=build_document_retriever(retrieval_backend)
    return FinancialAnalysisAgent(LLMClient(provider=provider),ToolRegistry(document_retriever=retriever))
