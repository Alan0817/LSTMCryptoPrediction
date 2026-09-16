"""Small composition helper for the configured financial-analysis application."""
from agent.financial_agent import FinancialAnalysisAgent
from llm.client import LLMClient
from retrieval.factory import build_document_retriever
from tools.registry import ToolRegistry


def build_financial_analysis_agent(provider=None, retrieval_backend=None, web_search_provider=None):
    retriever = build_document_retriever(retrieval_backend)
    registry = ToolRegistry(document_retriever=retriever, web_search_provider=web_search_provider)
    return FinancialAnalysisAgent(LLMClient(provider=provider), registry)
