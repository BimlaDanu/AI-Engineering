"""
Application core: framework-independent pipeline logic used by the UI layer.

Keeps orchestration separate from UI code for easier testing and allows future
replacement of the linear RAG pipeline with a LangGraph-based workflow.
"""
