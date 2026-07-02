"""
Milestone 3 — LangChain RAG pipeline.

Flow:
    anomaly dict  →  retrieve context (ChromaDB)
                  →  build prompt
                  →  call LLM (Groq | Ollama | Azure)
                  →  parse structured JSON recommendation

The LLM provider is selected via LLM_PROVIDER in .env — no code changes needed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import LLMProvider, settings

# ---------------------------------------------------------------------------
# System prompt — enforces "Expert BI Consultant" persona
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an Expert Business Intelligence Consultant for a Brazilian \
e-commerce marketplace (Olist).
Your task is to analyse a detected sales anomaly using the supporting business context \
provided, identify its root cause, and deliver concise, actionable strategic recommendations.

Respond ONLY with a valid JSON object using this exact schema — no markdown, no extra text:
{
  "diagnosis": "<one sentence root cause>",
  "impact_level": "<High | Medium | Low>",
  "recommendations": ["<action 1>", "<action 2>", "<action 3>"],
  "confidence": <float 0.0–1.0>
}"""


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def get_llm():
    """Return a configured LangChain LLM based on ``settings.LLM_PROVIDER``."""
    provider = settings.LLM_PROVIDER

    if provider == LLMProvider.GROQ:
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError("Run: pip install langchain-groq")
        return ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model_name=settings.GROQ_MODEL,
            temperature=0.3,
        )

    if provider == LLMProvider.OLLAMA:
        try:
            from langchain_community.llms import Ollama
        except ImportError:
            raise ImportError("Run: pip install langchain-community")
        return Ollama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=0.3,
        )

    if provider == LLMProvider.AZURE:
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError:
            raise ImportError("Run: pip install langchain-openai")
        return AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            temperature=0.3,
        )

    raise ValueError(f"Unknown LLM provider: {provider}")


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------

class BIRAGPipeline:
    """
    Orchestrates context retrieval and LLM generation for a single anomaly.

    Usage::

        pipeline = BIRAGPipeline()
        result = pipeline.generate_insight(anomaly_dict)
    """

    def __init__(self, vector_store=None):
        from rag_service.vector_store import BusinessVectorStore

        self._vs  = vector_store or BusinessVectorStore()
        self._llm = get_llm()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_insight(self, anomaly: dict) -> dict:
        """
        Full RAG cycle for one anomaly. Handles BOTH anomaly types:

        - Time-series (Prophet): keys ``ds``, ``y``, ``yhat``, ``deviation_pct``, ``direction``
        - Order-level (Isolation Forest): ``type == "order_anomaly"`` with
          ``order_id``, ``payment_value``, ``freight_value``, ``delivery_days`` …

        Returns:
            Dict with ``anomaly``, ``anomaly_description``,
            ``context_used``, ``recommendation``.
        """
        if anomaly.get("type") == "order_anomaly":
            anomaly_text = self._describe_order_anomaly(anomaly)
        else:
            anomaly_text = self._describe_anomaly(anomaly)
        context_docs = self._vs.query(anomaly_text, n_results=5)
        context_text = "\n".join(d["document"] for d in context_docs)

        user_message = (
            f"Business Anomaly Detected:\n{anomaly_text}\n\n"
            f"Relevant Business Context:\n{context_text}\n\n"
            "Provide your analysis and strategic recommendations."
        )

        raw = self._call_llm(user_message)
        recommendation = self._parse_json(raw)

        return {
            "anomaly":              anomaly,
            "anomaly_description":  anomaly_text,
            "context_used":         [d["document"] for d in context_docs],
            "recommendation":       recommendation,
        }

    def generate_insights_batch(self, anomalies: list[dict]) -> list[dict]:
        """Process multiple anomalies sequentially."""
        return [self.generate_insight(a) for a in anomalies]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _describe_anomaly(a: dict) -> str:
        direction = "spike" if a.get("direction") == "spike" else "drop"
        date_str  = str(a.get("ds", "unknown date"))[:10]
        return (
            f"Sales {direction} detected on {date_str}: "
            f"actual sales R${a.get('y', 0):,.2f} vs "
            f"expected R${a.get('yhat', 0):,.2f} "
            f"({a.get('deviation_pct', 0):+.1f}% deviation)."
        )

    @staticmethod
    def _describe_order_anomaly(a: dict) -> str:
        return (
            f"Operationally anomalous order detected (order {str(a.get('order_id', 'N/A'))[:12]}): "
            f"payment R${a.get('payment_value', 0):,.2f}, "
            f"freight R${a.get('freight_value', 0):,.2f}, "
            f"delivery {a.get('delivery_days', 0):.0f} days, "
            f"review score {a.get('review_score', 'N/A')}, "
            f"product weight {a.get('product_weight_g', 0):,.0f}g, "
            f"category '{a.get('product_category', 'N/A')}' "
            f"in state {a.get('customer_state', 'N/A')} "
            f"(seller in {a.get('seller_state', 'N/A')})."
        )

    def _call_llm(self, user_message: str) -> str:
        """Dispatch to chat or completion interface depending on provider."""
        from langchain_core.messages import HumanMessage, SystemMessage

        if settings.LLM_PROVIDER == LLMProvider.OLLAMA:
            # Plain completion interface
            return self._llm.invoke(f"{_SYSTEM_PROMPT}\n\n{user_message}")

        # Chat interface (Groq, Azure)
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        response = self._llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Extract JSON from the LLM response; degrade gracefully on parse error."""
        text = raw.strip()
        # Strip markdown code fences if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        try:
            return json.loads(text)
        except (json.JSONDecodeError, IndexError):
            return {
                "diagnosis":        text[:500],
                "impact_level":     "Unknown",
                "recommendations":  [],
                "confidence":       0.0,
                "parse_error":      True,
            }
