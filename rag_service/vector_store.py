"""
Milestone 3 — ChromaDB vector store for business intelligence context.

Indexes narrative text summaries generated from the Olist master table so
the RAG pipeline can retrieve relevant context when an anomaly is flagged.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import settings


class BusinessVectorStore:
    """
    Persistent ChromaDB collection of business-data narrative summaries.

    Documents stored:
        - Regional sales summaries (per Brazilian state)
        - Product category performance
        - Monthly revenue trends
    """

    COLLECTION_NAME = "olist_business_context"

    def __init__(self):
        import chromadb
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )

        settings.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(settings.VECTOR_DB_DIR))
        self._ef = SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBEDDING_MODEL
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_business_data(self, master_df: pd.DataFrame) -> int:
        """
        Generate narrative text summaries from *master_df* and upsert into ChromaDB.

        Returns the number of documents indexed.
        """
        docs, ids, metas = [], [], []

        # 1. Regional summaries (per state)
        self._add_regional_summaries(master_df, docs, ids, metas)

        # 2. Product category performance
        self._add_category_summaries(master_df, docs, ids, metas)

        # 3. Monthly revenue trends
        self._add_monthly_summaries(master_df, docs, ids, metas)

        if docs:
            self._collection.upsert(documents=docs, ids=ids, metadatas=metas)

        print(f"Vector store: indexed {len(docs)} business context documents.")
        return len(docs)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(self, query_text: str, n_results: int = 5) -> list[dict]:
        """
        Return the *n_results* most semantically relevant documents.

        Each result is ``{document, metadata, distance}``.
        """
        count = self._collection.count()
        if count == 0:
            return []

        results = self._collection.query(
            query_texts=[query_text],
            n_results=min(n_results, count),
        )

        output = []
        if results and results.get("documents"):
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                output.append({"document": doc, "metadata": meta, "distance": dist})
        return output

    @property
    def document_count(self) -> int:
        return self._collection.count()

    # ------------------------------------------------------------------
    # Private summary builders
    # ------------------------------------------------------------------

    def _add_regional_summaries(
        self,
        df: pd.DataFrame,
        docs: list, ids: list, metas: list,
    ) -> None:
        needed = {"customer_state", "payment_value", "order_id", "review_score"}
        if not needed.issubset(df.columns):
            return

        regional = (
            df.groupby("customer_state")
            .agg(
                total_sales=("payment_value", "sum"),
                total_orders=("order_id", "nunique"),
                avg_review=("review_score", "mean"),
            )
            .reset_index()
        )
        for _, row in regional.iterrows():
            text = (
                f"State {row['customer_state']}: "
                f"total sales R${row['total_sales']:,.2f}, "
                f"{row['total_orders']:,} orders, "
                f"average review score {row['avg_review']:.2f}."
            )
            docs.append(text)
            ids.append(f"region_{row['customer_state']}")
            metas.append({"type": "regional", "state": str(row["customer_state"])})

    def _add_category_summaries(
        self,
        df: pd.DataFrame,
        docs: list, ids: list, metas: list,
    ) -> None:
        if "product_category" not in df.columns:
            return

        cat = (
            df.groupby("product_category")
            .agg(
                total_orders=("order_id", "nunique"),
                avg_price=("price", "mean"),
                avg_review=("review_score", "mean"),
            )
            .reset_index()
            .sort_values("total_orders", ascending=False)
            .head(30)
        )
        for _, row in cat.iterrows():
            cat_slug = str(row["product_category"]).replace(" ", "_").replace("/", "_")
            text = (
                f"Product category '{row['product_category']}': "
                f"{row['total_orders']:,} orders, "
                f"average price R${row['avg_price']:.2f}, "
                f"average review score {row['avg_review']:.2f}."
            )
            docs.append(text)
            ids.append(f"category_{cat_slug}")
            metas.append({"type": "category", "category": str(row["product_category"])})

    def _add_monthly_summaries(
        self,
        df: pd.DataFrame,
        docs: list, ids: list, metas: list,
    ) -> None:
        if "order_purchase_timestamp" not in df.columns:
            return

        tmp = df.copy()
        tmp["month"] = (
            pd.to_datetime(tmp["order_purchase_timestamp"])
            .dt.to_period("M")
            .astype(str)
        )
        monthly = (
            tmp.groupby("month")
            .agg(
                total_sales=("payment_value", "sum"),
                total_orders=("order_id", "nunique"),
            )
            .reset_index()
        )
        for _, row in monthly.iterrows():
            text = (
                f"Month {row['month']}: "
                f"total sales R${row['total_sales']:,.2f}, "
                f"{row['total_orders']:,} orders."
            )
            docs.append(text)
            ids.append(f"monthly_{row['month']}")
            metas.append({"type": "monthly_trend", "month": str(row["month"])})
