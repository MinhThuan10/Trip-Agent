from typing import List
from langchain_core.documents import Document
from app.src.services.rag_service import rag_service
from app.src.services.base import base_service


def retrieve_and_rerank(
        query: str,
        top_k_retrieve: int = 10,
        top_k_final: int = 5,
        rerank_threshold: float = 0.6,
    ) -> List[Document]:
        """
        Retrieve documents từ vector store
        và rerank bằng ranking model.
        """

        retrieved_docs = rag_service.similarity_search(
            query=query,
            k=top_k_retrieve,
        )

        if not retrieved_docs:
            return []

        pairs = [
            (query, doc.page_content)
            for doc in retrieved_docs
        ]

        scores = base_service.ranking_model.predict(pairs)

        scored_docs = sorted(
            zip(scores, retrieved_docs),
            key=lambda item: item[0],
            reverse=True,
        )

        final_docs = [
            doc
            for score, doc in scored_docs[:top_k_final]
            if score >= rerank_threshold
        ]

        return final_docs


def build_context(
        docs: List[Document],
    ) -> str:
        """
        Chuyển retrieved documents thành context
        cho LLM.
        """

        if not docs:
            return "Không có tài liệu liên quan."

        context_parts = []

        for idx, doc in enumerate(docs, 1):

            source = doc.metadata.get(
                "file_name",
                "Tài liệu hỗ trợ",
            )

            category = doc.metadata.get(
                "category",
                "chung",
            )

            context_parts.append(
                f"--- Tài liệu {idx} "
                f"[{source} | {category}] ---\n"
                f"{doc.page_content}"
            )

        return "\n\n".join(context_parts)