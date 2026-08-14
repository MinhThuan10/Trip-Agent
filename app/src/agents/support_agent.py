from typing import List, Dict, Any
from langchain_core.documents import Document
from app.src.services.base import base_service
from app.src.services.rag_service import rag_service
import numpy as np

def cosine_similarity(a: List[float], b: List[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

class SupportRAGAgent:
    def __init__(self):
        self.rag_service = rag_service
        self.ranking_model = base_service.ranking_model
        self.llm = base_service.llm

    def retrieve_and_rerank(self, query: str, top_k_retrieve: int = 10, top_k_final: int = 4, threshold = 0.6) -> List[Document]:
        retrieved_docs = self.rag_service.similarity_search(query=query, k=top_k_retrieve)
        if not retrieved_docs:
            return []
        scores = self.ranking_model.predict([(query, docs.page_content) for docs in retrieved_docs])
        scored_docs = sorted(
            zip(scores, retrieved_docs),
            key=lambda x: x[0],
            reverse=True
        )
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        final_docs = [
            doc
            for score, doc in scored_docs[:top_k_final]
            if score >= threshold
        ]
        return final_docs

    def build_context(self, docs: List[Document]) -> str:
        if not docs:
            return "Không có tài liệu liên quan."
        
        context_parts = []
        for idx, doc in enumerate(docs, 1):
            source = doc.metadata.get("file_name", "Tài liệu hỗ trợ")
            category = doc.metadata.get("category", "chung")
            context_parts.append(f"--- Tài liệu {idx} [{source} | {category}] ---\n{doc.page_content}")
        
        return "\n\n".join(context_parts)

    def answer_query(self, query: str) -> Dict[str, Any]:
        try:
            relevant_docs = self.retrieve_and_rerank(query, top_k_retrieve=10, top_k_final=4)
            context = self.build_context(relevant_docs)

            system_prompt = f"""
Bạn là một trợ lý ảo chăm sóc khách hàng chuyên nghiệp của hệ thống bán vé (vé máy bay, vé xe).
Nhiệm vụ của bạn là giải đáp thắc mắc của khách hàng dựa vào phần TÀI LIỆU THAM KHẢO được cung cấp dưới đây.
Hãy trả lời một cách chính xác, lịch sự, ngắn gọn và hữu ích. Nếu thông tin không có trong tài liệu, hãy khéo léo thông báo bạn chưa có thông tin đó và đề nghị liên hệ tổng đài.

=== TÀI LIỆU THAM KHẢO ===
{context}
===========================
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            response = self.llm.invoke(messages)
            answer_text = getattr(response, "content", str(response))

            return {
                "success": True,
                "answer": answer_text,
                "sources": [{"file_name": doc.metadata.get("file_name"), "category": doc.metadata.get("category")} for doc in relevant_docs]
            }

        except Exception as e:
            return {
                "success": False,
                "answer": f"Đã xảy ra lỗi khi xử lý yêu cầu hỗ trợ: {str(e)}",
                "sources": []
            }

support_rag_agent = SupportRAGAgent()
