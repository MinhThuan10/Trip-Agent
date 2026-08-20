from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage

from app.src.services.base import base_service
from app.src.services.rag_service import rag_service


class SupportRAGAgent:

    SYSTEM_PROMPT = """
Bạn là một trợ lý ảo chăm sóc khách hàng chuyên nghiệp
của hệ thống bán vé máy bay.

Nhiệm vụ của bạn là giải đáp thắc mắc của khách hàng
dựa ONLY vào TÀI LIỆU THAM KHẢO được cung cấp.

Quy tắc:

1. Chỉ sử dụng thông tin có trong TÀI LIỆU THAM KHẢO.
2. Không tự suy đoán hoặc bịa thêm thông tin.
3. Nếu tài liệu không chứa đủ thông tin để trả lời,
   hãy thông báo rằng bạn chưa có đủ thông tin trong
   tài liệu hiện có và đề nghị khách hàng liên hệ
   bộ phận hỗ trợ.
4. Trả lời bằng tiếng Việt.
5. Trả lời ngắn gọn, rõ ràng, lịch sự và hữu ích.
6. Nếu có nhiều tài liệu liên quan, hãy tổng hợp chúng
   thành một câu trả lời thống nhất.
"""

    def __init__(
        self,
        llm=None,
        rag_service_instance=None,
        ranking_model=None,
        top_k_retrieve: int = 10,
        top_k_final: int = 4,
        rerank_threshold: float = 0.6,
    ):
        """
        Khởi tạo Support RAG Agent.

        Args:
            llm:
                LLM dùng để sinh câu trả lời.

            rag_service_instance:
                Service dùng để retrieve documents.

            ranking_model:
                Model dùng để rerank documents.

            top_k_retrieve:
                Số lượng documents lấy từ vector search.

            top_k_final:
                Số lượng documents tối đa sau reranking.

            rerank_threshold:
                Ngưỡng score tối thiểu để giữ document.
        """

        self.llm = llm or base_service.llm
        self.rag_service = (
            rag_service_instance
            or rag_service
        )
        self.ranking_model = (
            ranking_model
            or base_service.ranking_model
        )

        self.top_k_retrieve = top_k_retrieve
        self.top_k_final = top_k_final
        self.rerank_threshold = rerank_threshold

    # =========================================================
    # Retrieval
    # =========================================================

    def retrieve_and_rerank(
        self,
        query: str,
    ) -> List[Document]:
        """
        Retrieve documents từ vector store
        và rerank bằng ranking model.
        """

        retrieved_docs = self.rag_service.similarity_search(
            query=query,
            k=self.top_k_retrieve,
        )

        if not retrieved_docs:
            return []

        pairs = [
            (query, doc.page_content)
            for doc in retrieved_docs
        ]

        scores = self.ranking_model.predict(pairs)

        scored_docs = sorted(
            zip(scores, retrieved_docs),
            key=lambda item: item[0],
            reverse=True,
        )

        final_docs = [
            doc
            for score, doc in scored_docs[:self.top_k_final]
            if score >= self.rerank_threshold
        ]

        return final_docs

    # =========================================================
    # Context
    # =========================================================

    def build_context(
        self,
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

    # =========================================================
    # Generate answer
    # =========================================================

    def generate_answer(
        self,
        query: str,
        context: str,
    ) -> str:
        """
        Sinh câu trả lời dựa trên query + retrieved context.
        """

        system_prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            "=== TÀI LIỆU THAM KHẢO ===\n"
            f"{context}\n"
            "============================"
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": query,
            },
        ]

        response = self.llm.invoke(messages, config={"callbacks": [base_service.langfuse_handler]})

        return getattr(
            response,
            "content",
            str(response),
        )

    # =========================================================
    # Main execution
    # =========================================================

    def invoke(
        self,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Interface chuẩn để Multi-Agent Graph gọi Support Agent.

        Expected input:

        {
            "messages": [...]
        }
        """

        try:

            messages = input_data.get(
                "messages",
                [],
            )

            query = self._extract_latest_user_message(
                messages
            )

            if not query:
                return {
                    "success": False,
                    "answer": "Không tìm thấy câu hỏi của khách hàng.",
                    "sources": [],
                }

            relevant_docs = self.retrieve_and_rerank(
                query
            )

            context = self.build_context(
                relevant_docs
            )

            answer = self.generate_answer(
                query=query,
                context=context,
            )

            sources = [
                {
                    "file_name": doc.metadata.get(
                        "file_name"
                    ),
                    "category": doc.metadata.get(
                        "category"
                    ),
                }
                for doc in relevant_docs
            ]

            return {
                "success": True,
                "answer": answer,
                "sources": sources,
            }

        except Exception as e:

            return {
                "success": False,
                "answer": (
                    "Đã xảy ra lỗi khi xử lý "
                    f"yêu cầu hỗ trợ: {str(e)}"
                ),
                "sources": [],
            }

    # =========================================================
    # Helpers
    # =========================================================

    @staticmethod
    def _extract_latest_user_message(
        messages: List[Any],
    ) -> Optional[str]:
        """
        Lấy message user mới nhất.
        """

        for message in reversed(messages):

            if isinstance(message, dict):

                if message.get("role") == "user":
                    return message.get(
                        "content",
                        "",
                    )

            elif isinstance(
                message,
                HumanMessage,
            ):

                return message.content

            elif hasattr(message, "type"):

                if message.type == "human":
                    return message.content

        return None


support_rag_agent = SupportRAGAgent()