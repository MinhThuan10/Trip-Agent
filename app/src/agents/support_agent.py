from typing import List, Dict, Any, Optional
import json

from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage

from app.src.services.base import base_service
from langchain_core.tools import tool
from app.src.tools.search_support_tools import (
    retrieve_and_rerank,
    build_context
)
# ============================================================
# Tools
# ============================================================

@tool
def search_support(query: str) -> str:
    """
    Search internal customer support knowledge base.
    """

    documents = retrieve_and_rerank(query)

    context = build_context(documents)

    return {
        "query": query,
        "documents": documents,
        "context": context,
    }


class SupportRAGAgent:
    support_prompt = base_service.langfuse.get_prompt("Support_Prompt")
    def __init__(
            self,
            model=None,
            tools: List[Any] | None = None,
        ):

        self.model = model or base_service.llm

        self.tools = tools or [
            search_support,
        ]

        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=self.support_prompt.prompt,
        )

    # ========================================================
    # Process request
    # ========================================================
    def process_request(
            self,
            input_data: Dict[str, Any],
        ) -> Dict[str, Any]:
    
        try:
            response = self.agent.invoke(
                input_data,
                config={
                    "callbacks": [
                        base_service.langfuse_handler
                    ]
                },
            )

            messages = response.get("messages", [])

            if not messages:
                return {
                    "success": True,
                    "answer": "Đã xử lý yêu cầu.",
                    "sources": [],
                }

            # =========================
            # 1. Lấy answer cuối cùng
            # =========================

            answer = ""

            for message in reversed(messages):
                # Không lấy ToolMessage làm answer
                if isinstance(message, ToolMessage):
                    continue

                content = getattr(message, "content", None)

                if content:
                    answer = content
                    break

            # =========================
            # 2. Lấy kết quả từ tools
            # =========================

            sources = []

            for message in messages:
                if isinstance(message, ToolMessage):

                    content = message.content

                    if isinstance(content, str):
                        try:
                            content = json.loads(content)
                        except json.JSONDecodeError:
                            pass

                    sources.append({
                        "tool": getattr(
                            message,
                            "name",
                            None,
                        ),
                        "tool_call_id": getattr(
                            message,
                            "tool_call_id",
                            None,
                        ),
                        "content": content,
                    })

            return {
                "success": True,
                "answer": answer,
                "sources": sources,
            }

        except Exception as e:

            return {
                "success": False,
                "answer": f"Lỗi khi xử lý agent: {str(e)}",
                "sources": [],
            }


support_rag_agent = SupportRAGAgent()