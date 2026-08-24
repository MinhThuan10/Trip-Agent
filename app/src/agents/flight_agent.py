from datetime import date
from typing import Any, Dict, List
import json
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from app.src.services.base import base_service
from app.src.tools.search_flight_tools import (
    search_airports,
    search_airlines,
    get_all_airline_codes,
)
from app.src.tools.llm_tools import extract_text_content
from app.src.tools.flight_search_tool import search_flights_external


# ============================================================
# Tools
# ============================================================

@tool
def tool_get_current_date() -> str:
    """Lấy ngày hiện tại (định dạng YYYY-MM-DD) và thông tin thứ
    để tính toán ngày mai, ngày kia hoặc các ngày trong tương lai.
    """
    today = date.today()

    return (
        f"Hôm nay là ngày "
        f"{today.strftime('%Y-%m-%d')}."
    )


@tool
def tool_search_airports(
    query: str,
    limit: int = 5,
) -> list:
    """Tìm kiếm sân bay dựa trên tên, thành phố hoặc mã IATA
    (ví dụ: 'Hà Nội', 'SGN', 'Đà Nẵng').
    """
    return search_airports(
        query,
        limit,
    )


@tool
def tool_search_airlines(
    query: str,
    limit: int = 5,
) -> list:
    """Tìm kiếm hãng hàng không dựa trên tên hoặc mã hãng
    (ví dụ: 'Vietnam Airlines', 'VJ').
    """
    return search_airlines(
        query,
        limit,
    )


@tool
def tool_get_all_airline_codes() -> list:
    """Lấy danh sách tất cả các mã hãng hàng không có sẵn."""
    return get_all_airline_codes()


from typing import List

@tool
def tool_search_flights(
    start_point: str,
    end_point: str,
    depart_date: str,
    adt: int = 1,
    chd: int = 0,
    inf: int = 0,
    flight_type: str = "DOMESTIC",
    airlines: List[str] | None = None,
    itinerary: int = 0,
    language: str = "VI",
    view_mode: int = 2,
) -> dict:
    """Thực hiện tìm kiếm chuyến bay qua hệ thống bên ngoài.

    Args:
        start_point: Mã IATA sân bay đi, ví dụ HAN, SGN.
        end_point: Mã IATA sân bay đến, ví dụ SGN, UIH.
        depart_date: Ngày khởi hành, định dạng YYYY-MM-DD.
        adt: Số lượng người lớn.
        chd: Số lượng trẻ em.
        inf: Số lượng em bé.
        flight_type: Loại chuyến bay, mặc định DOMESTIC.
        airlines: Danh sách mã hãng bay IATA, ví dụ
            ["VN"], ["VJ"], hoặc ["VN", "VJ", "QH"].
        itinerary: Loại hành trình.
        language: Ngôn ngữ.
        view_mode: Chế độ hiển thị.

    Returns:
        Kết quả tìm kiếm chuyến bay từ các hãng được yêu cầu.
    """

    return search_flights_external(
        start_point=start_point,
        end_point=end_point,
        depart_date=depart_date,
        adt=adt,
        chd=chd,
        inf=inf,
        flight_type=flight_type,
        airlines=airlines,
        itinerary=itinerary,
        language=language,
        view_mode=view_mode,
    )


# ============================================================
# Flight Agent
# ============================================================

class FlightAgent:
    flight_prompt = base_service.langfuse.get_prompt("Flight_Prompt")

    def __init__(
        self,
        model=None,
        tools: List[Any] | None = None,
    ):
        """
        Khởi tạo Flight Agent.

        Args:
            model:
                LLM dùng cho agent.
                Nếu không truyền thì sử dụng base_service.llm.

            tools:
                Danh sách tools tùy chỉnh.
                Nếu không truyền sẽ sử dụng tools mặc định.
        """

        self.model = model or base_service.llm

        self.tools = tools or [
            tool_get_current_date,
            tool_search_airports,
            tool_search_airlines,
            tool_get_all_airline_codes,
            tool_search_flights,
        ]

        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=self.flight_prompt.prompt,
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

                if isinstance(message, ToolMessage):
                    continue

                content = getattr(message, "content", None)

                answer = extract_text_content(content)

                if answer:
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


flight_agent = FlightAgent()