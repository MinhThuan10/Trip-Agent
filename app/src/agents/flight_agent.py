from datetime import date
from typing import Any, Dict, List
import json
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.messages import ToolMessage
from app.src.services.base import base_service
from app.src.tools.search_tools import (
    search_airports,
    search_airlines,
    get_all_airline_codes,
)
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


@tool
def tool_search_flights(
    start_point: str,
    end_point: str,
    depart_date: str,
    adt: int = 1,
    chd: int = 0,
    inf: int = 0,
    flight_type: str = "DOMESTIC",
    airline: str = "VN",
    itinerary: int = 0,
    language: str = "VI",
    view_mode: int = 2,
) -> dict:
    """Thực hiện tìm kiếm chuyến bay qua hệ thống bên ngoài.

    Yêu cầu mã sân bay IATA đi, đến (ví dụ: 'HAN', 'SGN')
    và ngày khởi hành định dạng YYYY-MM-DD.
    """
    return search_flights_external(
        start_point=start_point,
        end_point=end_point,
        depart_date=depart_date,
        adt=adt,
        chd=chd,
        inf=inf,
        flight_type=flight_type,
        airline=airline,
        itinerary=itinerary,
        language=language,
        view_mode=view_mode,
    )


# ============================================================
# Flight Agent
# ============================================================

class FlightAgent:

    SYSTEM_PROMPT = """
Bạn là một trợ lý AI thông minh chuyên tư vấn và tìm kiếm
vé máy bay cho khách hàng.

Bạn có quyền truy cập vào các công cụ:
- lấy ngày hiện tại
- tìm kiếm sân bay
- tìm kiếm hãng hàng không
- lấy danh sách mã hãng hàng không
- tìm kiếm vé máy bay

Hãy luôn tuân thủ các bước sau khi xử lý yêu cầu:

1. Khi khách hàng nói về các mốc thời gian tương đối như
   "hôm nay", "ngày mai", "3 ngày nữa", hãy sử dụng
   `tool_get_current_date` để xác định chính xác ngày tháng
   (YYYY-MM-DD).

2. Nếu người dùng cung cấp tên địa điểm dạng thông thường
   (như "Sài Gòn", "Hà Nội"), hãy dùng
   `tool_search_airports` để tìm mã IATA tương ứng.

3. Nếu người dùng chỉ định hãng bay, hãy dùng
   `tool_search_airlines` để xác định mã hãng nếu cần.

4. Sau khi đã có đầy đủ:
   - mã sân bay đi
   - mã sân bay đến
   - ngày khởi hành

   hãy gọi `tool_search_flights` để lấy danh sách
   chuyến bay.

5. Nếu thiếu thông tin bắt buộc:
   - điểm đi
   - điểm đến
   - ngày đi

   hãy hỏi lại khách hàng một cách thân thiện.

6. Không được tự suy đoán mã IATA.
   Nếu người dùng cung cấp tên địa điểm, phải sử dụng
   `tool_search_airports` để xác định mã sân bay.

7. Không được tự tạo thông tin chuyến bay.
   Chỉ trả về các chuyến bay được cung cấp bởi
   `tool_search_flights`.

Hãy trả lời bằng tiếng Việt, rõ ràng và hữu ích, Bạn chỉ có nhiệm vụ là tìm kiếm chuyến bay, không được trả lời các thông tin ngoài lề.
"""

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
            system_prompt=self.SYSTEM_PROMPT,
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

 
# ============================================================
# Instance
# ============================================================

flight_agent = FlightAgent()