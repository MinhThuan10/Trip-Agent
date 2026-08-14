from langchain.agents import create_agent
from langchain_core.tools import tool
from app.src.services.base import base_service
from app.src.tools.search_tools import search_airports, search_airlines, get_all_airline_codes
from app.src.tools.flight_search_tool import search_flights_external
from datetime import date, timedelta

@tool
def tool_get_current_date() -> str:
    """Lấy ngày hiện tại (định dạng YYYY-MM-DD) và thông tin thứ để tính toán ngày mai, ngày kia hoặc các ngày trong tương lai."""
    today = date.today()
    return f"Hôm nay là ngày {today.strftime('%Y-%m-%d')}."

@tool
def tool_search_airports(query: str, limit: int = 5) -> list:
    """Tìm kiếm sân bay dựa trên tên, thành phố hoặc mã IATA (ví dụ: 'Hà Nội', 'SGN', 'Đà Nẵng')."""
    return search_airports(query, limit)

@tool
def tool_search_airlines(query: str, limit: int = 5) -> list:
    """Tìm kiếm hãng hàng không dựa trên tên hoặc mã hãng (ví dụ: 'Vietnam Airlines', 'VJ')."""
    return search_airlines(query, limit)

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
    view_mode: int = 2
) -> dict:
    """Thực hiện tìm kiếm chuyến bay qua hệ thống bên ngoài. Yêu cầu mã sân bay IATA đi, đến (ví dụ: 'HAN', 'SGN') và ngày khởi hành định dạng YYYY-MM-DD."""
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
        view_mode=view_mode
    )

system_prompt = """
Bạn là một trợ lý AI thông minh chuyên tư vấn và tìm kiếm vé máy bay cho khách hàng.
Bạn có quyền truy cập vào các công cụ lấy ngày hiện tại, tìm kiếm sân bay, hãng hàng không và tìm kiếm vé máy bay.

Hãy luôn tuân thủ các bước sau khi xử lý yêu cầu:
1. Khi khách hàng nói về các mốc thời gian tương đối như "hôm nay", "ngày mai", "3 ngày nữa", hãy sử dụng công cụ `tool_get_current_date` để xác định chính xác ngày tháng (YYYY-MM-DD).
2. Nếu người dùng cung cấp tên địa điểm dạng thông thường (như "Sài Gòn", "Hà Nội"), hãy dùng công cụ `tool_search_airports` để tìm mã IATA tương ứng (ví dụ: SGN, HAN).
3. Nếu người dùng chỉ định hãng bay, hãy dùng `tool_search_airlines` để xác định mã hãng nếu cần.
4. Sau khi đã có đầy đủ thông tin mã sân bay đi, mã sân bay đến và ngày khởi hành, hãy gọi `tool_search_flights` để lấy danh sách chuyến bay và trả kết quả chi tiết, rõ ràng cho khách hàng.
5. Nếu thiếu thông tin bắt buộc (điểm đi, điểm đến, ngày đi), hãy hỏi lại khách hàng một cách thân thiện.
"""

flight_agent = create_agent(
    model=base_service.llm,
    tools=[
        tool_get_current_date,
        tool_search_airports,
        tool_search_airlines,
        tool_get_all_airline_codes,
        tool_search_flights
    ],
    system_prompt=system_prompt
)

def process_flight_request(user_message: str) -> dict:
    try:
        response = flight_agent.invoke({"messages": [{"role": "user", "content": user_message}]})
        messages = response.get("messages", [])
        if messages:
            last_message = messages[-1]
            content = getattr(last_message, "content", str(last_message))
            return {
                "success": True,
                "message": content,
                "messages": [{"role": m.type if hasattr(m, "type") else "assistant", "content": m.content} for m in messages]
            }
        return {"success": True, "message": "Đã xử lý yêu cầu."}
    except Exception as e:
        return {"success": False, "message": f"Lỗi khi xử lý agent: {str(e)}"}
