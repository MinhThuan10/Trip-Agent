from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.src.agents.flight_agent import process_flight_request

router = APIRouter()

class ChatRequest(BaseModel):
    message: str = Field(..., description="Yêu cầu tìm kiếm vé máy bay của người dùng")

class ChatResponse(BaseModel):
    success: bool
    message: str = ""
    data: dict = {}

@router.post("/flight-agent", response_model=ChatResponse)
def chat_with_flight_agent(payload: ChatRequest):
    """
    Endpoint xử lý chat với Flight Agent sử dụng Structured Output và tra cứu thông tin sân bay/hãng bay tự động.
    """
    try:
        result = process_flight_request(payload.message)
        if not result.get("success"):
            return ChatResponse(
                success=False,
                message=result.get("message", "Thiếu thông tin hoặc không tìm thấy dữ liệu."),
                data={}
            )
        return ChatResponse(
            success=True,
            message="Tìm kiếm chuyến bay thành công.",
            data=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
