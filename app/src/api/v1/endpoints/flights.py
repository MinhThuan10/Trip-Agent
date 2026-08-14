from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional
from app.src.tools.flight_search_tool import search_flights_external

router = APIRouter()

class FlightSearchRequest(BaseModel):
    start_point: str = Field(..., description="Mã sân bay đi, ví dụ: SGN")
    end_point: str = Field(..., description="Mã sân bay đến, ví dụ: UIH")
    depart_date: str = Field(..., description="Ngày đi, định dạng YYYY-MM-DD")
    adt: int = Field(1, description="Số lượng người lớn")
    chd: int = Field(0, description="Số lượng trẻ em")
    inf: int = Field(0, description="Số lượng em bé")
    flight_type: str = Field("DOMESTIC", description="Loại chuyến bay: DOMESTIC hoặc INTERNATIONAL")
    airline: Optional[str] = Field("VN", description="Mã hãng hàng không")
    itinerary: int = Field(0, description="Loại hành trình: 0 (một chiều), 1 (khứ hồi)")
    language: str = Field("VI", description="Ngôn ngữ")
    view_mode: int = Field(2, description="Chế độ hiển thị")

@router.post("/search")
def search_flights(payload: FlightSearchRequest):
    """
    API tìm kiếm chuyến bay gọi tới hệ thống ngoài vere.me.
    """
    result = search_flights_external(
        start_point=payload.start_point,
        end_point=payload.end_point,
        depart_date=payload.depart_date,
        adt=payload.adt,
        chd=payload.chd,
        inf=payload.inf,
        flight_type=payload.flight_type,
        airline=payload.airline,
        itinerary=payload.itinerary,
        language=payload.language,
        view_mode=payload.view_mode
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=502, detail=result.get("error"))
        
    return result.get("data")
