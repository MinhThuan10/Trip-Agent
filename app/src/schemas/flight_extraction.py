from typing import Optional, List
from pydantic import BaseModel, Field

class FlightSegment(BaseModel):
    start_point: Optional[str] = Field(None, description="Tên thành phố hoặc sân bay đi nếu người dùng cung cấp")
    end_point: Optional[str] = Field(None, description="Tên thành phố hoặc sân bay đến nếu người dùng cung cấp")
    depart_date: Optional[str] = Field(None, description="Ngày khởi hành theo định dạng YYYY-MM-DD nếu người dùng cung cấp")
    airline: Optional[str] = Field(None, description="Tên hãng hàng không nếu người dùng có chỉ định")

class FlightSearchExtraction(BaseModel):
    adt: int = Field(1, description="Số lượng người lớn")
    chd: int = Field(0, description="Số lượng trẻ em")
    inf: int = Field(0, description="Số lượng em bé")
    flight_type: str = Field("DOMESTIC", description="Loại chuyến bay (DOMESTIC hoặc INTERNATIONAL)")
    flights: List[FlightSegment] = Field(..., description="Danh sách các chặng bay")
    itinerary: int = Field(0, description="0: Một chiều, 1: Khứ hồi")
    language: str = Field("VI", description="Ngôn ngữ")
    view_mode: int = Field(2, description="Chế độ hiển thị")
