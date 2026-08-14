import httpx
from typing import Optional, Dict, Any, List
from app.src.services.base import base_service

def search_flights_external(
    start_point: str,
    end_point: str,
    depart_date: str,
    adt: int = 1,
    chd: int = 0,
    inf: int = 0,
    flight_type: str = "DOMESTIC",
    airline: Optional[str] = "VN",
    itinerary: int = 0,
    language: str = "VI",
    view_mode: int = 2
) -> Dict[str, Any]:
    """
    Search for flights using the external flight search API.
    Mandatory parameters: start_point, end_point, depart_date.
    """
    payload = {
        "adt": adt,
        "chd": chd,
        "inf": inf,
        "flight_type": flight_type.upper(),
        "flights": [
            {
                "start_point": start_point,
                "end_point": end_point,
                "depart_date": depart_date,
                "airline": airline
            }
        ],
        "itinerary": itinerary,
        "language": language.upper(),
        "view_mode": view_mode
    }
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(base_service.flight_search_api_url, json=payload)
            response.raise_for_status()
            return {"success": True, "data": response.json()}
    except httpx.HTTPStatusError as e:
        return {"success": False, "error": f"HTTP error occurred: {e.response.status_code} - {e.response.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

