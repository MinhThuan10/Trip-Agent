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
    airlines: Optional[List[str]] = None,
    itinerary: int = 0,
    language: str = "VI",
    view_mode: int = 2,
) -> Dict[str, Any]:
    """
    Search flights from one or multiple airlines.

    airlines:
        ["VN"]
        ["VN", "VJ", "QH"]
    """

    if not airlines:
        airlines = ["VN"]

    all_fare_data = []
    results = []

    try:
        with httpx.Client(timeout=30.0) as client:

            for airline in airlines:

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
                            "airline": airline,
                        }
                    ],
                    "itinerary": itinerary,
                    "language": language.upper(),
                    "view_mode": view_mode,
                }

                print(f"Searching airline: {airline}")
                print(payload)

                response = client.post(
                    base_service.flight_search_api_url,
                    json=payload,
                )

                response.raise_for_status()

                data = response.json()

                results.append({
                    "airline": airline,
                    "success": True,
                    "data": data,
                })

                # Gộp fare_data
                fare_data = data.get("fare_data", [])

                if fare_data:
                    all_fare_data.extend(fare_data)

        return {
            "success": True,
            "data": {
                "flight_type": flight_type.upper(),
                "itinerary": itinerary,
                "fare_data": all_fare_data,
                "airlines": airlines,
                "results": results,
            },
        }

    except httpx.HTTPStatusError as e:

        return {
            "success": False,
            "error": (
                f"HTTP error occurred: "
                f"{e.response.status_code} - "
                f"{e.response.text}"
            ),
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
        }