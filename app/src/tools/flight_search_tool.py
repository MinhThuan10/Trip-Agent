import httpx

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, List

from app.src.services.base import base_service


def search_one_airline(
    client: httpx.Client,
    airline: str,
    start_point: str,
    end_point: str,
    depart_date: str,
    adt: int,
    chd: int,
    inf: int,
    flight_type: str,
    itinerary: int,
    language: str,
    view_mode: int,
) -> Dict[str, Any]:
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

    try:
        response = client.post(
            base_service.flight_search_api_url,
            json=payload,
        )

        response.raise_for_status()

        data = response.json()

        return {
            "airline": airline,
            "success": True,
            "data": data,
        }

    except httpx.HTTPStatusError as e:
        return {
            "airline": airline,
            "success": False,
            "error": (
                f"HTTP error occurred: "
                f"{e.response.status_code} - "
                f"{e.response.text}"
            ),
        }

    except Exception as e:
        return {
            "airline": airline,
            "success": False,
            "error": str(e),
        }


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
    Search flights from multiple airlines in parallel.

    airlines:
        ["VN"]
        ["VN", "VJ", "QH"]
    """

    if not airlines:
        return {
            "success": True,
            "data": {},
        }

    all_fare_data = []
    results = []

    try:
        # Dùng chung HTTP connection pool cho các thread
        with httpx.Client(timeout=30.0) as client:

            max_workers = min(len(airlines), 10)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:

                futures = {
                    executor.submit(
                        search_one_airline,
                        client,
                        airline,
                        start_point,
                        end_point,
                        depart_date,
                        adt,
                        chd,
                        inf,
                        flight_type,
                        itinerary,
                        language,
                        view_mode,
                    ): airline
                    for airline in airlines
                }

                for future in as_completed(futures):
                    airline = futures[future]

                    try:
                        result = future.result()
                    except Exception as e:
                        result = {
                            "airline": airline,
                            "success": False,
                            "error": str(e),
                        }

                    results.append(result)

                    # Chỉ merge fare_data nếu request thành công
                    if result["success"]:
                        fare_data = result["data"].get("fare_data", [])

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

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
