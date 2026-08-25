from fastapi import APIRouter, HTTPException
import os
import json
from pathlib import Path

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[4]
AIRPORTS_FILE = BASE_DIR / "data" / "airports.jsonl"
AIRLINES_FILE = BASE_DIR / "data" / "airlines.jsonl"



def load_jsonl(file_path: str):
    try:
        data = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if line:
                    data.append(json.loads(line))

        return data

    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail=f"File not found: {file_path}"
        )

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid JSONL file: {file_path}, error: {str(e)}"
        )

@router.get("/airports")
def get_airports(limit: int = 100, offset: int = 0):
    airports = load_jsonl(AIRPORTS_FILE)

    data = airports[offset: offset + limit]

    return {
        "success": True,
        "data": data,
        "total": len(airports),
        "limit": limit,
        "offset": offset,
    }


@router.get("/airlines")
def get_airlines(limit: int = 100, offset: int = 0):
    airlines = load_jsonl(AIRLINES_FILE)

    data = airlines[offset: offset + limit]

    return {
        "success": True,
        "data": data,
        "total": len(airlines),
        "limit": limit,
        "offset": offset,
    }
