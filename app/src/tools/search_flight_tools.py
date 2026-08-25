import json
import os
import unicodedata
from difflib import get_close_matches
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
AIRPORTS_FILE = BASE_DIR / "data" / "airports.jsonl"
AIRLINES_FILE = BASE_DIR / "data" / "airlines.jsonl"

def normalize_text(text: str) -> str:
    if not text:
        return ""
    nfkd_form = unicodedata.normalize('NFKD', text)
    no_diacritics = "".join([c for c in nfkd_form if unicodedata.category(c) != 'Mn'])
    no_diacritics = no_diacritics.replace('đ', 'd').replace('Đ', 'D')
    return " ".join(no_diacritics.lower().split())

def search_airports(query: str, limit: int = 5) -> list:
    results = []
    q_norm = normalize_text(query)


    if not os.path.exists(AIRPORTS_FILE):
        return results

    all_items = []
    with open(AIRPORTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    all_items.append(json.loads(line))
                except Exception:
                    continue

    matched = []
    for item in all_items:
        iata = normalize_text(item.get("iata_code", ""))
        name = normalize_text(item.get("airport_name", ""))
        city = normalize_text(item.get("city", ""))
        prov = normalize_text(item.get("province", ""))
        aliases = [normalize_text(a) for a in item.get("aliases", [])]
        
        if (q_norm in iata or q_norm in name or q_norm in city or q_norm in prov or 
            any(q_norm in alias or alias in q_norm for alias in aliases)):
            results.append(item)
            if len(results) >= limit:
                break

    if matched:
        return matched[:limit]

    city_names = [normalize_text(item.get("city", "")) for item in all_items]
    close_cities = get_close_matches(q_norm, city_names, n=limit, cutoff=0.6)
    
    for item in all_items:
        if normalize_text(item.get("city", "")) in close_cities:
            if item not in results:
                results.append(item)
                if len(results) >= limit:
                    break

    return results

def search_airlines(query: str, limit: int = 5) -> list:
    results = []
    q_norm = normalize_text(query)
    if not os.path.exists(AIRLINES_FILE):
        return results

    all_items = []
    with open(AIRLINES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    all_items.append(json.loads(line))
                except Exception:
                    continue

    for item in all_items:
        iata = normalize_text(item.get("iata_code", ""))
        icao = normalize_text(item.get("icao_code", ""))
        name = normalize_text(item.get("airline_name", ""))
        short = normalize_text(item.get("short_name", ""))
        aliases = [normalize_text(a) for a in item.get("aliases", [])]
        
        if (q_norm in iata or q_norm in icao or q_norm in name or q_norm in short or 
            any(q_norm in alias or alias in q_norm for alias in aliases)):
            results.append(item)
            if len(results) >= limit:
                break

    return results

def get_all_airline_codes() -> list:
    """Lấy danh sách tất cả các mã hãng hàng không (iata_code) hiện có."""
    codes = []
    if not os.path.exists(AIRLINES_FILE):
        return codes
    with open(AIRLINES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    code = item.get("iata_code")
                    if code and code not in codes:
                        codes.append(code)
                except Exception:
                    continue
    return codes
