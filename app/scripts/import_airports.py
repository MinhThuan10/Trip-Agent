import json
import os
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "airports.jsonl"

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://myuser:mysecretpassword@localhost:5432/mydatabase",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "minhthuan77f1/binhdinh-embedding",
)

EMBEDDING_DIM = 768

BATCH_SIZE = 64


# ============================================================
# DATABASE
# ============================================================

CREATE_TABLE_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS airport_embeddings (
    id BIGSERIAL PRIMARY KEY,

    iata_code VARCHAR(3) NOT NULL UNIQUE,
    airport_name TEXT NOT NULL,

    city TEXT,
    province TEXT,
    country TEXT,
    airport_type TEXT,

    embedding VECTOR({EMBEDDING_DIM}) NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS airport_embedding_hnsw_idx
ON airport_embeddings
USING hnsw (embedding vector_cosine_ops);
"""


# ============================================================
# HELPERS
# ============================================================

def load_jsonl(file_path: Path) -> list[dict]:
    data = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            data.append(json.loads(line))

    return data


def build_embedding_text(airport: dict) -> str:
    """
    Build text used for embedding.

    Example:

    Sân bay Quốc tế Nội Bài, Hà Nội, Hà Nội, Việt Nam
    """

    fields = [
        airport.get("airport_name"),
        airport.get("city"),
        airport.get("province"),
        airport.get("country"),
    ]

    return ", ".join(
        value.strip()
        for value in fields
        if value
    )


# ============================================================
# IMPORT
# ============================================================

def main():
    print(f"Loading airport data from: {DATA_FILE}")

    airports = load_jsonl(DATA_FILE)

    print(f"Found {len(airports)} airports")

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    # --------------------------------------------------------
    # Connect PostgreSQL
    # --------------------------------------------------------

    print("Connecting to PostgreSQL...")

    with psycopg.connect(DATABASE_URL) as conn:

        register_vector(conn)

        # ----------------------------------------------------
        # Create table
        # ----------------------------------------------------

        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            cur.execute(CREATE_INDEX_SQL)

        conn.commit()

        print("Airport table/index ready.")

        # ----------------------------------------------------
        # Generate embeddings
        # ----------------------------------------------------

        texts = [
            build_embedding_text(airport)
            for airport in airports
        ]

        print("Generating embeddings...")

        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        # ----------------------------------------------------
        # Insert / Update
        # ----------------------------------------------------

        print("Importing airports...")

        with conn.cursor() as cur:

            for airport, embedding in zip(
                airports,
                embeddings,
            ):
                cur.execute(
                    """
                    INSERT INTO airport_embeddings (
                        iata_code,
                        airport_name,
                        city,
                        province,
                        country,
                        airport_type,
                        embedding
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (iata_code)
                    DO UPDATE SET
                        airport_name = EXCLUDED.airport_name,
                        city = EXCLUDED.city,
                        province = EXCLUDED.province,
                        country = EXCLUDED.country,
                        airport_type = EXCLUDED.airport_type,
                        embedding = EXCLUDED.embedding,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        airport["iata_code"],
                        airport["airport_name"],
                        airport.get("city"),
                        airport.get("province"),
                        airport.get("country"),
                        airport.get("airport_type"),
                        embedding,
                    ),
                )

        conn.commit()

    print("Airport import completed successfully.")


if __name__ == "__main__":
    main()