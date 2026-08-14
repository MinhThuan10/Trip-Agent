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

DATA_FILE = BASE_DIR / "data" / "airlines.jsonl"

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

CREATE TABLE IF NOT EXISTS airline_embeddings (
    id BIGSERIAL PRIMARY KEY,

    iata_code VARCHAR(3) NOT NULL UNIQUE,
    icao_code VARCHAR(4),

    airline_name TEXT NOT NULL,
    short_name TEXT,

    aliases JSONB,

    embedding VECTOR({EMBEDDING_DIM}) NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS airline_embedding_hnsw_idx
ON airline_embeddings
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


def build_embedding_text(airline: dict) -> str:
    """
    Embedding is generated only from airline_name.
    """

    return airline["airline_name"].strip()


# ============================================================
# IMPORT
# ============================================================

def main():
    print(f"Loading airline data from: {DATA_FILE}")

    airlines = load_jsonl(DATA_FILE)

    print(f"Found {len(airlines)} airlines")

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

        print("Airline table/index ready.")

        # ----------------------------------------------------
        # Generate embeddings
        # ----------------------------------------------------

        texts = [
            build_embedding_text(airline)
            for airline in airlines
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

        print("Importing airlines...")

        with conn.cursor() as cur:

            for airline, embedding in zip(
                airlines,
                embeddings,
            ):
                cur.execute(
                    """
                    INSERT INTO airline_embeddings (
                        iata_code,
                        icao_code,
                        airline_name,
                        short_name,
                        aliases,
                        embedding
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (iata_code)
                    DO UPDATE SET
                        icao_code = EXCLUDED.icao_code,
                        airline_name = EXCLUDED.airline_name,
                        short_name = EXCLUDED.short_name,
                        aliases = EXCLUDED.aliases,
                        embedding = EXCLUDED.embedding,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        airline["iata_code"],
                        airline.get("icao_code"),
                        airline["airline_name"],
                        airline.get("short_name"),
                        json.dumps(
                            airline.get("aliases", []),
                            ensure_ascii=False,
                        ),
                        embedding,
                    ),
                )

        conn.commit()

    print("Airline import completed successfully.")


if __name__ == "__main__":
    main()