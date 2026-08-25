from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse
import os
import shutil
from app.src.services.rag_service import rag_service
import psycopg2
from app.src.config.settings import settings
import json

router = APIRouter()

AIRPORTS_FILE = os.path.join(os.path.dirname(__file__), "../../../../data/airports.jsonl")
AIRLINES_FILE = os.path.join(os.path.dirname(__file__), "../../../../data/airlines.jsonl")



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

@router.get("/rag/documents")
def get_rag_documents():
    try:
        with psycopg2.connect(settings.DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        cmetadata->>'category' as category,
                        cmetadata->>'file_name' as file_name,
                        COUNT(*) as chunk_count
                    FROM langchain_pg_embedding
                    GROUP BY cmetadata->>'category', cmetadata->>'file_name'
                    ORDER BY category, file_name;
                """)
                rows = cur.fetchall()
                docs = [{"category": r[0] or "general", "file_name": r[1] or "unknown", "chunk_count": r[2]} for r in rows]
                
                cur.execute("SELECT DISTINCT cmetadata->>'category' FROM langchain_pg_embedding WHERE cmetadata->>'category' IS NOT NULL;")
                categories = [r[0] for r in cur.fetchall() if r[0]]
                if not categories:
                    categories = ["general"]
                    
                return {"success": True, "documents": docs, "categories": categories}
    except Exception as e:
        return {"success": True, "documents": [], "categories": ["general"]}

@router.post("/rag/upload")
async def upload_rag_document(
    file: UploadFile = File(...),
    category: str = Form(...)
):
    # Ràng buộc chặt chẽ chỉ cho phép file PDF (kiểm tra đuôi file và content type nếu có)
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Chỉ cho phép tải lên file định dạng PDF (.pdf).")
    
    os.makedirs("app/data/uploads", exist_ok=True)
    file_path = os.path.join("app/data/uploads", file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        num_chunks = rag_service.process_and_store_pdf(
            pdf_path=file_path,
            category=category,
            file_name=file.filename
        )
        return {"success": True, "message": f"Đã upload và chia thành {num_chunks} chunks thành công.", "chunk_count": num_chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rag/download")
def download_rag_document(file_name: str = Query(...)):
    file_path = os.path.join("app/data/uploads", file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Không tìm thấy file trên hệ thống.")
    return FileResponse(path=file_path, filename=file_name, media_type='application/pdf')

@router.delete("/rag/documents")
def delete_rag_document(file_name: str = Query(...)):
    try:
        # 1. Xóa file vật lý trong app/data/uploads (nếu tồn tại)
        file_path = os.path.join("app/data/uploads", file_name)
        if os.path.exists(file_path):
            os.remove(file_path)

        # 2. Xóa các embedding chunks trong PostgreSQL (langchain_pg_embedding)
        with psycopg2.connect(settings.DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM langchain_pg_embedding WHERE cmetadata->>'file_name' = %s;",
                    (file_name,)
                )
                deleted_count = cur.rowcount
            conn.commit()

        return {
            "success": True,
            "message": f"Đã xóa tài liệu {file_name} và {deleted_count} vector chunks khỏi cơ sở dữ liệu."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
