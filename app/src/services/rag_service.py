import os
from typing import List, Optional
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.src.services.base import base_service
from sqlalchemy import create_engine
from app.src.config.settings import settings

class RAGService:
    def __init__(self):
        self.embedding_model = base_service.embedding_model
        # Kết nối tới cơ sở dữ liệu postgresql với pgvector
        self.connection_string = settings.DATABASE_URL
        self.collection_name = "customer_support_docs"

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Trích xuất toàn bộ nội dung văn bản từ file PDF."""
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    def process_and_store_pdf(self, pdf_path: str, category: str, file_name: Optional[str] = None) -> int:
        """
        Đọc PDF, chunking với chunk_size=512, chunk_overlap=100,
        tạo embedding và lưu vào PGVector kèm filter metadata (file_name, category).
        """
        if not file_name:
            file_name = os.path.basename(pdf_path)

        # 1. Trích xuất text
        raw_text = self.extract_text_from_pdf(pdf_path)
        if not raw_text.strip():
            raise ValueError(f"Không thể trích xuất nội dung từ file PDF: {pdf_path}")

        # 2. Chunking dữ liệu (chunk_size=512, chunk_overlap=100)
        # Sử dụng Character-based hoặc Token-based tùy theo splitter. RecursiveCharacterTextSplitter hoạt động tốt với ký tự/token tương đương.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )

        chunks = text_splitter.split_text(raw_text)

        # 3. Tạo Document object kèm metadata
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={
                    "file_name": file_name,
                    "category": category,
                }
            )
            documents.append(doc)

        # 4. Lưu vào PGVector
        # PGVector.from_documents sẽ tự tạo bảng nếu chưa có
        base_service.vector_store.add_documents(documents)

        return len(documents)

    def similarity_search(self, query: str, k: int = 4, filter_metadata: Optional[dict] = None) -> List[Document]:
        """
        Tìm kiếm tương đồng với filter tùy chọn (ví dụ: {"category": "ban_ve", "file_name": "chinh_sach.pdf"})
        """

       
        # PGVector hỗ trợ filter qua collection hoặc retriever/similarity_search với filter
        results = base_service.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter_metadata
        )
        return results

rag_service = RAGService()
