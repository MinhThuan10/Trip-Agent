import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.src.services.rag_service import rag_service

def main():
    print("=== SCRIPT IMPORT PDF CHO RAG ===")
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = input("Nhập đường dẫn file PDF cần import: ").strip()

    if not os.path.exists(pdf_path):
        print(f"Lỗi: Không tìm thấy file tại đường dẫn: {pdf_path}")
        return

    category = input("Nhập danh mục (category) cho tài liệu (ví dụ: chinh_sach, huong_dan, quy_dinh): ").strip()
    if not category:
        category = "general"

    file_name = os.path.basename(pdf_path)
    print(f"Đang xử lý file: {file_name} với category: {category}...")

    try:
        num_chunks = rag_service.process_and_store_pdf(
            pdf_path=pdf_path,
            category=category,
            file_name=file_name
        )
        print(f"Thành công! Đã chia và lưu {num_chunks} chunks vào cơ sở dữ liệu VectorDB.")
    except Exception as e:
        print(f"Lỗi trong quá trình xử lý và import PDF: {str(e)}")

if __name__ == "__main__":
    main()
