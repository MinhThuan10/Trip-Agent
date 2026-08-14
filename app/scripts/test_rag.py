import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.src.services.rag_service import rag_service

def main():
    print("=== SCRIPT TEST RAG QUERY ===")
    
    query = input("Nhập câu hỏi test: ").strip()
    if not query:
        print("Câu hỏi không được để trống.")
        return

    use_filter = input("Bạn có muốn lọc theo category hoặc file_name không? (y/n): ").strip().lower()
    filter_dict = None

    if use_filter == 'y':
        filter_dict = {}
        cat = input("Nhập category cần lọc (hoặc để trống nếu bỏ qua): ").strip()
        if cat:
            filter_dict["category"] = cat
        
        fname = input("Nhập tên file (file_name) cần lọc (hoặc để trống nếu bỏ qua): ").strip()
        if fname:
            filter_dict["file_name"] = fname
        
        if not filter_dict:
            filter_dict = None

    print(f"\nĐang tìm kiếm với query: '{query}' và filter: {filter_dict}...")
    try:
        results = rag_service.similarity_search(query=query, k=3, filter_metadata=filter_dict)
        
        if not results:
            print("Không tìm thấy tài liệu phù hợp.")
            return

        print(f"\nTìm thấy {len(results)} kết quả phù hợp nhất:\n")
        for idx, doc in enumerate(results, 1):
            print(f"--- Kết quả {idx} ---")
            print(f"Nội dung: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
            print("-" * 30)

    except Exception as e:
        print(f"Lỗi khi thực hiện tìm kiếm RAG: {str(e)}")

if __name__ == "__main__":
    main()
