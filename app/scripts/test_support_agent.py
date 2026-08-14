import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.src.agents.support_agent import support_rag_agent
from sentence_transformers import CrossEncoder
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
def main():
    print("=== TEST SUPPORT RAG AGENT ===")
    query = input("Nhập câu hỏi hỗ trợ cho khách hàng: ").strip()
    if not query:
        print("Câu hỏi không được để trống.")
        return

    print("\nĐang tìm kiếm (top 10), rerank (top 4), build context và gọi LLM...")
    result = support_rag_agent.answer_query(query)

    print("\n--- KẾT QUẢ PHẢN HỒI ---")
    print(f"Thành công: {result['success']}")
    print(f"Câu trả lời:\n{result['answer']}")
    print("\n--- NGUỒN TÀI LIỆU SỬ DỤNG ---")
    for idx, src in enumerate(result['sources'], 1):
        print(f"{idx}. File: {src['file_name']} | Category: {src['category']}")



if __name__ == "__main__":
    main()
