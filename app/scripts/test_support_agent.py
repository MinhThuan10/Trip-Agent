import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from app.src.agents.support_agent import support_rag_agent


def main():
    print("=== TEST SUPPORT RAG AGENT ===")

    query = input(
        "Nhập câu hỏi hỗ trợ cho khách hàng: "
    ).strip()

    if not query:
        print("Câu hỏi không được để trống.")
        return

    print(
        "\nĐang tìm kiếm (top 10), rerank (top 4), "
        "build context và gọi LLM..."
    )

    # invoke() yêu cầu input_data có dạng:
    # {
    #     "messages": [...]
    # }
    input_data = {
        "messages": [
            {
                "role": "user",
                "content": query,
            }
        ]
    }

    result = support_rag_agent.invoke(input_data)

    print("\n--- KẾT QUẢ PHẢN HỒI ---")
    print(f"Thành công: {result.get('success')}")
    print(
        f"Câu trả lời:\n"
        f"{result.get('answer', 'Không có câu trả lời.')}"
    )

    print("\n--- NGUỒN TÀI LIỆU SỬ DỤNG ---")

    sources = result.get("sources", [])

    if not sources:
        print("Không có nguồn tài liệu.")
        return

    for idx, src in enumerate(sources, 1):
        print(
            f"{idx}. "
            f"File: {src.get('file_name')} | "
            f"Category: {src.get('category')}"
        )


if __name__ == "__main__":
    main()