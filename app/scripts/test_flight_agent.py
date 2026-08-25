from app.src.agents.flight_agent import flight_agent


def test_flight_agent():
    print("--- Testing Flight Agent with Natural Language Input ---")

    input_data = {
        "messages": [
            {
                "role": "user",
                "content": "Tôi muốn tìm chuyến bay từ Sài Gòn đến Quy Nhơn ngày mai của hãng Việt Nam airline",
            }
        ]
    }
    result = flight_agent.process_request(input_data)
    
    print("\n--- KẾT QUẢ PHẢN HỒI ---")
    print(f"Thành công: {result.get('success')}")
    print(
        f"Câu trả lời:\n"
        f"{result.get('answer', 'Không có câu trả lời.')}"
    )

    print("\n--- Nuồn Dữ Liệu ---")

    sources = result.get("sources", [])

    if not sources:
        print("Không có nguồn tài liệu.")
        return

    for idx, src in enumerate(sources, 1):
        print(
            src
        )


if __name__ == "__main__":
    test_flight_agent()