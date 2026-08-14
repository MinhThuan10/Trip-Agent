from app.src.agents.flight_agent import process_flight_request

def test_flight_agent():
    print("--- Testing Flight Agent with Natural Language Input ---")
    executor = process_flight_request(user_message = "Tôi muốn tìm chuyến bay từ Sài Gòn đến Quy Nhơn ngày mai của hãng Việt Nam airline")
    
    
    print("\nAgent Response:")
    print(executor)

if __name__ == "__main__":
    test_flight_agent()
