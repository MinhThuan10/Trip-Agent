from langchain_core.messages import HumanMessage

from app.src.graphs.multi_agent_graph import multi_agent_graph
from app.src.services.base import base_service


def run_test():
    queries = [
        # "Xin chào bạn",
        # "Thông tin vé máy bay trẻ em như nào?",
        "Tôi muốn tìm chuyến bay từ Sài Gòn đến Quy Nhơn ngày mai của Vietnam Airlines",
    ]

    for q in queries:

        print("\n" + "=" * 80)
        print(f"USER: {q}")
        print("=" * 80)

        result = multi_agent_graph.invoke(
            {
                "messages": [
                    HumanMessage(content=q)
                ],
                "worker_results": {},
                "completed_workers": [],
                "iteration": 0,
            },
            config={
                "callbacks": [
                    base_service.langfuse_handler
                ]
            },
        )

        # =====================================================
        # WORKFLOW INFO
        # =====================================================

        print("\n--- Workflow ---")

        print(
            f"Current worker : "
            f"{result.get('current_worker')}"
        )

        print(
            f"Iteration      : "
            f"{result.get('iteration')}"
        )

        print(
            f"Completed      : "
            f"{result.get('completed_workers')}"
        )

        # =====================================================
        # WORKER RESULTS
        # =====================================================

        print("\n--- Worker Results ---")

        worker_results = result.get(
            "worker_results",
            {}
        )

        if not worker_results:
            print("Không có worker nào được gọi.")

        else:
            for worker_name, worker_result in worker_results.items():

                print(f"\n[{worker_name}]")

                print(worker_result)

        # =====================================================
        # FINAL RESPONSE
        # =====================================================

        print("\n--- Response Parts ---")

        response_parts = result.get("response_parts", [])

        print("\n--- Response Parts ---")

        for index, part in enumerate(response_parts, 1):
            print(f"\nPart {index}")
            print(f"Type: {part.get('type')}")
            print(f"Data: {part.get('data')}")

        print("\n" + "-" * 80)

    print(
        "\nĐã test và gửi trace lên Langfuse thành công!"
    )


if __name__ == "__main__":
    run_test()