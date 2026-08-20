from typing import List, TypedDict, Any, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from app.src.agents.flight_agent import flight_agent
from app.src.agents.support_agent import support_rag_agent
from app.src.services.base import base_service
from pydantic import BaseModel, Field

class State(TypedDict):
    messages: List[Any]
    plan: List[str]
    current_worker: str
    worker_results: dict
    iteration: int
    completed_workers: list[str]
    response_parts: list[dict[str, Any]]

class OrchestratorPlan(BaseModel):
    plan_steps: List[str] = Field(description="Các bước thực hiện công việc")
    worker_to_call: Literal["flight_agent", "support_agent", "synthesize"] = Field(
        description="Worker tiếp theo cần gọi: flight_agent cho vé máy bay, support_agent cho quy định/hỗ trợ/chính sách/tra cứu, synthesize khi đã hoàn thành để trả lời"
    )
    instruction_for_worker: str = Field(
        description="Hướng dẫn cụ thể cho worker được gọi"
    )


class ResponsePart(BaseModel):
    type: Literal[
        "text",
        "flight_table",
        "policy_table",
        "action"
    ]
    data: dict[str, Any]


class AgentResponse(BaseModel):
    parts: list[ResponsePart]



MAX_ITERATIONS = 5


def orchestrator_node(state: State):
    messages = state.get("messages", [])
    worker_results = state.get("worker_results", {})
    completed_workers = set(state.get("completed_workers", []))

    iteration = state.get("iteration", 0) + 1

    # Safety guard
    if iteration >= MAX_ITERATIONS:
        return {
            "current_worker": "synthesize",
            "iteration": iteration,
        }

    prompt = f"""
Bạn là Orchestrator (Điều phối viên chính) của hệ thống đặt vé.

Nhiệm vụ:
Phân tích yêu cầu người dùng và điều phối worker phù hợp.

Các worker:
- `flight_agent`: tìm kiếm chuyến bay, giá vé, lịch trình.
- `support_agent`: quy định, dịch vụ, chính sách và hỗ trợ khách hàng.
- `synthesize`: tổng hợp kết quả cuối cùng và trả lời người dùng.

Kết quả các worker đã chạy:
{worker_results}

Các worker ĐÃ HOÀN THÀNH:
{list(completed_workers)}

QUY TẮC:
1. Không gọi lại worker đã nằm trong danh sách "ĐÃ HOÀN THÀNH".
2. Worker được xem là hoàn thành ngay cả khi:
   - không tìm thấy thông tin
   - không đủ thông tin
   - không có kết quả
   - trả về lỗi.
3. Nếu đã có đủ thông tin để trả lời → chọn `synthesize`.
4. Nếu không còn worker phù hợp chưa chạy → chọn `synthesize`.
5. Chỉ gọi worker chưa hoàn thành khi thực sự cần thiết.

Hãy quyết định bước tiếp theo.
"""

    llm = base_service.llm.with_structured_output(
        OrchestratorPlan
    )

    response = llm.invoke(
        [
            {"role": "system", "content": prompt},
            *messages,
        ],
        config={
            "callbacks": [base_service.langfuse_handler]
        }
    )

    next_worker = response.worker_to_call

    # ==================================================
    # HARD GUARD
    # ==================================================

    if next_worker in completed_workers:
        next_worker = "synthesize"

    return {
        "current_worker": next_worker,
        "plan_steps": response.plan_steps,
        "iteration": iteration,
    }

def call_flight_worker(state: State):
    messages = state["messages"]
    result = flight_agent.process_request({"messages": messages})

    completed = set(state.get("completed_workers", []))
    completed.add("flight_agent")

    return {
        "worker_results": {
            **state.get("worker_results", {}),
            "flight_agent": result,
        },
        "completed_workers": list(completed),
    }

def call_support_worker(state: State):
    messages = state["messages"]
    result = support_rag_agent.invoke({"messages": messages})
    
    completed = set(state.get("completed_workers", []))
    completed.add("support_agent")

    return {
        "worker_results": {
            **state.get("worker_results", {}),
            "support_agent": result,
        },
        "completed_workers": list(completed),
    }



def synthesize_node(state: State):

    worker_results = state.get("worker_results", {})

    parts = []

    # ============================================================
    # Không có worker result
    # ============================================================
    if not worker_results:
        parts.append(
            ResponsePart(
                type="text",
                data={
                    "content": (
                        "Mình là trợ lý ảo Trip, rất vui được trò chuyện "
                        "với bạn 💗. Trip có thể hỗ trợ gì cho bạn hôm nay ạ?"
                    ),
                    "sources": [],
                },
            )
        )

        response = AgentResponse(parts=parts)

        return {
            "response_parts": response.model_dump()["parts"]
        }

    # ============================================================
    # SUPPORT AGENT
    # ============================================================
    support_result = worker_results.get("support_agent")

    if support_result and support_result.get("success"):

        for answer in support_result.get("answer", []):

            if answer.get("type") == "text":

                parts.append(
                    ResponsePart(
                        type="text",
                        data={
                            "content": answer.get("text", ""),
                            "sources": support_result.get(
                                "sources", []
                            ),
                        },
                    )
                )

    # ============================================================
    # FLIGHT AGENT
    # ============================================================
    flight_result = worker_results.get("flight_agent")

    if flight_result and flight_result.get("success"):

        # --------------------------------------------------------
        # 1. Natural language response
        # --------------------------------------------------------
        for answer in flight_result.get("answer", []):

            if answer.get("type") == "text":

                parts.append(
                    ResponsePart(
                        type="text",
                        data={
                            "content": answer.get("text", ""),
                            "sources": [],
                        },
                    )
                )

        # --------------------------------------------------------
        # 2. Flight data từ API
        # --------------------------------------------------------
        flight_sources = flight_result.get("sources", [])

        if flight_sources:

            parts.append(
                ResponsePart(
                    type="flight_table",
                    data={
                        "sources": flight_sources,
                    },
                )
            )

    # ============================================================
    # FINAL RESPONSE
    # ============================================================

    response = AgentResponse(parts=parts)

    return {
        "response_parts": response.model_dump()["parts"]
    }

def route_from_orchestrator(state: State):
    worker = state["current_worker"]

    completed = set(
        state.get("completed_workers", [])
    )

    if worker in completed:
        return "synthesize"

    return worker

workflow = StateGraph(State)
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("flight_agent", call_flight_worker)
workflow.add_node("support_agent", call_support_worker)
workflow.add_node("synthesize", synthesize_node)

workflow.add_edge(START, "orchestrator")

workflow.add_conditional_edges(
    "orchestrator",
    route_from_orchestrator,
    {
        "flight_agent": "flight_agent",
        "support_agent": "support_agent",
        "synthesize": "synthesize",
    }
)

workflow.add_edge("flight_agent", "orchestrator")
workflow.add_edge("support_agent", "orchestrator")
workflow.add_edge("synthesize", END)

multi_agent_graph = workflow.compile()
