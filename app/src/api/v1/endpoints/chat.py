from fastapi import APIRouter, HTTPException, Security
from fastapi.security import APIKeyHeader
import uuid
from app.src.services.base import base_service
from app.src.graphs.multi_agent_graph import multi_agent_graph
from app.src.config.settings import settings
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from typing import Any, Dict, List
import json
from datetime import datetime


router = APIRouter()

class ChatRequest(BaseModel):
    conversation_id: str = Field(..., description="ID của cuộc hội thoại (bắt buộc, đóng vai trò là user_id)")
    message: str = Field(..., description="Nội dung tin nhắn của người dùng")


class ChatMessageResponse(BaseModel):
    id: str
    role: str
    content: str
    message_type: str
    metadata: Dict[str, Any] = {}

class ChatResponse(BaseModel):
    success: bool
    conversation_id: str
    messages: List[ChatMessageResponse] = []


api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)

API_KEY = settings.CHAT_API_KEY

async def verify_api_key(
    api_key: str = Security(api_key_header),
):
    if not API_KEY or api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API Key",
        )

    return api_key

@router.post("", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest, api_key=Security(verify_api_key)):
    try:
        conv_id = payload.conversation_id
        user_message = payload.message

        if not conv_id:
            raise HTTPException(status_code=400, detail="conversation_id là bắt buộc.")

        conn = base_service.database
        with conn.cursor() as cur:
            # Kiểm tra conversation đã tồn tại dưới DB chưa (dựa vào user_id = conv_id)
            cur.execute("SELECT id FROM conversations WHERE user_id = %s;", (conv_id,))
            row = cur.fetchone()

            if not row:
                # Chưa có dưới DB thì tạo mới
                cur.execute(
                    "INSERT INTO conversations (id, user_id, created_at, updated_at) VALUES (%s, %s, NOW(), NOW());",
                    (str(uuid.uuid4()), conv_id)
                )

            # Lấy id thực sự của conversation từ bảng conversations dựa theo user_id (conv_id)
            cur.execute("SELECT id FROM conversations WHERE user_id = %s;", (conv_id,))
            conv_row = cur.fetchone()
            db_conversation_id = conv_row[0]

            # 2. Lấy lịch sử tin nhắn của conversation để cung cấp context cho Agent
            cur.execute(
                "SELECT role, content FROM messages WHERE conversation_id = %s ORDER BY created_at ASC;",
                (db_conversation_id,)
            )
            history_rows = cur.fetchall()
            
            langchain_messages = []
            for role, content in history_rows:
                if role == "user":
                    langchain_messages.append(HumanMessage(content=content))
                else:
                    langchain_messages.append(AIMessage(content=content))

            # Thêm tin nhắn mới nhất của user vào danh sách
            langchain_messages.append(HumanMessage(content=user_message))

            # 3. Lưu tin nhắn của user vào DB
            cur.execute(
                "INSERT INTO messages (id, conversation_id, role, content, message_type, created_at) VALUES (%s, %s, %s, %s, 'message', NOW());",
                (str(uuid.uuid4()), db_conversation_id, "user", user_message)
            )

            # 4. Gọi hệ thống Agent (multi_agent_graph)
            result_state = multi_agent_graph.invoke(
                    {
                        "messages": langchain_messages
                    },
                    config={
                        "callbacks": [
                            base_service.langfuse_handler
                        ]
                    }
                )

            
            # ============================================================
            # 5. Lấy response parts
            # ============================================================

            response_parts = result_state.get(
                "response_parts",
                []
            )


            # ============================================================
            # 6. Lưu từng part thành một message
            # ============================================================
            response_messages = []
            for part in response_parts:

                part_type = part.get("type")
                data = part.get("data", {})

                # ---------------------------------------------
                # Text
                # ---------------------------------------------

                if part_type == "text":

                    content = data.get("content", "")
                    metadata = {
                        "part_type": "text",
                        "data": data.get("sources", []),
                    }

                # ---------------------------------------------
                # Flight table
                # ---------------------------------------------

                elif part_type == "flight_table":
                    sources = data.get("sources", [])

                    depart_date = None

                    for source in sources:
                        if source.get("tool") == "tool_search_flights":

                            content_data = source.get("content", {})
                            data = content_data.get("data", {})
                            fare_data = data.get("fare_data", [])

                            if fare_data:
                                first_fare = fare_data[0]
                                flights = first_fare.get("flights", [])

                                if flights:
                                    start_date = flights[0].get("start_date")
                                    if start_date:
                                        depart_date = datetime.fromisoformat(
                                            start_date.replace("Z", "+00:00")
                                        ).strftime("%d-%m-%Y")

                            if depart_date:
                                break

                    
                    content = (
                        f"Danh sách các chuyến bay vào ngày {depart_date}"
                        if depart_date
                        else "Danh sách chuyến bay"
                    )

                    metadata = {
                        "part_type": "flight_table",
                        "data": sources,
                    }

                # ---------------------------------------------
                # Các loại part khác
                # ---------------------------------------------

                else:

                    content = data.get(
                        "content",
                        ""
                    )

                    metadata = {
                        "part_type": part_type,
                        "data": data,
                    }


                # 5. Lưu phản hồi của AI vào DB
                message_id = str(uuid.uuid4())
                cur.execute(
                    "INSERT INTO messages (id, conversation_id, role, content, message_type, metadata, created_at) VALUES (%s, %s, %s, %s, %s, %s, NOW());",
                    (message_id, db_conversation_id, "assistant", content, part_type, json.dumps(metadata,ensure_ascii=False))
                )

                # Build API response
                response_messages.append(
                    ChatMessageResponse(
                        id=message_id,
                        role="assistant",
                        content=content,
                        message_type=part_type,
                        metadata=metadata,
                    )
                )

            # Cập nhật updated_at cho conversation
            cur.execute(
                "UPDATE conversations SET updated_at = NOW() WHERE id = %s;",
                (db_conversation_id,)
            )
            conn.commit()

        return ChatResponse(
            success=True,
            conversation_id=conv_id,
            messages=response_messages,
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        conn.rollback()
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))
