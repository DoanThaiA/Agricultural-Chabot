import json
import traceback
from typing import AsyncGenerator, Optional
from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import HumanMessage
from database import get_db_session, Conversation, ChatMessage, DiseaseDetection
from graph import app as langgraph_app


class AgricultureChatbot:

    def __init__(self, db: AsyncSession = Depends(get_db_session)):
        self.db = db
        self.graph = langgraph_app

    # --- HÀM get_or_create_conversation --
    async def get_or_create_conversation(self, user_id: int, conversation_id: Optional[str], title: str) -> str:
        """Lấy hoặc tạo một cuộc hội thoại mới."""
        if conversation_id:
            convo = await self.db.get(Conversation, conversation_id)
            if convo and convo.user_id == user_id:
                return convo.id
            else:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                    detail="Conversation not found or access denied")
        convo_title = title[:50].strip() or "Hội thoại mới"
        convo = Conversation(user_id=user_id, title=convo_title)
        self.db.add(convo)
        await self.db.commit()
        await self.db.refresh(convo)
        return convo.id

    async def save_user_message(self, user_id: int, conversation_id: str, content: str):
        """Lưu tin nhắn của người dùng vào DB."""
        user_msg = ChatMessage(
            user_id=user_id,
            conversation_id=conversation_id,
            sender='user',
            content=content
        )
        self.db.add(user_msg)
        await self.db.commit()

    async def _parse_confidence(self, confidence_str: Optional[str]) -> Optional[float]:
        if not confidence_str:
            return None
        try:
            value = float(confidence_str.replace('%', '').strip())
            return round(value / 100, 3)
        except ValueError:
            return None

    async def process_query(self, user_id: int, user_query: str, conversation_id: str,
                            image_data: Optional[str] = None) -> \
            AsyncGenerator[str, None]:
        """
        Xử lý truy vấn, CHỜ KẾT QUẢ CUỐI CÙNG, lưu vào DB và trả về.
        """

        # 1. LƯU TIN NHẮN NGƯỜI DÙNG
        try:

            await self.save_user_message(user_id, conversation_id, user_query or "[Image Sent]")
        except Exception as db_error:
            print(f"\n--- LỖI NGHIÊM TRỌNG KHI LƯU TIN NHẮN NGƯỜI DÙNG ---")
            traceback.print_exc()
            yield f"data: {json.dumps({'event': 'error', 'detail': 'Không thể lưu tin nhắn người dùng.'})}\n\n"
            return

        if not self.graph:
            yield f"data: {json.dumps({'event': 'error', 'detail': 'Chatbot service không khả dụng.'})}\n\n"
            return

        config = {"configurable": {"thread_id": conversation_id}}
        inputs = {
            "image_data": image_data,
            "messages": [HumanMessage(content=user_query)]
        }

        final_bot_response = ""
        final_state = None

        try:
            print("Đang gọi graph.ainvoke...")
            final_state = await self.graph.ainvoke(inputs, config)
            print("Graph đã chạy xong.")

            if final_state:
                if final_state.get("raw_output"):
                    final_bot_response = final_state.get("raw_output")
                elif final_state.get("messages"):
                    # Lấy tin nhắn cuối cùng (thường là của AIMessage)
                    final_bot_response = final_state["messages"][-1].content
                else:
                    final_bot_response = "Lỗi: Không nhận được phản hồi từ bot."
            else:
                final_bot_response = "Lỗi: Graph không trả về state."

            bot_msg_id = None

            # 1. Thêm tin nhắn của bot vào session
            bot_msg = ChatMessage(
                user_id=user_id,
                conversation_id=conversation_id,
                sender='bot',
                content=final_bot_response
            )
            self.db.add(bot_msg)

            # 2. GIAO DỊCH 1: COMMIT TIN NHẮN BOT
            try:
                await self.db.commit()
                await self.db.refresh(bot_msg)
                bot_msg_id = bot_msg.id
            except Exception as commit_error:
                print(f"\n--- LỖI NGHIÊM TRỌNG KHI COMMIT TIN NHẮN BOT ---")
                await self.db.rollback()
                raise commit_error

            # 3. GIAO DỊCH 2: LƯU DETECTION
            try:
                if final_state and bot_msg_id:
                    query_type = final_state.get('query_type')
                    info = final_state.get('disease_info')

                    if query_type == 'image_disease' and info:
                        disease_name = info.get('disease_detected')
                        if disease_name and disease_name not in ["Analysis inconclusive", "Error processing image"]:
                            confidence_float = 0.0
                            confidence_str = info.get('confidence')
                            if hasattr(self, '_parse_confidence'):
                                confidence_float = await self._parse_confidence(confidence_str)
                            elif confidence_str:
                                try:
                                    confidence_float = float(confidence_str.strip('%')) / 100.0
                                except:
                                    pass

                            detection = DiseaseDetection(
                                message_id=bot_msg_id,
                                disease_name=disease_name,
                                confidence=confidence_float
                            )
                            self.db.add(detection)
                            await self.db.commit()
            except Exception as detection_error:
                print(f"\n--- LỖI (Đã bỏ qua) KHI LƯU DISEASE DETECTION ---")
                traceback.print_exc()
                await self.db.rollback()

            # 4. GỬI SỰ KIỆN KẾT THÚC
            yield f"data: {json.dumps({'event': 'end', 'final_message': final_bot_response, 'conversation_id': conversation_id})}\n\n"

        except Exception as e:
            print(f"\n--- LỖI NGHIÊM TRỌNG TRONG process_query (Graph hoặc DB Error) ---")
            traceback.print_exc()
            await self.db.rollback()
            error_detail = f"Lỗi server: {type(e).__name__}"
            yield f"data: {json.dumps({'event': 'error', 'detail': error_detail})}\n\n"