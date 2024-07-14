# converters.py 또는 기존 파일에 추가
from datetime import datetime
from langchain_core.messages.base import BaseMessage
from .models import Message as DBMessage


def db_message_to_base_message(db_message: DBMessage) -> BaseMessage:
    # 데이터베이스 메시지를 BaseMessage 형태로 변환
    return BaseMessage(
        content=db_message.content,
        id=str(db_message.id),
        additional_kwargs={
            "sender": db_message.sender,
            "conversation_id": db_message.conversation_id,
            "timestamp": db_message.timestamp.isoformat(),
        },
    )


def base_message_to_db_message(
    base_message: BaseMessage, conversation_id: int, sender: str
) -> DBMessage:
    # BaseMessage를 데이터베이스 메시지 형태로 변환
    return DBMessage(
        conversation_id=conversation_id,
        sender=sender,
        content=base_message.content,
        timestamp=datetime.fromisoformat(
            base_message.additional_kwargs.get("timestamp")
        ),
    )
