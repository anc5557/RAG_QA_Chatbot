# llm.py
from fastapi import APIRouter, HTTPException, Depends, FastAPI
from sqlalchemy.orm import Session
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.chat_message_histories import SQLChatMessageHistory

from app.schemas import GenerateRequest
from .database import get_db
from .models import Conversation, Message


router = APIRouter()

# 기본 PDF 이름 설정
DEFAULT_PDF_NAME = "CM9_manual"
MODEL_NAME = "llama-3-Korean-Bllossom-8B-gguf-Q4_K_M"
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"
embedding_model = None
llm = None
vectorstore = None


@router.on_event("startup")
async def startup_event():
    global embedding_model, llm, vectorstore

    # 모델 및 벡터 스토어 초기화
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    llm = Ollama(model=MODEL_NAME)
    vectorstore = FAISS.load_local(
        f"index/faiss_index_pdf/{DEFAULT_PDF_NAME}.index",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )
    print("Models and index loaded")


def get_memory(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id, connection_string="sqlite:///./chatbot.db"
    )


@router.post("/generate/")
async def generate(request: GenerateRequest, db: Session = Depends(get_db)):
    question = request.question
    session_id = request.session_id

    # 벡터 스토어가 로드되었는지 확인
    if vectorstore is None:
        raise HTTPException(status_code=400, detail="PDF index is not loaded")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    memory = get_memory(session_id)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                """시스템 메시지:
                당신은 크리니티 Q&A 챗봇입니다. 호기심 많은 사용자와 인공지능 어시스턴트 간의 대화입니다. 어시스턴트는 사용자의 질문에 대해 도움이 되고, 간략한 답변을 제공합니다.
                사용자의 질문에 정확한 답변을 문서에 기반하여 제공해주세요. 문서에 없는 정보는 만들어내지 마세요.
                다음의 검색된 문서를 사용하여 질문에 답변해주세요. 만약 답을 모른다면, 모른다고 말해주세요. 한국어로 답변해주세요.
                
                문서입니다.
                {document}
                """
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )

    chain = RunnableSequence(
        {"document": retriever, "question": RunnablePassthrough(), "history": memory}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke({"question": question, "history": memory.messages})

    return {"response": response}


app = FastAPI()
app.include_router(router)
