from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from .schemas import GenerateRequest

router = APIRouter()

# 글로벌 변수로 모델 초기화
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
llm = Ollama(model="llama-3-Korean-Bllossom-8B-gguf-Q4_K_M")
vectorstore = None


def load_pdf_index(pdf_name: str):
    global vectorstore
    vectorstore = FAISS.load_local(
        f"index/faiss_index_pdf/{pdf_name}.index",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )
    print("Index loaded")


@router.post("/generate/")
async def generate(request: GenerateRequest):
    question = request.question
    pdf_name = request.pdf_name

    # PDF 인덱스 로드
    load_pdf_index(pdf_name)

    # 벡터 스토어가 로드되었는지 확인
    if vectorstore is None:
        raise HTTPException(status_code=400, detail="PDF index is not loaded")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

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
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )

    chain = RunnableSequence(
        {
            "document": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)

    return {"response": response}
