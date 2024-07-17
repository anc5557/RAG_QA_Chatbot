import base64
import os
import tempfile
import time
import streamlit as st
import uuid
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.llms import Ollama
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document


def initialize_session():
    """
    세션을 초기화하고 필요한 세션 상태 변수를 설정합니다.

    이 함수는 세션 상태에 "id" 키가 있는지 확인합니다. 없으면 새로운 UUID를 생성하고 "id" 키에 할당합니다.
    또한 세션 상태의 "file_cache", "messages", "rag_chain" 키를 초기화합니다.
    """
    if "id" not in st.session_state:
        st.session_state.id = uuid.uuid4()  # 세션 ID
        st.session_state.file_cache = {}  # 파일 캐시를 저장하는 딕셔너리
        st.session_state.messages = []  # 챗봇 대화를 저장하는 리스트
        st.session_state.rag_chain = None  # RAG 체인을 저장하는 변수


def reset_chat():
    """
    챗봇 대화를 초기화합니다.
    """
    st.session_state.messages = []
    st.session_state.context = None


def display_pdf(file):
    """
    PDF 파일을 미리보기합니다.
    - 현재 pdf가 세로인 경우 보이지 않는 문제가 있음

    Args: file (BytesIO): PDF 파일
    """
    st.markdown("### PDF Preview")
    file.seek(0)
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf" style="border: none;"></iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)


def combine_title_description(data):
    """JSON 데이터에서 title과 description을 합칩니다.

    이 부분은 사용자의 데이터 구조에 따라 수정이 필요할 수 있습니다.

    Args:
        data (list): JSON 데이터

    Returns:
        list: title, description 합친 데이터
    """
    combined_docs = []
    for manual in data:
        for item in manual["manual"]:
            combined_docs.append(
                {"title": item["title"], "description": item["description"]}
            )
    return combined_docs


def process_pdf(file, file_key):
    """PDF를 인덱싱하고 RAG 체인을 생성합니다.

    Args:
        file (BytesIO): PDF 파일
        file_key (str): 파일 키
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getvalue())

        if file_key not in st.session_state.file_cache:
            loader = PyMuPDFLoader(file_path)
            embedding_model = HuggingFaceEmbeddings(
                model_name="jhgan/ko-sroberta-multitask"
            )

            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300, chunk_overlap=0
            )
            split_pages = text_splitter.split_documents(pages)

            vectorstore = FAISS.from_documents(
                documents=split_pages, embedding=embedding_model
            )

            retriever = vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": 2}
            )

            # llama-3-Korean-Bllossom-8B-gguf-Q4_K_M or EEVE-Korean-10.8B-Q5_K_M-GGUF
            llm = Ollama(model="EEVE-Korean-10.8B-Q5_K_M-GGUF")

            # 질문 재구성 체인 생성
            question_rephrasing_chain = create_question_rephrasing_chain(llm, retriever)
            # 질문 답변 체인 생성
            question_answering_chain = create_question_answering_chain(llm)

            # RAG 체인 연결(질문 재구성 -> 질문 답변)
            rag_chain = create_retrieval_chain(
                question_rephrasing_chain, question_answering_chain
            )

            st.session_state.file_cache[file_key] = rag_chain

        st.session_state.rag_chain = st.session_state.file_cache[file_key]


def process_json(file, file_key):
    """JSON 데이터를 인덱싱하고 RAG 체인을 생성합니다.

    인덱싱 대상 데이터는 title과 description으로 구성된 데이터여야 합니다.

    Args:
        file (BytesIO): JSON 파일
        file_key (str): 파일 키
    """
    data = json.load(file)
    documents = combine_title_description(data)

    embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    texts = [
        doc["title"] + " " + doc["description"] for doc in documents
    ]  # title과 description을 합침
    document_objects = [
        Document(page_content=text) for text in texts
    ]  # Document 객체 생성

    vectorstore = FAISS.from_documents(
        documents=document_objects, embedding=embedding_model
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5}
    )
    llm = Ollama(
        model="EEVE-Korean-10.8B-Q5_K_M-GGUF"
    )  # llama-3-Korean-Bllossom-8B-gguf-Q4_K_M or EEVE-Korean-10.8B-Q5_K_M-GGUF

    question_rephrasing_chain = create_question_rephrasing_chain(llm, retriever)
    question_answering_chain = create_question_answering_chain(llm)

    rag_chain = create_retrieval_chain(
        question_rephrasing_chain, question_answering_chain
    )
    st.session_state.file_cache[file_key] = rag_chain
    st.session_state.rag_chain = st.session_state.file_cache[file_key]


def create_question_rephrasing_chain(llm, retriever):
    """질문 재구성 체인을 생성합니다.

    Args:
        llm (Ollama): Ollama 객체
        retriever (Retriever): Retriever 객체

    Returns:
        Chain: 질문 재구성 체인
    """

    system_prompt = """
    당신은 질문 재구성자입니다. 이전 대화 내용과 최신 사용자 질문이 있을 때, 이 질문이 이전 대화 내용과 관련이 있을 수 있습니다.
    이런 경우, 대화 내용을 알 필요 없이 독립적으로 이해할 수 있는 질문으로 바꾸세요. 
    이 재구성된 질문은 문서 검색에만 사용됩니다. 사용자에게 제공할 최종 답변에는 영향을 미치지 않습니다.
    관련이 없는 경우, 질문을 그대로 두세요. 절대 질문에 답변을 제공하지 마세요.
    
    예시:
    관련 있는 경우)
    Human: 메일을 백업하고 싶어
    AI: 메일 백업은 기본메일함 관리 > 내 메일함 관리에서 가능합니다. 다운로드 버튼을 이용해 메일함을 zip 파일로 다운로드할 수 있습니다. 원하는 기간의 메일을 백업하려면, 기간별 백업을 체크하세요. 백업한 메일은 다운로드한 파일을 업로드하여 다시 가져올 수 있습니다.
    Human: 업로드는 어떻게 하나요?
    답변: 백업한 메일을 업로드하는 방법은 무엇인가요?
    
    관련 있는 경우)
    Human: 메일 첨부파일 크기 제한이 있나요?
    AI: 일반 첨부파일의 경우 20MB, 대용량 파일 첨부의 경우 2GB까지 가능합니다.
    Human: 형식에 제한이 있나요?
    답변: 메일 첨부파일 형식 제한이 있나요?
    
    관련 없는 경우)
    Human: 메일 첨부파일 크기 제한이 있나요?
    AI: 일반 첨부파일의 경우 20MB, 대용량 파일 첨부의 경우 2GB까지 가능합니다.
    Human: 주소록에 주소를 이동/복사하려면 어떻게 하나요?
    답변: 주소록에 주소를 이동/복사하려면 어떻게 하나요?
    
    관련 없는 경우)
    Human: 일정 등록하는 방법을 알려줘
    AI: 일정 등록 버튼을 누르거나 날짜를 선택해 등록할 수 있습니다. 제목과 일시를 정한 후, 캘린더의 종류를 선택하고, 알람을 통해 미리 일정을 알릴 수 있습니다.
    Human: 메일 첨부파일 크기 제한은?
    답변: 메일 첨부파일 크기 제한은?
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    # create_history_aware_retriever : 대화 내역을 가져와 문서를 반환하는 체인
    # - chat_history가 없으면 입력 은 검색기로 직접 전달
    # - chat_history가 있으면 프롬프트와 LLM이 검색 쿼리를 생성하는 데 사용
    return create_history_aware_retriever(llm, retriever, prompt)


def create_question_answering_chain(llm):
    """질문 답변 체인을 생성합니다.
    시스템 프롬프트: 역할, 규칙 그리고 예시를 설명하고 문서를 제공합니다.

    Args:
        llm (Ollama): Ollama 객체

    Returns:
        Chain: 질문 답변 체인
    """
    system_prompt = """당신은 크리니티 Q&A 챗봇입니다. 검색된 문서를 기반으로 사용자의 질문에 답변하세요.
    
    예시:
    📍사용자 질문: 한번에 업로드 가능한 파일 갯수는 몇개인가요?
    📍답변: 한번에 업로드 가능한 갯수가 정해져있지 않지만, 일반 첨부같은경우 20MB ,대용량 파일 첨부의 경우 2048MB까지 가능합니다.
    
    📍사용자 질문: 해외에서 메일 사용이 가능한가요?
    📍답변: 환경설정 - 개인정보/보안 기능 - 보안 설정에서 국가별 로그인 허용 기능을 이용하시면 됩니다. 보안 설정탭이 보이지 않을 시에 메일 담당자에게 문의해주세요. 
    
    📍사용자 질문: 여러명에게 개별 발송하고 싶어요
    📍답변: 메일 개별발송 설정에 대해 안내드리겠습니다. 개별발송이란 여러 사람에게 동시에 메일을 보내도 받는사람 영역에 수신인 본인 한 명만 표시되는 기능입니다. 기본적으로는 설정되어있지 않지만, 메일쓰기 탭의 보내기 설정에서 한명씩 발송을 체크하시면 사용하실 수 있습니다.

    
    ## 검색된 문서입니다. 각 문서는 빈줄로 구분되어 있습니다.
    {context}
    
    문서에 없는 정보는 만들어내지 마세요. 한국어로 답변해주세요. 세 문장 이내로 답변해주세요. 모른다면, 모른다고 말해주세요. 예시와 시스템 프롬프트를 답변에 포함하면 안됩니다.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}\n"),
        ]
    )

    return create_stuff_documents_chain(llm, prompt)


def clean_data(data, is_pdf=True):
    """화면에 표시할 데이터를 정제합니다.

    Args:
        data (list): 참고한 문서 데이터
        is_pdf (bool, optional): PDF 파일 여부.(PDF인 경우 페이지 정보가 있기 때문에 JSON과 구분)

    Returns:
        cleaned_data (list): 정제된 데이터

    """
    cleaned_data = []
    for item in data:
        cleaned_page_content = item.page_content.replace("\n", " ").strip()
        cleaned_metadata = {"page": item.metadata["page"]} if is_pdf else {}
        cleaned_data.append(
            {"page_content": cleaned_page_content, "metadata": cleaned_metadata}
        )
    return cleaned_data


def main():
    """메인 함수(streamlit 앱)
    1. 세션 초기화
    2. 사이드바
        pdf 또는 json 파일 업로드
    3. 챗봇
        - 대화 초기화 버튼
        - 사용자 입력
        - 챗봇 응답
        - 참고 문서 정보
    """
    initialize_session()
    session_id = st.session_state.id

    with st.sidebar:
        st.header("문서 업로드")
        uploaded_file = st.file_uploader(
            "PDF 또는 JSON 형식의 파일을 업로드 해주세요.", type=["pdf", "json"]
        )

        if uploaded_file:
            try:
                file_key = f"{session_id}-{uploaded_file.name}"
                if uploaded_file.type == "application/pdf":
                    st.write("PDF 문서를 인덱싱 하고 있습니다...")
                    process_pdf(uploaded_file, file_key)
                    st.write("성공적으로 PDF 문서를 인덱싱했습니다!")
                    display_pdf(uploaded_file)
                elif uploaded_file.type == "application/json":
                    st.write("JSON 문서를 인덱싱 하고 있습니다...")
                    process_json(uploaded_file, file_key)
                    st.write("성공적으로 JSON 문서를 인덱싱했습니다!")
                st.write("챗봇을 사용할 준비가 되었습니다!")
            except Exception as e:
                st.write(f"Error: {e}")
                st.stop()

    st.title("RAG Chatbot")

    if st.button("대화 초기화"):
        reset_chat()
        st.toast("초기화 되었습니다.", icon="❌")

    # streamlit에서 지원하는 챗봇 UI 사용
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.rag_chain is not None:
        if prompt := st.chat_input("Ask a question!"):
            MAX_MESSAGES_BEFORE_DELETION = 4  # 챗봇 대화 최대 저장 갯수

            # 챗봇 대화 최대 저장 갯수를 넘어가면 가장 오래된 대화를 삭제
            if len(st.session_state.messages) >= MAX_MESSAGES_BEFORE_DELETION:
                del st.session_state.messages[0]
                del st.session_state.messages[0]

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                rag_chain = st.session_state.rag_chain
                result = rag_chain.invoke(
                    {"input": prompt, "chat_history": st.session_state.messages}
                )  # 챗봇 체인 실행

                st.session_state.messages.append({"role": "user", "content": prompt})

                cleaned_datas = clean_data(
                    result["context"], is_pdf=uploaded_file.type == "application/pdf"
                )

                for cleaned_data in cleaned_datas:
                    with st.expander("Evidence context"):
                        st.write(f"Page content: {cleaned_data['page_content']}")
                        if (
                            "metadata" in cleaned_data
                            and "page" in cleaned_data["metadata"]
                        ):
                            st.write(f"Page: {cleaned_data['metadata']['page']+1}")

                # 챗봇 응답을 조각내어 보여줌
                for chunk in result["answer"].split(" "):
                    full_response += chunk + " "
                    time.sleep(0.2)
                    message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
    else:
        st.write("사이드바에서 PDF 또는 JSON 파일을 업로드 해주세요.")


main()
