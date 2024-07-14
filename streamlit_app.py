import base64
import os
import tempfile
import time
import streamlit as st
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.llms import Ollama
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.chains.combine_documents import create_stuff_documents_chain


def initialize_session():
    if "id" not in st.session_state:
        st.session_state.id = uuid.uuid4()
        st.session_state.file_cache = {}
        st.session_state.messages = []
        st.session_state.rag_chain = None  # RAG 체인 초기화


def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None


def display_pdf(file):
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf" style="height:100vh; width:100%"></iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)


def process_pdf(file, file_key):
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
                chunk_size=1000, chunk_overlap=50
            )
            split_pages = text_splitter.split_documents(pages)
            print(f"분할된 청크의수: {len(split_pages)}")

            vectorstore = FAISS.from_documents(
                documents=split_pages, embedding=embedding_model
            )

            retriever = vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": 3}
            )
            llm = Ollama(model="llama-3-Korean-Bllossom-8B-gguf-Q4_K_M")

            question_rephrasing_chain = create_question_rephrasing_chain(llm, retriever)
            question_answering_chain = create_question_answering_chain(llm)

            rag_chain = create_retrieval_chain(
                question_rephrasing_chain, question_answering_chain
            )

            st.session_state.file_cache[file_key] = rag_chain

        st.session_state.rag_chain = st.session_state.file_cache[
            file_key
        ]  # RAG 체인 설정


def create_question_rephrasing_chain(llm, retriever):
    system_prompt = """이전 대화 내용과 최신 사용자 질문이 있을 때, 이 질문이 이전 대화 내용과 관련이 있을 수 있습니다. 이런 경우, 대화 내용을 알 필요 없이 독립적으로 이해할 수 있는 질문으로 바꾸세요. 질문에 답할 필요는 없고, 필요하다면 그저 다시 구성하거나 그대로 두세요."""

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    return create_history_aware_retriever(llm, retriever, prompt)


def create_question_answering_chain(llm):
    system_prompt = """질문-답변 업무를 돕는 보조원입니다. 질문에 답하기 위해 검색된 내용을 사용하세요. 답을 모르면 모른다고 말하세요. 답변은 세 문장 이내로 간결하게 유지하세요.\n\n## 답변 예시\n답변 내용:\n증거:\n\n{context}"""

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    return create_stuff_documents_chain(llm, prompt)


def clean_data(data):
    cleaned_data = []
    for item in data:
        cleaned_page_content = item.page_content.replace("\n", " ").strip()
        cleaned_metadata = {
            "page": item.metadata["page"],
            "total_pages": item.metadata["total_pages"],
        }
        cleaned_data.append(
            {"page_content": cleaned_page_content, "metadata": cleaned_metadata}
        )
    return cleaned_data


def main():
    initialize_session()
    session_id = st.session_state.id

    with st.sidebar:
        st.header("Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

        if uploaded_file:
            try:
                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")
                process_pdf(uploaded_file, file_key)
                st.write("Document indexed successfully!")
                st.write("Ready to chat!")
                # display_pdf(uploaded_file) # 버그로 인해 주석 처리
            except Exception as e:
                st.write(f"Error: {e}")
                st.stop()

    st.title("PDF Chatbot")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if (
        st.session_state.rag_chain is not None
    ):  # RAG 체인이 생성되었을 때만 채팅 입력 활성화
        if prompt := st.chat_input("Ask a question!"):
            MAX_MESSAGES_BEFORE_DELETION = 4

            if len(st.session_state.messages) >= MAX_MESSAGES_BEFORE_DELETION:
                del st.session_state.messages[0]
                del st.session_state.messages[0]

            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                rag_chain = st.session_state.rag_chain
                result = rag_chain.invoke(
                    {"input": prompt, "chat_history": st.session_state.messages}
                )

                cleaned_datas = clean_data(result["context"])

                for cleaned_data in cleaned_datas:
                    with st.expander("Evidence context"):
                        st.write(f"Page Content: {cleaned_data['page_content']}")
                        st.write(
                            f"Page: {cleaned_data['metadata']['page']}/{cleaned_data['metadata']['total_pages']}"
                        )

                for chunk in result["answer"].split(" "):
                    full_response += chunk + " "
                    time.sleep(0.2)
                    message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
    else:
        st.write("사이드 바에서 pdf 파일을 업로드 해주세요.")


main()
