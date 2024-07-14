# faiss
# pdf 경로 받아서 인덱스 생성하고 로컬에 저장

import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def main(pdf_filepath):

    embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

    # 페이지 넘버를 받아서 pdf를 로드
    loader = PyMuPDFLoader(pdf_filepath)
    pdf_name = pdf_filepath.split("/")[-1].split(".")[0]

    docs = loader.load()
    print(f"문서의 페이지수: {len(docs)}")

    # 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)
    print(f"분할된 청크의수: {len(split_documents)}")

    vectorstore = FAISS.from_documents(
        documents=split_documents, embedding=embedding_model
    )

    # 로컬에 저장
    vectorstore.save_local(f"index/faiss_index_{pdf_name}.index")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_vector.py <pdf_filepath>")
        sys.exit(1)

    pdf_filepath = sys.argv[1]
    main(pdf_filepath)
