import streamlit as st
import requests

# Streamlit 페이지 제목 설정
st.title("크리니티 manual-chatbot")

# 세션 시작
if "session_id" not in st.session_state:
    response = requests.post("http://localhost:8000/start_session")
    if response.status_code == 200:
        st.session_state.session_id = response.json().get("session_id")
        st.success("세션이 시작되었습니다.")
    else:
        st.error("세션을 시작하는 중 오류가 발생했습니다.")

# PDF 리스트 (로컬 파일 기준)
pdf_list = ["CM9_manual", "example2", "example3"]

# PDF 파일 선택
selected_pdf = st.selectbox("PDF 파일을 선택하세요", pdf_list)

# 인덱스 로드 버튼
if st.button("인덱스 로드"):
    response = requests.post(
        "http://localhost:8000/load_pdf_index", json={"pdf_name": selected_pdf}
    )
    if response.status_code == 200:
        st.success("인덱스가 성공적으로 로드되었습니다.")
    else:
        st.error("인덱스를 로드하는 중 오류가 발생했습니다.")

# 챗봇 인터페이스
st.subheader("챗봇과 대화하기")

# 질문 입력
question = st.text_input("질문을 입력하세요")

# 질문 전송 버튼
if st.button("질문 전송"):
    if not question:
        st.error("질문을 입력하세요.")
    else:
        response = requests.post(
            "http://localhost:8000/generate/",
            json={"question": question},
            cookies={"session_id": st.session_state.session_id},
            stream=True,
        )

        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    st.write(decoded_line)
        else:
            st.error("답변을 가져오는 중 오류가 발생했습니다.")
