__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# 사용할 라이브러리 선언
import streamlit as st
import openai
import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pprint import pprint
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import json
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import time
import logging

print("AA")

# 문제 진단: 프로그램이 복잡할수록 버그나 오류를 찾기 어려워지는데, 로그는 코드 실행 흐름을 기록하여 문제 발생 지점을 쉽게 찾아낼 수 있게 도와줍니다.
# 로그 설정: 기본 로그 설정을 초기화합니다.
# 특정 로거 레벨 설정: "langchain.retrievers.multi_query" 로거의 로그 레벨을 INFO로 설정하여 해당 모듈의 로그를 표시합니다.
# 파일에 로그 저장: 로그 메시지를 'info.log' 파일에 시간, 로거 이름, 로그 레벨, 메시지 형식으로 저장합니다. 'a' 모드는 기존 파일에 내용을 추가합니다.
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
logging.basicConfig(filename='info.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filemode='a')

# 사용할 챗봇 LLM 모델 설정해주기
# temperature는 0에 가까워질수록 전달해준 설명 및 질문 내에서만 답변을 내뱉고, 1에 가까워질수록 창의적인 답변을 내뱉음
chat = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.1)

# 1) WEB 데이터 로드하기
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Elon_Musk")
data = loader.load()

# 2) 데이터 쪼개기
# chunk_size는 쪼갤 텍스트의 사이즈를 뜻합니다.
# 쪼개는 이유는 context window가 제한되어 있기 때문.
# 만약 더 큰 데이터 파일이고 chunk_size가 높다면, chunk_overlap = 200으로 설정
# 각 청크 사이에 중요한 맥락을 놓치지 않도록 어느정도 중복되게 하는 것
chunk_size = 10000
chunk_overlap = 2000

# RecursiveCharacterTextSplitter: 문서를 \n과 같은 일반적인 구분자를 사용하여 재귀적으로 분할하여 각 청크가 적절한 크기가 될 때까지 분할합니다.
# chunk_overlap은 전 문장의 의미를 어느정도 연결할 수 있도록 하는 역할을 합니다. 보통 10 ~ 20% 정도의 chunk를 할당합니다.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
all_splits = text_splitter.split_documents(data)

# pprint(all_splits)
# print("="*50)
# # 메타데이터가 잘 작성되었는지, 텍스트가 잘 잘렸는지 확인
# pprint(all_splits[0].metadata)
# print("="*50)
# pprint(all_splits[0].page_content)

# 임베딩이라는 뜻은 'similarity search'를 하기 위해서임.
# 각 토큰은 컴퓨터가 이해할 수 있는 숫자인 백터로 바뀌고, 이는 위치와 방향을 갖고 있는 숫자.
# 즉 이 토큰-백터 유사성이 높은 다른 토큰들을 불러오기 위해 임베딩을 해야함.

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 여기서 "./embedding_db/openai_large"는 텍스트를 임베딩한 후 해당 데이터를 저장할 디렉토리 경로를 의미합니다.
# 즉, Chroma는 임베딩된 문서 데이터를 이 경로에 저장하여, 나중에 다시 사용할 수 있도록 합니다.
# 이렇게 하면 매번 새로 임베딩할 필요 없이, 저장된 데이터를 불러와 더 효율적으로 작업을 진행할 수 있습니다.
vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="./embedding_db/openai_large")

# 4) 데이터를 불러올 수 있는 retriever 만들기
# 멀티쿼리 리트리버란 질문을 쪼개서 더 다양한 문서가 나올 수 있도록 함

# MMR(Maximal Marginal Relevance) search 알고리즘 검색 결과에서 관련성과 다양성을 동시에 고려하는 방식입니다.
# 관련성: 쿼리와 가장 관련 있는 문서들을 우선 선택.
# 다양성: 이미 선택된 문서들과 중복되지 않도록 새로운 정보를 제공하는 문서들을 추가로 선택.

# search_kwargs={'k': 5, 'fetch_k': 50}는 검색 시 관련 문서 5개를 반환하면서,
# 최대 50개의 문서를 먼저 가져와 그 중에서 가장 관련성 높은 5개를 선택하도록 설정한 것.

mmr_retriever = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 5, 'fetch_k': 50}
    ), llm=chat
)


# LLM에게 역할을 주어주는 프롬프트 작성
# 역할을 상세하게 적어줄수록 내가 원하는 반응을 하도록 할 수 있음
template = """From now on, you are an expert who finds and provides information that fits the user's question in the provided context.

1. When you want information that is not in the context, answer that you do not have the information you want.
2. Never answer with information that is not in the context.
3. If you do not know the content, say that you do not know.
4. Make the explanation as detailed and long as possible.
5. Make sure to answer in Korean!!

context: {context}
user input: {input}
"""

prompt = ChatPromptTemplate.from_template(template)

# RunnableParallel의 역할: 병렬로 속도를 높이기 위해 작업 처리.
# 여기서는 mmr_retriever와 RunnablePassthrough() 두 개의 작업을 동시에 실행하여 처리 속도를 높이는 역할을 합니다.

# RunnablePassthrough()의 역할:
# 변경 없이 전달: 입력된 값에 대해 아무런 처리를 하지 않고, 그대로 다음 단계로 넘깁니다.
# 병렬 처리 환경에서: 병렬 작업에서 여러 가지 작업을 동시에 실행할 때, 일부 입력을 가공하지 않고 그대로 사용할 경우 유용합니다.
# "입력 데이터를 있는 그대로 전달한다"
retrieval = RunnableParallel({"context": mmr_retriever, "input": RunnablePassthrough()})

chain = retrieval | prompt | chat | StrOutputParser()

# response = chain.invoke({'input':"Explain Dalpha AI Store"})
# print(response)


st.title("AI Chatbot")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini-2024-07-18"

# 만약 이전 사용자 대화 내용 기록이 없다면 기록할 새로운 리스트 생성
if "messages" not in st.session_state:
    st.session_state.messages = []

# 리스트에 메세지 역할과 내용을 기록하도록 함
for message in st.session_state.messages:
  # 특정 블록의 시작과 끝에서 자동으로 설정 및 정리 작업을 처리하는 역할
  # st.chat_message() 블록 내에서 채팅 메시지를 생성하고, 그 안에서 st.markdown()으로 콘텐츠를 표시
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

MAX_MESSAGES_BEFORE_DELETION = 4

if prompt := st.chat_input("대화를 시작하세요"):

    # 4개의 메세지만 화면에 나오도록 표시. 4개 이상의 메세지는 토큰값이 많이 나오기 때문에 2개씩 삭제한다.
    if len(st.session_state.messages) >= MAX_MESSAGES_BEFORE_DELETION:
        del st.session_state.messages[0]
        del st.session_state.messages[0]

    # 대화 기록하기. 역할은 사용자로, 사용자가 날린 질문을 프롬프트에 저장해서 기록한다.
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 만약 메세지 기록에서 역할이 "유저"로 되어있으면 프롬프트를 유저 아이콘으로 표기하기
    with st.chat_message("user"):
        st.markdown(prompt)

    # 만약 AI가 답변하면 RAG chain에서 받아온 답변 기록하기
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # 위에서 만든 RAG 체인을 불러온다.
        result = chain.invoke({"input": prompt, "chat_history": st.session_state.messages})

        for chunk in result.split(" "):
            full_response += chunk + " "
            time.sleep(0.2)
            message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

print("_______________________")
print(st.session_state.messages)
