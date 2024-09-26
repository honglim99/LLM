import streamlit as st
import json
import requests
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain.chains import RetrievalQA

 
url = 'http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getCtprvnRltmMesureDnsty'
params ={'serviceKey' : 'uuiP9ecF/DdtMXPgok9ActF0w9gbQkIiKeJvtjjDBtdOFA90FQnzF5i48cAH7QTk2wPOqmipRVfS32TcrQMo6w==', 'returnType' : 'json', 'numOfRows' : '100', 'pageNo' : '1', 'sidoName' : '서울', 'ver' : '1.0' }
 
response = requests.get(url, params=params)


sido = "서울"
def seoul_pm_query(sido):
    url = 'http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getCtprvnRltmMesureDnsty'
    params = {'serviceKey' : 'uuiP9ecF/DdtMXPgok9ActF0w9gbQkIiKeJvtjjDBtdOFA90FQnzF5i48cAH7QTk2wPOqmipRVfS32TcrQMo6w==', 'returnType' : 'json', 'numOfRows' : '100', 'pageNo' : '1', 'sidoName' : '서울', 'ver' : '1.0' }
    response = requests.get(url, params=params)
    content = response.content.decode('utf-8')
    data = json.loads(content)
    return data

def parse_air_quality_data(data):
    items = data['response']['body']['items']
    air_quality_info = []
    for item in items:
        info = {
            '측정소명': item.get('stationName'),
            '날짜': item.get('dataTime'),
            'pm10농도': item.get('pm10Value'),
            'pm25농도': item.get('pm25Value'),
            'so2농도': item.get('so2Value'),
            'co농도': item.get('coValue'),
            'o3농도': item.get('o3Value'),
            'no2농도': item.get('no2Value'),
            '통합대기환경수치': item.get('khaiValue'),
            '통합대기환경지수': item.get('khaiGrade'),
            'pm10등급': item.get('pm10Grade'),
            'pm25등급': item.get('pm25Grade')
        }
        air_quality_info.append(info)
    return air_quality_info




def main():
    st.title("대기질 정보 제공 챗봇")
    text_bar = st.text_input("텍스트를 입력해주세요.")
    question = st.text_input("구도 입력해주세요.")
    clicked_button = st.button("제출")
    if clicked_button:
        result = seoul_pm_query(text_bar)
        air_quality_info = parse_air_quality_data(result)
        documents = [Document(page_content=", ".join([f"{key}: {str(info[key])}" for key in ['측정소명', '날짜', 'pm10농도', 'pm25농도', '통합대기환경수치']])) for info in air_quality_info]
        text_splitter = RecursiveCharacterTextSplitter( separators=["',"])
        docs = text_splitter.split_documents(documents)
        embedding_function = SentenceTransformerEmbeddings(model_name="jhgan/ko-sroberta-multitask")
        db = FAISS.from_documents(docs, embedding_function )

        llm = ChatOllama(model="gemma2:9b", temperature=0, base_url="http://127.0.0.1:11434/") 

        qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever(search_type="similarity", search_kwargs={'k':5, 'fetch_k': 100}))
        result = qa_chain({"query": question})
        st.write(result['result'])
 
if __name__ == "__main__":
    main()