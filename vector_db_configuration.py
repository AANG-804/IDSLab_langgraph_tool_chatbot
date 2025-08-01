## 데이터베이스 검색 도구
def create_vector_db_tool(pdf_dir: str = "data"):
    """벡터 데이터베이스 도구 생성"""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_community.vectorstores import Chroma
        from langchain_openai import OpenAIEmbeddings
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_core.tools import tool
        import tqdm
        import os
        
        # /data 폴더가 존재하는지 확인
        if not os.path.exists(pdf_dir):
            print(f"Warning: {pdf_dir} 폴더가 존재하지 않습니다. 벡터 DB 도구를 건너뜁니다.")
            return None
        
        pdf_files = [
            os.path.join(pdf_dir, f)
            for f in os.listdir(pdf_dir)
            if f.lower().endswith(".pdf")
        ]
        
        if not pdf_files:
            print(f"Warning: {pdf_dir} 폴더에 PDF 파일이 없습니다. 벡터 DB 도구를 건너뜁니다.")
            return None
        
        # chunk + batch 세팅
        chunk_size = 1000
        chunk_overlap = 200
        batch_size = 40
        
        print("="*100)
        print("벡터 DB 초기화 중...")
        print("="*100)
        for pdf_file in pdf_files:
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            docs = text_splitter.split_documents(docs)
            
            for i in tqdm.tqdm(range(0, len(docs), batch_size)):
                batch_docs = docs[i:i+batch_size]
                db = Chroma.from_documents(batch_docs, embeddings, persist_directory="vector_db")
        
        retriever = db.as_retriever()
        
        # Retriever를 Tool로 변환
        @tool
        def vector_db_retriever(query: str) -> str:
            """2025년 1분기 삼성전자의 사업 분기 보고서에서 query와 가장 관련있는 정보를 반환합니다."""
            return retriever.invoke(query)
        
        return vector_db_retriever
        
    except Exception as e:
        print(f"Warning: 벡터 DB 도구 생성 중 오류 발생: {e}")
        return None
