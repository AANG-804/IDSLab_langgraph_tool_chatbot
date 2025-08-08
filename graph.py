"""
LangGraph를 사용한 도구 기반 챗봇 구현
"""

from typing import Annotated, List, Optional, Any
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

# 도구 생성
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_chroma import Chroma
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

from langgraph.managed.is_last_step import RemainingSteps

from dotenv import load_dotenv
load_dotenv()


class State(TypedDict):
    """그래프 상태를 정의하는 타입"""
    messages: Annotated[List[BaseMessage], add_messages]
    remaining_steps: RemainingSteps


# Tool 생성
## 위키피디아 도구
wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=2,
                                       doc_content_chars_max=2000)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

## SQL 도구
db = SQLDatabase.from_uri("sqlite:///stock.db")
# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)

from langchain.tools import tool


@tool
def get_database_schema() -> str:
    """Get basic information about the database schema for stock data."""
    return """Database contains a 'Nasdaq100' table with columns:
    - 순번: Primary key (INTEGER)
    - 날짜: Trading date (TEXT format)
    - 시가: Opening price (REAL)
    - 고가: Highest price (REAL)  
    - 저가: Lowest price (REAL)
    - 종가: Closing price (REAL)
    - 수정종가: Adjusted closing price (REAL)
    - 거래량: Trading volume (INTEGER)
    - 종목명: Stock Ticker Symbol (TEXT, e.g., AAPL for Apple Inc.)

    Example query: SELECT * FROM Nasdaq100 WHERE 종목명 = 'AAPL' ORDER BY 날짜 DESC LIMIT 5;

    Available stock tickers include: AAPL, ADBE, ADI, ADP, ADSK, AEP, ALGN, AMAT, AMD, AMGN, AMZN, ANSS, ASML, ATVI, AVGO, BIDU, BIIB, BKNG, CDNS, CDW, CERN, CHKP, CHTR, CMCSA, COST, CPRT, CRWD, CSCO, CSX, CTAS, CTSH, DLTR, DOCU, DXCM, EA, EBAY, EXC, FAST, FB, FISV, FOX, FOXA, GILD, GOOG, GOOGL, HON, IDXX, ILMN, INCY, INTC, INTU, ISRG, JD, KDP, KHC, KLAC, LRCX, LULU, MAR, MCHP, MDLZ, MELI, MNST, MRNA, MRVL, MSFT, MTCH, MU, NFLX, NTES, NVDA, NXPI, OKTA, ORLY, PAYX, PCAR, PDD, PEP, PTON, PYPL, QCOM, REGN, ROST, SBUX, SGEN, SIRI, SNPS, SPLK, SWKS, TCOM, TEAM, TMUS, TSLA, TXN, VRSK, VRSN, VRTX, WBA, WDAY, XEL, XLNX, ZM

    **Note**: cannot perform tasks like direct statistical calculations; let the user know that you should ask only about general informations based on data.
    """


db = Chroma(persist_directory="vector_db")
retriever = db.as_retriever(k=1)


# Retriever를 Tool로 변환
@tool
def vector_db_retriever(query: str) -> str:
    """2025년 1분기 삼성전자의 사업 분기 보고서에서 query와 가장 관련있는 정보를 반환합니다."""
    results = retriever.invoke(query)
    # Convert Document objects to string
    if isinstance(results, list):
        return "\n\n".join([
            doc.page_content if hasattr(doc, 'page_content') else str(doc)
            for doc in results
        ])
    return str(results)


# 기본 도구들 (벡터 DB 제외)
BASE_TOOLS = [wiki_tool, get_database_schema, vector_db_retriever
              ] + sql_toolkit.get_tools()


def get_available_tools():
    """사용 가능한 도구들을 동적으로 가져오기"""
    tools = BASE_TOOLS.copy()

    return tools


# 기본 설정
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.4
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that can use tools to answer questions. if you can't answer the question with the tools, you can answer it with your knowledge."


def build_graph(model: str = DEFAULT_MODEL,
                temperature: float = DEFAULT_TEMPERATURE,
                system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                tools: Optional[List[Any]] = None):
    """
    LangGraph 애플리케이션 빌드
    
    Args:
        model: 사용할 모델명
        temperature: 모델의 temperature 설정
        system_prompt: 시스템 프롬프트
        tools: 사용할 도구 리스트
    
    Returns:
        컴파일된 그래프
    """
    if tools is None:
        tools = get_available_tools()

    # None 값 필터링
    tools = [tool for tool in tools if tool is not None]

    # LLM 생성
    llm = ChatOpenAI(model=model, temperature=temperature).bind_tools(tools)

    # 챗봇 노드 정의
    def chatbot_node(state: State) -> State:
        response = llm.invoke([SystemMessage(content=system_prompt)] +
                              state["messages"])
        return {"messages": [response]}

    # 라우터 정의
    def router(state: State) -> str:

        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and hasattr(
                last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        else:
            return END

    # 그래프 빌더 생성
    builder = StateGraph(State)

    # 노드 추가
    builder.add_node("chatbot", chatbot_node)
    builder.add_node("tools", ToolNode(tools))

    # 엣지 추가
    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", tools_condition)
    builder.add_edge("tools", "chatbot")

    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


# 기본 그래프 생성 (지연 실행)
graph = build_graph()
