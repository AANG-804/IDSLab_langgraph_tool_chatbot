"""
LangGraph를 사용한 도구 기반 챗봇 구현
"""

from typing import Annotated, List
from typing_extensions import TypedDict

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

# 도구 생성
from langgraph.prebuilt import ToolNode
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from vector_db_configuration.vector_db_configuration import create_vector_db_tool
from langchain import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

# 환경 변수 로드
load_dotenv()

class State(TypedDict):
    """그래프 상태를 정의하는 타입"""
    messages: Annotated[List[BaseMessage], add_messages]


# Tool 생성
## 위키피디아 도구
wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=2)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

## SQL 도구
db = SQLDatabase.from_uri("sqlite:///stock.db")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)



# 기본 도구들 (벡터 DB 제외)
BASE_TOOLS = [wiki_tool] + sql_toolkit.get_tools()


def get_available_tools():
    """사용 가능한 도구들을 동적으로 가져오기"""
    tools = BASE_TOOLS.copy()
    
    ## 벡터 DB 도구 생성 시도
    vector_db_tool = create_vector_db_tool()
    if vector_db_tool is not None:
        tools.append(vector_db_tool)
    
    return tools


# 기본 설정
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that can use tools to answer questions."

def build_graph(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    tools: List = None
):
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
    llm = ChatOpenAI(
        model=model,
        temperature=temperature
    ).bind_tools(tools)
    
    # 챗봇 노드 정의
    def chatbot_node(state: State) -> State:
        response = llm.invoke(
            [SystemMessage(content=system_prompt)] +
            state["messages"]
        )
        return {"messages": [response]}
    
    # 라우터 정의
    def router(state: State) -> str:
        last_message = state["messages"][-1]
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return END
        else:
            return "tools"
    
    # 그래프 빌더 생성
    builder = StateGraph(State)
    
    # 노드 추가
    builder.add_node("chatbot", chatbot_node)
    builder.add_node("tools", ToolNode(tools))
    
    # 엣지 추가
    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", router, ["tools", END])
    builder.add_edge("tools", "chatbot")
    
    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


# 기본 그래프 생성 (지연 실행)
graph = build_graph()




