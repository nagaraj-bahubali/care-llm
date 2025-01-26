import json
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
import operator
from utils.llm_factory import LLMFactory
from utils.helpers import get_redis_checkpointer
from models.v3.schemas import ConverseResponse
from fastapi.encoders import jsonable_encoder
import config as cfg

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    llm: str


def generate_text(state: AgentState):
    llm = state['llm']
    return {"messages": [llm.invoke(state["messages"])]}


def build_graph():
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("conversation_agent", generate_text)
    graph_builder.add_edge(START, "conversation_agent")
    graph_builder.add_edge('conversation_agent', END)

    return graph_builder


async def converse(user_question: str, medical_text: str, user_profile: dict, chat_id: str,  llm_name: str):
    prompts = cfg.load_prompts()
    llm_factory = LLMFactory()
    llm = llm_factory.load_llm(llm_name)

    config = {"configurable": {"thread_id": chat_id}}
    user_profile_json = json.dumps(jsonable_encoder(user_profile), indent=2)
    sys_msg = prompts["v3"]["conversation_agent"]

    prompt = [
        SystemMessage(content=f"{sys_msg}\n\nUser Profile:\n{user_profile_json}"),
        HumanMessage(content=f"Medical Text:\n{medical_text}\n\n Question:\n{user_question}"),
    ]

    input_message = {"messages": prompt, "llm": llm}

    async with get_redis_checkpointer() as checkpointer:
        graph_with_checkpointer = build_graph().compile(checkpointer=checkpointer)

    async for chunk in graph_with_checkpointer.astream(input_message, config, stream_mode="values"):
        response = chunk

    reply = response["messages"][-1].content

    return ConverseResponse(reply=reply)

# async def initiate_chat(user_question: str, medical_text: str, user_profile: dict, chat_id: str,  llm_name: str):
#     llm_factory = LLMFactory()
#     llm = llm_factory.load_llm(llm_name)

#     config = {"configurable": {"thread_id": chat_id}}
#     user_profile_json = json.dumps(jsonable_encoder(user_profile), indent=2)
#     sys_msg = prompts["v3"]["conversation_agent"]

#     prompt = [
#         SystemMessage(content=sys_msg),
#         HumanMessage(content=f"Medical Text:\n{medical_text}\n\nUser Profile:\n{user_profile_json}\n\nUser Question:\n{user_question}"),
#     ]

#     input_message = {"messages": prompt, "llm": llm}

#     async with get_redis_checkpointer() as checkpointer:
#         graph_with_checkpointer = build_graph().compile(checkpointer=checkpointer)

#     async for chunk in graph_with_checkpointer.astream(input_message, config, stream_mode="values"):
#         response = chunk

#     reply = response["messages"][-1].content

#     return ConverseResponse(reply=reply)


# async def resume_chat(user_question: str, chat_id: str):
#     config = {"configurable": {"thread_id": chat_id}}
#     prompt = [HumanMessage(content=f"User Question:\n{user_question}")]
#     input_message = {"messages": prompt}

#     async with get_redis_checkpointer() as checkpointer:
#         # Recompile the graph with the checkpointer
#         graph_with_checkpointer = build_graph().compile(checkpointer=checkpointer)

#     async for chunk in graph_with_checkpointer.astream(input_message, config, stream_mode="values"):
#         response = chunk

#     reply = response["messages"][-1].content

#     return ConverseResponse(reply=reply)
