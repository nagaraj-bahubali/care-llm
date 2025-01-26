import json
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
import operator
import config as cfg
from contextlib import asynccontextmanager
from utils.async_redis_saver import AsyncRedisSaver
from utils.llm_factory import LLMFactory
from fastapi.encoders import jsonable_encoder
import config as cfg


llm_factory = LLMFactory()
llm = llm_factory.load_llm(model_name=cfg.LLM_NAME)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def generate_text(state: AgentState):
    return {"messages": [llm.invoke(state["messages"])]}


@asynccontextmanager
async def get_redis_checkpointer():
    """Creates an async context manager for the Redis checkpoint saver."""
    async with AsyncRedisSaver.from_conn_info(host=cfg.REDIS_HOST, port=cfg.REDIS_PORT, db=cfg.REDIS_DB) as checkpointer:
        yield checkpointer


graph_builder = StateGraph(AgentState)
graph_builder.add_node("conversation_agent", generate_text)
graph_builder.add_edge(START, "conversation_agent")
graph_builder.add_edge('conversation_agent', END)


async def initiate_chat(user_question: str, medical_text: str, user_profile: dict, chat_id: str):
    prompts = cfg.load_prompts()
    async with get_redis_checkpointer() as checkpointer:
        config = {"configurable": {"thread_id": chat_id}}
        user_profile_json = json.dumps(
            jsonable_encoder(user_profile), indent=2)
        sys_msg = prompts["v2"]["conversation_agent"]

        prompt = [SystemMessage(content=sys_msg),
                  HumanMessage(content=f"Medical Text:\n{medical_text}\n\nUser Profile:\n{user_profile_json}\n\nUser Question:\n{user_question}"),
                  ]

        input_message = {"messages": prompt}

        # Recompile the graph with the checkpointer
        graph_with_checkpointer = graph_builder.compile(
            checkpointer=checkpointer)

        async for chunk in graph_with_checkpointer.astream(input_message, config, stream_mode="values"):
            response = chunk

        reply = response["messages"][-1].content

        return reply


async def resume_chat(user_question: str, chat_id: str):
    async with get_redis_checkpointer() as checkpointer:
        config = {"configurable": {"thread_id": chat_id}}

        prompt = [
            HumanMessage(content=f"User Question:\n{user_question}"),
        ]

        input_message = {"messages": prompt}

        # Recompile the graph with the checkpointer
        graph_with_checkpointer = graph_builder.compile(
            checkpointer=checkpointer)

        async for chunk in graph_with_checkpointer.astream(input_message, config, stream_mode="values"):
            response = chunk

        reply = response["messages"][-1].content

        return reply
