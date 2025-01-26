import json
import requests
from typing import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
import config as cfg
from utils.llm_factory import LLMFactory
from models.v2.schemas import TaskType
from fastapi.encoders import jsonable_encoder


llm_factory = LLMFactory()
llm = llm_factory.load_llm(model_name=cfg.LLM_NAME)


class AgentState(TypedDict):
    medical_text: str
    user_profile: dict
    task: str
    generated_text: str
    simplified_text: str
    repeat: bool
    repeat_counter: int


def generate_text(state: AgentState):
    prompts = cfg.load_prompts()
    medical_text = state['medical_text']
    task_type = state['task']
    sys_msg = ""

    if task_type == TaskType.disease:
        sys_msg = prompts["v2"]["generator_agent"]["disease"]
    elif task_type == TaskType.diagnosis:
        sys_msg = prompts["v2"]["generator_agent"]["diagnosis"]

    prompt = [
        SystemMessage(content=sys_msg),
        HumanMessage(content=medical_text),
    ]

    generated_text = llm.invoke(prompt).content
    state['generated_text'] = generated_text
    return state


def simplify_text(state: AgentState):
    prompts = cfg.load_prompts()
    generated_text = state['generated_text']
    task_type = state['task']
    user_profile = state['user_profile']
    user_profile_json = json.dumps(jsonable_encoder(user_profile), indent=2)

    sys_msg = ""

    if task_type == TaskType.disease:
        sys_msg = prompts["v2"]["simplifier_agent"]["disease"]
    elif task_type == TaskType.diagnosis:
        sys_msg = prompts["v2"]["simplifier_agent"]["diagnosis"]

    prompt = [
        SystemMessage(content=sys_msg),
        HumanMessage(content=f"Generated Text:\n{generated_text}\n\nUser Profile:\n{user_profile_json}")
    ]

    simplified_text = llm.invoke(prompt).content
    state['simplified_text'] = simplified_text
    return state


async def validate_text(state: AgentState):
    repeat_counter = state['repeat_counter']

    validation_url = "http://validator:8002/v1/validate"

    payload = {
        "simplified_text": state['simplified_text'],
        "original_text": state['medical_text']
    }
    validation_result = requests.post(validation_url, json=payload)
    validation_result = validation_result.json()
    validation_score = validation_result["similarity"] * 100

    if (validation_score < 60) and (repeat_counter < 3):
        repeat_counter += 1
        state['repeat'] = True
        state['repeat_counter'] = repeat_counter
    else:
        state['repeat'] = False

    return state


def decide_repeat(state: AgentState):
    return state['repeat']


graph_builder = StateGraph(AgentState)
graph_builder.add_node("generator_agent", generate_text)
graph_builder.add_node("simplifier_agent", simplify_text)
graph_builder.add_node("validator_agent", validate_text)
graph_builder.add_edge(START, "generator_agent")
graph_builder.add_edge('generator_agent', "simplifier_agent")
graph_builder.add_edge('simplifier_agent', "validator_agent")
graph_builder.add_conditional_edges('validator_agent', decide_repeat, {
                                    True: 'generator_agent', False: END})

graph = graph_builder.compile()


async def summarize(medical_text: str, user_profile: dict, task: TaskType):
    input_message = {"medical_text": medical_text,
                     "user_profile": user_profile, "task": task.value, "repeat_counter": 0}

    async for chunk in graph.astream(input_message, stream_mode="values"):
        response = chunk

    simplified_text = response['simplified_text']
    return simplified_text
