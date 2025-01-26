import json
from typing import TypedDict
import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from utils.llm_factory import LLMFactory
from models.v3.schemas import TaskType
from models.v3.schemas import SummarizeResponse
from fastapi.encoders import jsonable_encoder
import config as cfg


class AgentState(TypedDict):
    medical_text: str
    user_profile: dict
    task: str
    generated_text: str
    simplified_text: str
    validation_repeat: bool
    validator_enabled: bool
    validation_repeat_counter: int
    validation_score: float
    language: str
    llm: str


def generate_text(state: AgentState):
    prompts = cfg.load_prompts()
    medical_text = state['medical_text']
    task_type = state['task']
    llm = state['llm']
    sys_msg = ""

    if task_type == TaskType.disease:
        sys_msg = prompts["v3"]["generator_agent"]["disease"]

    elif task_type == TaskType.diagnosis:
        sys_msg = prompts["v3"]["generator_agent"]["diagnosis"]

    prompt = [
        SystemMessage(content=sys_msg),
        HumanMessage(content=f"Medical content for summarization: {medical_text}"),
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
    llm = state['llm']
    sys_msg = ""

    if task_type == TaskType.disease:
        sys_msg = prompts["v3"]["simplifier_agent"]["disease"]
    elif task_type == TaskType.diagnosis:
        sys_msg = prompts["v3"]["simplifier_agent"]["diagnosis"]
    
    prompt = [
        SystemMessage(content=f"{sys_msg}\n\n Patient Profile:\n{user_profile_json}"),
        HumanMessage(content=f"Now, personalize this medical content:\n{generated_text}")
    ]

    simplified_text = llm.invoke(prompt).content
    state['simplified_text'] = simplified_text
    return state


def validate_text(state: AgentState):
    validation_repeat_counter = state['validation_repeat_counter']
    validation_score = state["validation_score"]

    if (validation_score < cfg.VALIDATOR_SCORE_THRESHOLD) and (validation_repeat_counter < cfg.VALIDATOR_REPEAT_COUNT):
        validation_repeat_counter += 1
        state['validation_repeat_counter'] = validation_repeat_counter

        payload = {
            "simplified_text": state['simplified_text'],
            "original_text": state['medical_text']
        }

        validation_result = requests.post(cfg.VALIDATOR_URL, json=payload)
        validation_result = validation_result.json()
        validation_score = validation_result["similarity"] * 100
        state['validation_score'] = validation_score
    else:
        state['validation_repeat'] = False

    return state


def translate_text(state: AgentState):
    simplified_text = state['simplified_text']
    language = state['language']
    llm = state['llm']

    sys_msg = f"Translate the text to {language}"
    prompt = [
        SystemMessage(content=sys_msg),
        HumanMessage(content=f"{simplified_text}"),
    ]

    # Replacing the simplified_text in English to desired language
    simplified_text = llm.invoke(prompt).content
    state['simplified_text'] = simplified_text
    return state


def _route_from_simplifier(state: AgentState):
    validator_enabled = state['validator_enabled']
    repeat_validation = state['validation_repeat']
    perform_translation = state['language'] != 'English'

    if not validator_enabled and not perform_translation:
        return "end"

    if validator_enabled:
        if repeat_validation:
            return "validate"
        if perform_translation:
            return "translate"
        return "end"

    if perform_translation:
        return "translate"

    return "end"


def _route_from_validator(state: AgentState):
    repeat_validation = state['validation_repeat']
    perform_translation = state['language'] != 'English'

    if repeat_validation:
        return "generate"
    if perform_translation:
        return "translate"

    return "end"


def build_graph():
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("generator_agent", generate_text)
    graph_builder.add_node("simplifier_agent", simplify_text)
    graph_builder.add_node("validator_agent", validate_text)
    graph_builder.add_node("translator_agent", translate_text)
    graph_builder.add_edge(START, "generator_agent")
    graph_builder.add_edge('generator_agent', "simplifier_agent")
    graph_builder.add_conditional_edges('simplifier_agent', _route_from_simplifier, {"validate": "validator_agent", "translate": "translator_agent", "end": END})
    graph_builder.add_conditional_edges('validator_agent', _route_from_validator, {"generate": 'generator_agent', "translate": "translator_agent", "end": END})
    graph_builder.add_edge('translator_agent', END)

    return graph_builder


async def summarize(medical_text: str, user_profile: dict, task: TaskType, language: str,  validator_enabled: bool, llm_name: str):

    graph = build_graph().compile()
    llm_factory = LLMFactory()
    llm = llm_factory.load_llm(llm_name)
    input_message = {"medical_text": medical_text, "user_profile": user_profile, "task": task.value, "validation_score": 0.0,
                     "validation_repeat_counter": 0, "validation_repeat": True, "validator_enabled": validator_enabled, "language": language,  "llm": llm}

    async for chunk in graph.astream(input_message, stream_mode="values"):
        response = chunk

    simplified_text = response['simplified_text']
    validation_score = response['validation_score']

    return SummarizeResponse(simplified_text=simplified_text, validation_score=validation_score)
