from fastapi import APIRouter
from models.v3.schemas import SummarizeRequest, SummarizeResponse, ConverseRequest, ConverseResponse
from services.v3.summarization import summarize
from services.v3.conversation import converse
from fastapi.encoders import jsonable_encoder
from utils import helpers
import config as cfg

router = APIRouter()


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_v3(request: SummarizeRequest):
    user_profile = helpers.update_user_profile(
        request.user_profile.model_dump())
    llm_name = request.llm_name if request.llm_name is not None else cfg.LLM_NAME
    validator_enabled = request.validator_enabled if request.validator_enabled is not None else cfg.VALIDATOR_ENABLED
    response = await summarize(request.medical_text, user_profile, request.task, request.language, validator_enabled, llm_name)
    return jsonable_encoder(response)


@router.post("/converse", response_model=ConverseResponse)
async def converse_v3(request: ConverseRequest):
    user_profile = helpers.update_user_profile(request.user_profile.model_dump())
    llm_name = request.llm_name or cfg.LLM_NAME
    response = await converse(request.user_question, request.medical_text, user_profile, request.chat_id, llm_name)
    return jsonable_encoder(response)


# @router.post("/converse", response_model=ConverseResponse)
# async def converse_v3(request: ConverseRequest):
#     chat_id_exists = await helpers.check_key_pattern_exists(request.chat_id)
#     response = ""

#     if (chat_id_exists):
#         response = await resume_chat(request.user_question, request.chat_id)
#     else:
#         user_profile = helpers.update_user_profile(
#             request.user_profile.model_dump())
#         llm_name = request.llm_name or cfg.LLM_NAME
#         response = await initiate_chat(request.user_question, request.medical_text, user_profile, request.chat_id, llm_name)
#     return jsonable_encoder(response)
