from fastapi import APIRouter
from models.v2.schemas import SummarizeRequest, SummarizeResponse, InitiateChatRequest, ResumeChatRequest, ChatResponse
from services.v2.summarization import summarize
from services.v2.conversation import initiate_chat, resume_chat
from fastapi.encoders import jsonable_encoder
from utils import helpers

router = APIRouter()


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_v2(request: SummarizeRequest):
    user_profile = helpers.update_user_profile(
        request.user_profile.model_dump())
    simplified_text = await summarize(request.medical_text, user_profile, request.task)
    response = SummarizeResponse(simplified_text=simplified_text)
    return jsonable_encoder(response)


@router.post("/initiate-chat", response_model=ChatResponse)
async def initiate_chat_v2(request: InitiateChatRequest):
    user_profile = helpers.update_user_profile(
        request.user_profile.model_dump())
    reply = await initiate_chat(request.user_question, request.medical_text, user_profile, request.chat_id)
    response = ChatResponse(reply=reply)
    return jsonable_encoder(response)


@router.post("/resume-chat", response_model=ChatResponse)
async def resume_chat_v2(request: ResumeChatRequest):
    reply = await resume_chat(request.user_question, request.chat_id)
    response = ChatResponse(reply=reply)
    return jsonable_encoder(response)
