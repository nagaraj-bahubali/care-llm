from fastapi import APIRouter
from models.v1.schemas import SummarizeRequest, SummarizeResponse, ConverseRequest, ConverseResponse
from services.v1.summarization import summarize
from services.v1.conversation import converse
from fastapi.encoders import jsonable_encoder
from utils import helpers

router = APIRouter()


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_v1(request: SummarizeRequest):
    user_profile = helpers.update_user_profile(
        request.user_profile.model_dump())
    simplified_text = await summarize(request.medical_text, user_profile)
    response = SummarizeResponse(simplified_text=simplified_text)
    return jsonable_encoder(response)


@router.post("/converse", response_model=ConverseResponse)
async def converse_v1(request: ConverseRequest):
    user_profile = helpers.update_user_profile(
        request.user_profile.model_dump())
    reply = await converse(request.user_question, request.medical_text, user_profile, request.chat_id)
    response = ConverseResponse(reply=reply)
    return jsonable_encoder(response)
