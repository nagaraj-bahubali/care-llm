from fastapi import APIRouter
from schema import ValidationRequest, ValidationResponse
from validation import validate

router = APIRouter()

@router.post("/validate", response_model=ValidationResponse)
async def validate_v1(request: ValidationRequest):
    validation_result = await validate(request.simplified_text, request.original_text)
    return ValidationResponse(
        status_code=validation_result["status_code"],
        similarity=validation_result["similarity"],
        error_response=validation_result["error_response"]
    )