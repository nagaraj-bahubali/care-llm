from pydantic import BaseModel
    
class ValidationRequest(BaseModel):
    simplified_text: str
    original_text: str

class ValidationResponse(BaseModel):
    status_code: int
    similarity: float
    error_response: str
    