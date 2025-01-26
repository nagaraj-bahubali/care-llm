from fastapi import FastAPI
from api.v1 import endpoints as v1_endpoints
from api.v2 import endpoints as v2_endpoints
from api.v3 import endpoints as v3_endpoints

app = FastAPI(title="CARE-LLM")

app.include_router(v1_endpoints.router, prefix="/v1")
app.include_router(v2_endpoints.router, prefix="/v2")
app.include_router(v3_endpoints.router, prefix="/v3")
