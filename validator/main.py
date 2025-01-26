from fastapi import FastAPI
import endpoint

app = FastAPI(title="Validator")

app.include_router(endpoint.router, prefix="/v1")
