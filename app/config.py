from dotenv import load_dotenv
import os
import yaml

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FIT_AZURE_API_KEY = os.getenv("FIT_AZURE_API_KEY")
FIT_OLLAMA_API_KEY = os.getenv("FIT_OLLAMA_API_KEY")

FIT_AZURE_API_ENDPOINT = os.getenv("FIT_AZURE_API_ENDPOINT")
FIT_OLLAMA_API_ENDPOINT = os.getenv("FIT_OLLAMA_API_ENDPOINT")
FIT_AZURE_API_VERSION = os.getenv("FIT_AZURE_API_VERSION", '2024-10-01-preview')

LLM_NAME = os.getenv("LLM_NAME", "gpt-4o-2024-05-13")

# Redis Properties
REDIS_HOST = os.getenv("REDIS_HOST", "193.175.161.3")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# Validator Properties
VALIDATOR_URL = "http://validator:8002/v1/validate"
VALIDATOR_ENABLED = os.getenv("VALIDATOR_ENABLED", "true").lower() == "true"
VALIDATOR_REPEAT_COUNT = int(os.getenv("VALIDATOR_REPEAT_COUNT", 3))
VALIDATOR_SCORE_THRESHOLD = float(os.getenv("VALIDATOR_SCORE_THRESHOLD", 60))

# The path is wrt WORKDIR in Dockerfile 
PROMPTS_PATH = "/care-llm/app/prompts.yml"

def load_prompts():
    with open(PROMPTS_PATH, "r") as file:
        return yaml.safe_load(file)
