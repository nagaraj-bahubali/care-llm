from enum import Enum
from typing import Optional
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, AzureChatOpenAI
import config as cfg


class LLMProvider(Enum):
    GROQ = "groq"
    OLLAMA = "ollama"
    AZURE = "azure"


class LLMFactory:
    def __init__(self):

        self.provider_models = {
            LLMProvider.GROQ: ["gemma-7b-it", "gemma-9b-it", "distil-whisper-large-v3-en", "llama-3.1-70b-versatile", "llama-3.1-8b-instant", "llama-3.2-11b-text-preview", "llama-3.2-1b-preview", "llama-3.2-3b-preview", "llama-3.2-90b-text-preview", "llama-guard-3-8b", "llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "llava-v1.5-7b-4096-preview"],
            LLMProvider.OLLAMA: ["codegemma:7b", "codellama:70b", "gemma2:27b", "gemma2:2b", "gemma2:9b", "llama3.1:405b", "llama3.1:70b", "llama3.1:8b", "llama3.2:1b", "llama3.2:3b", "meditron:70b", "meditron:7b", "mistral-large:123b", "mistral-nemo:12b", "mistral:7b", "mixtral:8x22b", "mixtral:8x7b", "mxbai-embed-large:335m"],
            LLMProvider.AZURE: ["gpt-35-turbo-0613", "gpt-35-turbo-16k-0613", "gpt-4o-2024-05-13", "gpt-4-1106-preview", "gpt-4-32k-0613",
                                "text-embedding-ada-002-2", "gpt-4o-mini-2024-07-18", "text-embedding-3-large-1", "text-embedding-ada-002-2"]
        }

    def _get_provider(self, model_name: str) -> Optional[LLMProvider]:
        """
        Determine the provider based on the model name.

        Args:
            model_name: Name of the model to be used

        Returns:
            LLMProvider enum value or None if model not found
        """
        for provider, models in self.provider_models.items():
            if model_name.lower() in models:
                return provider
        return None

    def load_llm(self, model_name: str):
        """
        Return an LLM instance based on the model name.

        Args:
            model_name: Name of the model to be initialized

        Returns:
            Initialized LLM instance

        Raises:
            ValueError: If model name is not recognized or provider cannot be determined
        """
        provider = self._get_provider(model_name)

        if not provider:
            raise ValueError(f"Model '{model_name}' not recognized. Available models: "
                             f"{', '.join([m for models in self.provider_models.values() for m in models])}")

        if provider == LLMProvider.GROQ:
            return ChatGroq(
                model_name=model_name,
                api_key=cfg.GROQ_API_KEY,
                temperature=0
            )
        elif provider == LLMProvider.OLLAMA:
            return ChatOpenAI(
                model=model_name,
                base_url=cfg.FIT_OLLAMA_API_ENDPOINT,
                api_key=cfg.FIT_OLLAMA_API_KEY,
                temperature=0
            )
        elif provider == LLMProvider.AZURE:
            return AzureChatOpenAI(
                azure_deployment=model_name,
                api_version=cfg.FIT_AZURE_API_VERSION,
                api_key=cfg.FIT_AZURE_API_KEY,
                azure_endpoint=cfg.FIT_AZURE_API_ENDPOINT,
                temperature=0
            )

        raise ValueError(f"Unsupported provider: {provider}")
