from datetime import date
from enum import Enum
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    DIVERSE = "diverse"


class MartialStatus(str, Enum):
    SINGLE = "single"
    MARRIED = "married"
    DIVORCED = "divorced"
    WIDOWED = "widowed"


class EducationalBackground(str, Enum):
    NONE = "none"
    SCHOOL = "school"
    APPRENTICESHIP = "apprenticeship"
    UNIVERSITY = "university"
    DOCTORATE = "doctorate"
    OTHER = "other"


class ProfessionalBackground(str, Enum):
    NONE = "none"
    MEDICAL = "medical"
    TECHNICAL = "technical"
    FINANCIAL = "financial"
    KNOWLEDGE_WORK = "knowledgeWork"
    HANDICRAFT = "handicraft"
    CONSTRUCTION_WORK = "constructionWork"
    ENGINEER = "engineer"
    CREATIVE = "creative"
    MANAGER = "manager"
    RESEARCHER = "researcher"
    OTHER = "other"


class LanguagePreference(str, Enum):
    AVOID_ABSTRACT_TERMS = "avoidAbstractTerms"
    AVOID_TECHNICAL_TERMS = "avoidTechnicalTerms"
    AVOID_FIGURATIVE_LANGUAGE = "avoidFigurativeLanguage"
    AVOID_ABBREVIATIONS = "avoidAbbreviations"
    SHORT_SENTENCES = "shortSentences"
    ONLY_ACTIVE_SENTENCES = "onlyActiveSentences"
    ONE_STATEMENT_PER_SENTENCE = "oneStatementPerSentence"


class CommunicationLanguage(str, Enum):
    ENGLISH = "English"
    SPANISH = "Spanish"
    FRENCH = "French"
    GERMAN = "German"
    ITALIAN = "Italian"
    PORTUGUESE = "Portuguese"
    CHINESE = "Chinese"
    UKRAINIAN = "Ukrainian"
    ARABIC = "Arabic"


class UserProfile(BaseModel):
    gender: Gender
    birthday: date
    martialStatus: Optional[MartialStatus] = None
    religion: Optional[str] = None
    allergies: Optional[List[str]] = None
    educationalBackground: EducationalBackground
    professionalBackground: ProfessionalBackground
    languagePreferences: List[LanguagePreference]


class TaskType(str, Enum):
    disease = "diseaseInformation"
    diagnosis = "diagnosisReport"


class SummarizeRequest(BaseModel):
    medical_text: str
    user_profile: UserProfile
    language: CommunicationLanguage
    task: TaskType
    validator_enabled: Optional[bool] = Field(
        None, description="Enable or disable validation. Acceptable values: true or false."
    )
    llm_name: Optional[Literal[
        "gemma-7b-it", "gemma-9b-it", "distil-whisper-large-v3-en", "llama-3.1-70b-versatile", "llama-3.1-8b-instant",
        "llama-3.2-11b-text-preview", "llama-3.2-1b-preview", "llama-3.2-3b-preview", "llama-3.2-90b-text-preview",
        "llama-guard-3-8b", "llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "llava-v1.5-7b-4096-preview",
        "codegemma:7b", "codellama:70b", "gemma2:27b", "gemma2:2b", "gemma2:9b", "llama3.1:405b", "llama3.1:70b",
        "llama3.1:8b", "llama3.2:1b", "llama3.2:3b", "meditron:70b", "meditron:7b", "mistral-large:123b",
        "mistral-nemo:12b", "mistral:7b", "mixtral:8x22b", "mixtral:8x7b", "mxbai-embed-large:335m",
        "gpt-35-turbo-0613", "gpt-35-turbo-16k-0613", "gpt-4o-2024-05-13", "gpt-4-1106-preview",
        "gpt-4-32k-0613", "text-embedding-ada-002-2", "gpt-4o-mini-2024-07-18", "text-embedding-3-large-1",
        "text-embedding-ada-002-2"
    ]] = None


class SummarizeResponse(BaseModel):
    simplified_text: str
    validation_score: float


class ConverseRequest(BaseModel):
    user_question: str
    medical_text: str
    user_profile: UserProfile
    chat_id: str
    llm_name: Optional[Literal[
        "gemma-7b-it", "gemma-9b-it", "distil-whisper-large-v3-en", "llama-3.1-70b-versatile", "llama-3.1-8b-instant",
        "llama-3.2-11b-text-preview", "llama-3.2-1b-preview", "llama-3.2-3b-preview", "llama-3.2-90b-text-preview",
        "llama-guard-3-8b", "llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "llava-v1.5-7b-4096-preview",
        "codegemma:7b", "codellama:70b", "gemma2:27b", "gemma2:2b", "gemma2:9b", "llama3.1:405b", "llama3.1:70b",
        "llama3.1:8b", "llama3.2:1b", "llama3.2:3b", "meditron:70b", "meditron:7b", "mistral-large:123b",
        "mistral-nemo:12b", "mistral:7b", "mixtral:8x22b", "mixtral:8x7b", "mxbai-embed-large:335m",
        "gpt-35-turbo-0613", "gpt-35-turbo-16k-0613", "gpt-4o-2024-05-13", "gpt-4-1106-preview",
        "gpt-4-32k-0613", "text-embedding-ada-002-2", "gpt-4o-mini-2024-07-18", "text-embedding-3-large-1",
        "text-embedding-ada-002-2"
    ]] = None


class ConverseResponse(BaseModel):
    reply: str
