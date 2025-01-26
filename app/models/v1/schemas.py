from datetime import date
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel


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


class UserProfile(BaseModel):
    gender: Gender
    birthday: date
    martialStatus: Optional[MartialStatus] = None
    religion: Optional[str] = None
    allergies: Optional[List[str]] = None
    educationalBackground: EducationalBackground
    professionalBackground: ProfessionalBackground
    languagePreferences: List[LanguagePreference]


class SummarizeRequest(BaseModel):
    medical_text: str
    user_profile: UserProfile


class SummarizeResponse(BaseModel):
    simplified_text: str


class ConverseRequest(BaseModel):
    user_question: str
    medical_text: str
    user_profile: UserProfile
    chat_id: str


class ConverseResponse(BaseModel):
    reply: str
