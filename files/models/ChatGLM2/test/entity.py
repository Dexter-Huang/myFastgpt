import time
from dataclasses import Field
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, validator



class EmbeddingRequest(BaseModel):
    input: List[str]
    model: str


class EmbeddingResponse(BaseModel):
    data: list
    model: str
    object: str
    usage: dict

# class ChatMessage(BaseModel):
#     role: Literal["user", "assistant", "system"]
#     content: str
#     def __str__(self)->str:
#         return self.role+": "+self.content

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

    @validator('role')
    def check_role(cls, v):
        if v not in ["user", "assistant", "system"]:
            raise ValueError('role must be one of "user", "assistant", "system"')
        return v


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


# class ChatCompletionResponse(BaseModel):
#     model: str
#     object: Literal["chat.completion", "chat.completion.chunk"]
#     choices: List[
#         Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]
#     ]
#     created: Optional[int] = Field(default_factory=lambda: int(time.time()))

from typing import List, Optional, Union
from pydantic import BaseModel, Field, validator
import time

class ChatCompletionResponse(BaseModel):
    id: str
    object: str

    @validator('object')
    def check_object(cls, v):
        if v not in ["chat.completion", "chat.completion.chunk"]:
            raise ValueError("object必须是'chat.completion'或'chat.completion.chunk'之一")
        return v

    created: Optional[int] = Field(default_factory=lambda: int(time.time()), default=None, init=True, repr=True, hash=True, compare=True, metadata=None, kw_only=False)
    model: str
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]



# 百川

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[str] = []  # Assuming ModelCard is a string type. Replace with the correct type if not.


