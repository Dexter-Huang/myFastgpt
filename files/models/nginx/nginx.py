# coding=utf-8
import argparse
import time
from typing import List, Literal, Optional, Union

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
from starlette.status import HTTP_401_UNAUTHORIZED

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmbeddingRequest(BaseModel):
    input: List[str]
    model: str


class EmbeddingResponse(BaseModel):
    data: list
    model: str
    object: str
    usage: dict

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    def __str__(self)->str:
        return self.role+": "+self.content


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


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[
        Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]
    ]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))

async def verify_token(request: Request):
    auth_header = request.headers.get('Authorization')
    if auth_header:
        token_type, _, token = auth_header.partition(' ')
        if (
            token_type.lower() == "bearer"
            and token == "sk-aaabbbcccdddeeefffggghhhiiijjjkkk"
        ):  # 这里配置你的token
            return True
    raise HTTPException(
        status_code=HTTP_401_UNAUTHORIZED,
        detail="Invalid authorization credentials",
    )

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
        request: ChatCompletionRequest, token: bool = Depends(verify_token)
):
    if request.model == 'chatglm2':
        # 转发请求给8000端口
        url = "http://localhost:8000/v1/chat/completions"
    elif request.model == 'baichuan2':
        # 转发请求给5000端口
        url = "http://localhost:5000/v1/chat/completions"
    else:
        raise HTTPException(status_code=400, detail="Invalid model")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=request.dict())

    # 处理流式请求

    if request.stream:
        return response.json()


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8001, workers=1)
