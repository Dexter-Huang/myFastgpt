# coding=utf-8
import argparse
import json
import random
import string
import time
from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Union

import numpy as np
import tiktoken
import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import PolynomialFeatures
from sse_starlette.sse import EventSourceResponse
from starlette.status import HTTP_401_UNAUTHORIZED
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, GenerationConfig


@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "observation"]
    content: str
    def __str__(self)->str:
        return self.role+": "+self.content


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system", "observation"]] = None
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


class EmbeddingRequest(BaseModel):
    input: List[str]
    model: str


class EmbeddingResponse(BaseModel):
    data: list
    model: str
    object: str
    usage: dict


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding('cl100k_base')
    num_tokens = len(encoding.encode(string))
    return num_tokens


def expand_features(embedding, target_length):
    poly = PolynomialFeatures(degree=2)
    expanded_embedding = poly.fit_transform(embedding.reshape(1, -1))
    expanded_embedding = expanded_embedding.flatten()
    if len(expanded_embedding) > target_length:
        # 如果扩展后的特征超过目标长度，可以通过截断或其他方法来减少维度
        expanded_embedding = expanded_embedding[:target_length]
    elif len(expanded_embedding) < target_length:
        # 如果扩展后的特征少于目标长度，可以通过填充或其他方法来增加维度
        expanded_embedding = np.pad(
            expanded_embedding, (0, target_length - len(expanded_embedding))
        )
    return expanded_embedding


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest
):
    for message in request.messages:
        print(message)
    global chatglm2_model, chatglm2_tokenizer, chatglm3_model, chatglm3_tokenizer, baichuan_model, baichuan_tokenizer

    if request.model == 'chatglm2' :
        if request.messages[-1].role != "user":
            raise HTTPException(status_code=400, detail="Invalid request")
        query = request.messages[-1].content

        prev_messages = request.messages[:-1]
        if len(prev_messages) > 0 and prev_messages[0].role == "system":
            query = prev_messages.pop(0).content + query

        history = []
        if len(prev_messages) % 2 == 0:
            for i in range(0, len(prev_messages), 2):
                if (
                        prev_messages[i].role == "user"
                        and prev_messages[i + 1].role == "assistant"
                ):
                    history.append([prev_messages[i].content, prev_messages[i + 1].content])

        if request.stream:
            generate = chatglm2_predict(query, history, request.model)
            return EventSourceResponse(generate, media_type="text/event-stream")

        response, _ = chatglm2_model.chat(chatglm2_tokenizer, query, history=history)
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=response),
            finish_reason="stop",
        )
        return ChatCompletionResponse(
            model=request.model, choices=[choice_data], object="chat.completion"
        )

    elif request.model == 'baichuan2':
        if request.messages[-1].role != "user":
            raise HTTPException(status_code=400, detail="Invalid request")
        query = request.messages[-1].content
        prev_messages = request.messages[:-1]
        if len(prev_messages) > 0 and prev_messages[0].role == "system":
            query = prev_messages.pop(0).content + query
        messages = []
        for message in prev_messages:
            messages.append({"role": message.role, "content": message.content})

        messages.append({"role": "user", "content": query})

        if request.stream:
            generate = baichuan2_predict(messages, request.model)
            return EventSourceResponse(generate, media_type="text/event-stream")

        response = '本接口不支持非stream模式'
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=response),
            finish_reason="stop"
        )
        id = 'chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW'

        return ChatCompletionResponse(id=id, model=request.model, choices=[choice_data], object="chat.completion")
    else : # chatglm3
        tools = [
            {'name': 'track', 'description': '追踪指定股票的实时价格',
             'parameters': {'type': 'object', 'properties': {'symbol': {'description': '需要追踪的股票代码'}},
                            'required': []}},
            {'name': '/text-to-speech', 'description': '将文本转换为语音', 'parameters': {'type': 'object',
                                                                                          'properties': {'text': {
                                                                                              'description': '需要转换成语音的文本'},
                                                                                                         'voice': {
                                                                                                             'description': '要使用的语音类型（男声、女声等）'},
                                                                                                         'speed': {
                                                                                                             'description': '语音的速度（快、中等、慢等）'}},
                                                                                          'required': []}},
            {'name': '/image_resizer', 'description': '调整图片的大小和尺寸', 'parameters': {'type': 'object',
                                                                                             'properties': {
                                                                                                 'image_file': {
                                                                                                     'description': '需要调整大小的图片文件'},
                                                                                                 'width': {
                                                                                                     'description': '需要调整的宽度值'},
                                                                                                 'height': {
                                                                                                     'description': '需要调整的高度值'}},
                                                                                             'required': []}},
            {'name': '/foodimg', 'description': '通过给定的食品名称生成该食品的图片',
             'parameters': {'type': 'object', 'properties': {'food_name': {'description': '需要生成图片的食品名称'}},
                            'required': []}}]
        system_item = {"role": "system",
                       "content": "Answer the following questions as best as you can. You have access to the following tools:",
                       "tools": tools}
        if request.messages[-1].role != "user":
            raise HTTPException(status_code=400, detail="Invalid request")
        query = request.messages[-1].content

        prev_messages = request.messages[:-1]
        if len(prev_messages) > 0 and prev_messages[0].role == "system":
            query = prev_messages.pop(0).content + query
        query = system_item["content"] + json.dumps(system_item["tools"]) + query

        history = [system_item]
        if len(prev_messages) % 2 == 0:
            for i in range(0, len(prev_messages), 2):
                if (
                        prev_messages[i].role == "user"
                        and prev_messages[i + 1].role == "assistant"
                ):
                    history.append([prev_messages[i].content, prev_messages[i + 1].content])

        if request.stream:
            generate = chatglm3_predict(query, history, request.model)
            return EventSourceResponse(generate, media_type="text/event-stream")

        response, _ = chatglm3_model.chat(chatglm3_tokenizer, query, history=history)
        print(response)
        if response is not str:
            response = json.dumps(response)
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=response),
            finish_reason="stop",
        )
        return ChatCompletionResponse(
            model=request.model, choices=[choice_data], object="chat.completion"
        )

def generate_id():
    possible_characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choices(possible_characters, k=29))
    return 'chatcmpl-' + random_string

async def chatglm2_predict(query: str, history: List[List[str]], model_id: str):
    global chatglm2_model, chatglm2_tokenizer

    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(role="assistant"), finish_reason=None
    )
    chunk = ChatCompletionResponse(
        model=model_id, choices=[choice_data], object="chat.completion.chunk"
    )
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    current_length = 0

    for new_response, _ in chatglm2_model.stream_chat(chatglm2_tokenizer, query, history):
        if len(new_response) == current_length:
            continue

        new_text = new_response[current_length:]
        current_length = len(new_response)

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(content=new_text),
            finish_reason=None
        )
        chunk = ChatCompletionResponse(
            model=model_id,
            choices=[choice_data],
            object="chat.completion.chunk"
        )
        yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(), finish_reason="stop"
    )
    chunk = ChatCompletionResponse(
        model=model_id,
        choices=[choice_data],
        object="chat.completion.chunk"
    )
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))
    yield '[DONE]'

async def chatglm3_predict(query: str, history: List[List[str]], model_id: str):
    global chatglm3_model, chatglm3_tokenizer

    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(role="assistant"), finish_reason=None
    )
    chunk = ChatCompletionResponse(
        model=model_id, choices=[choice_data], object="chat.completion.chunk"
    )
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    current_length = 0
    past_key_values, history = None, []

    for new_response, history, past_key_values in chatglm3_model.stream_chat(chatglm3_tokenizer, query, history=history, past_key_values=past_key_values, return_past_key_values=True):
        if len(new_response) == current_length:
            continue

        new_text = new_response[current_length:]
        current_length = len(new_response)

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(content=new_text),
            finish_reason=None
        )
        chunk = ChatCompletionResponse(
            model=model_id,
            choices=[choice_data],
            object="chat.completion.chunk"
        )
        yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(), finish_reason="stop"
    )
    chunk = ChatCompletionResponse(
        model=model_id,
        choices=[choice_data],
        object="chat.completion.chunk"
    )
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))
    yield '[DONE]'


async def baichuan2_predict(messages: List[List[str]], model_id: str):
    global baichuan_model, baichuan_tokenizer
    id = generate_id()
    created = int(time.time())
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant", content=""),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(id=id, object="chat.completion.chunk", created=created, model=model_id,
                                   choices=[choice_data])
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    current_length = 0

    for new_response in baichuan_model.chat(baichuan_tokenizer, messages, stream=True):
        if len(new_response) == current_length:
            continue

        new_text = new_response[current_length:]
        current_length = len(new_response)

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(content=new_text),
            finish_reason=None
        )
        chunk = ChatCompletionResponse(id=id, object="chat.completion.chunk", created=created, model=model_id,
                                       choices=[choice_data])
        yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(id=id, object="chat.completion.chunk", created=created, model=model_id,
                                   choices=[choice_data])
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))
    yield '[DONE]'

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(
    request: EmbeddingRequest, token: bool = Depends(verify_token)
):
    # 计算嵌入向量和tokens数量
    embeddings = [embeddings_model.encode(text) for text in request.input]

    # 如果嵌入向量的维度不为1536，则使用插值法扩展至1536维度
    embeddings = [
        expand_features(embedding, 1536) if len(embedding) < 1536 else embedding
        for embedding in embeddings
    ]

    # Min-Max normalization 归一化
    embeddings = [embedding / np.linalg.norm(embedding) for embedding in embeddings]

    # 将numpy数组转换为列表
    embeddings = [embedding.tolist() for embedding in embeddings]
    prompt_tokens = sum(len(text.split()) for text in request.input)
    total_tokens = sum(num_tokens_from_string(text) for text in request.input)

    response = {
        "data": [
            {"embedding": embedding, "index": index, "object": "embedding"}
            for index, embedding in enumerate(embeddings)
        ],
        "model": request.model,
        "object": "list",
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
        },
    }
    return response


if __name__ == "__main__":
    chatglm2_model_path = '/home/huangml/ChatGLM2-6B/model'
    print("本次加载的大语言模型为: ChatGLM2-6B-Chat")
    chatglm2_tokenizer = AutoTokenizer.from_pretrained(chatglm2_model_path, trust_remote_code=True)
    chatglm2_model = AutoModel.from_pretrained(chatglm2_model_path, trust_remote_code=True).cuda()

    chatglm3_model_path = '/home/huangml/ChatGLM3/model/chatglm3-6b'
    print('本次加载的大语言模型为: ChatGLM3-6B-Chat')
    chatglm3_tokenizer = AutoTokenizer.from_pretrained(chatglm3_model_path, trust_remote_code=True)
    chatglm3_model = AutoModel.from_pretrained(chatglm3_model_path, trust_remote_code=True).cuda()

    print("本次加载的大语言模型为: Baichuan-13B-Chat")
    baichuan2_model_path = "/home/lihl/my_openai_api/Baichuan2-13B-Chat"
    baichuan_tokenizer = AutoTokenizer.from_pretrained(
        baichuan2_model_path, use_fast=False, trust_remote_code=True)
    baichuan_model = AutoModelForCausalLM.from_pretrained(
        baichuan2_model_path, device_map="auto", trust_remote_code=True)
    baichuan_model.generation_config = GenerationConfig.from_pretrained(baichuan2_model_path)


    embeddings_model = SentenceTransformer('/home/huangml/mokai_m3e_base', device='cpu')
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)