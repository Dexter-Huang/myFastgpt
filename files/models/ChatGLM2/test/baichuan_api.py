from typing import List

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from entity import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice, ChatMessage, DeltaMessage, EmbeddingRequest, EmbeddingResponse
from fastapi import Depends, FastAPI, HTTPException, Request
from sse_starlette.sse import EventSourceResponse
import time
import random
import string

def load_baichuan2_models():
    print("本次加载的大语言模型为: Baichuan-7B-Chat")
    model_name = "/home/lihl/my_openai_api/Baichuan2-13B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    return tokenizer, model

async def baichuan2_create_chat_completion(
    request: ChatCompletionRequest, model, tokenizer
):
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
        generate = predict(messages, request.model)
        return EventSourceResponse(generate, media_type="text/event-stream")

    response = '本接口不支持非stream模式'
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop"
    )
    id = 'chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW'

    return ChatCompletionResponse(id=id, model=request.model, choices=[choice_data], object="chat.completion")

async def predict(messages: List[List[str]], model_id: str, model, tokenizer):
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

    for new_response in model.chat(tokenizer, messages, stream=True):
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

def generate_id():
    possible_characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choices(possible_characters, k=29))
    return 'chatcmpl-' + random_string