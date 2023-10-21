from typing import List

from transformers import AutoTokenizer, AutoModel

from entity import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice, ChatMessage, DeltaMessage, EmbeddingRequest, EmbeddingResponse
from fastapi import Depends, FastAPI, HTTPException, Request
from sse_starlette.sse import EventSourceResponse
import argparse

def load_chatglm2_models():
    print("本次加载的大语言模型为: ChatGLM2-6B")
    model_path = '/home/huangml/ChatGLM2-6B/model'
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="16", type=str, help="Model name")
    args = parser.parse_args()

    model_dict = {
        "4": model_path,
        "8": model_path,
        "16": model_path,
    }

    model_name = model_dict.get(args.model_name, model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).cuda()
    return tokenizer, model

async def chatglm2_create_chat_completion(
    request: ChatCompletionRequest, model, tokenizer
):
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
        generate = predict(query, history, request.model)
        return EventSourceResponse(generate, media_type="text/event-stream")

    response, _ = model.chat(tokenizer, query, history=history)
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop",
    )

    return ChatCompletionResponse(
        model=request.model, choices=[choice_data], object="chat.completion"
    )


async def predict(query: str, history: List[List[str]], model_id: str, model, tokenizer):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(role="assistant"), finish_reason=None
    )
    chunk = ChatCompletionResponse(
        model=model_id, choices=[choice_data], object="chat.completion.chunk"
    )
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    current_length = 0

    for new_response, _ in model.stream_chat(tokenizer, query, history):
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