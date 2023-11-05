from transformers import AutoTokenizer, AutoModel

tools = [
    {
        "name": "track",
        "description": "追踪指定股票的实时价格",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "description": "需要追踪的股票代码"
                }
            },
            "required": ['symbol']
        }
    },
    {
        "name": "text-to-speech",
        "description": "将文本转换为语音",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "description": "需要转换成语音的文本"
                },
                "voice": {
                    "description": "要使用的语音类型（男声、女声等）"
                },
                "speed": {
                    "description": "语音的速度（快、中等、慢等）"
                }
            },
            "required": ['text']
        }
    }
]
system_info = {"role": "system", "content": "Answer the following questions as best as you can. You have access to the following tools:", "tools": tools}

chatglm3_model_path = '/home/huangml/ChatGLM3/model/chatglm3-6b'
print('本次加载的大语言模型为: ChatGLM3-6B-Chat')
chatglm3_tokenizer = AutoTokenizer.from_pretrained(chatglm3_model_path, trust_remote_code=True)
chatglm3_model = AutoModel.from_pretrained(chatglm3_model_path, trust_remote_code=True).cuda()

history = [system_info]
query = "帮我查询股票10111的价格"
response, history = chatglm3_model.chat(chatglm3_tokenizer , query, history=history)
print(response)