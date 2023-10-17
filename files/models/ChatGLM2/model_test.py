from transformers import AutoConfig, AutoModel, AutoTokenizer
import os, torch
CHECKPOINT_PATH = '/home/huangml/ChatGLM2-6B/ptuning/output/adgen-chatglm2-6b-pt-128-2e-2/checkpoint-3000'

model_path = '/home/huangml/ChatGLM2-6B/model'

# 载入Tokenizer
tokenizer= AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)

# 加载P-Tuning的checkPoint
config = AutoConfig.from_pretrained(model_path,trust_remote_code=True,pre_seq_len=128)
model = AutoModel.from_pretrained(model_path,config=config, trust_remote_code=True)
prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH,"pytorch_model.bin"))
print(prefix_state_dict)
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

# model = model.quantize(8)

model = model.half().cuda()
model.transformer.prefix_encoder.float().cuda()
model = model.eval()
response, history = model.chat(tokenizer=tokenizer, query="你好，你是谁？",history=[])
print(response)