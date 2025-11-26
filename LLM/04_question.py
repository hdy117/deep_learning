import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# 1. 加载QA模型（基于BERT微调）
model_name = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# 2. 上下文和问题
context = """
Hugging Face is a company based in New York City that develops tools for natural language processing. 
Its most popular product is the Transformers library, which provides pre-trained models for tasks like text classification, translation, and question answering.
"""
question = "Where is Hugging Face based?"

# 3. 编码（QA任务需要同时传入question和context）
inputs = tokenizer(
    question,
    context,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
)

# 4. 推理
with torch.no_grad():
    outputs = model(**inputs)

# 5. 解析结果（QA模型输出start_logits和end_logits，对应答案在上下文的起止位置）
start_scores = outputs.start_logits
end_scores = outputs.end_logits

# 找到得分最高的起止位置
start_idx = torch.argmax(start_scores).item()
end_idx = torch.argmax(end_scores).item() + 1  # 左闭右开区间

# 解码答案（使用input_ids映射回文本）
answer = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx], skip_special_tokens=True)

print(f"问题：{question}")
print(f"答案：{answer}")