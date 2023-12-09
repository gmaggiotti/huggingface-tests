from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import LlamaModelForCausalLM

model_id = "princeton-nlp/Sheared-LLaMA-1.3B-ShareGPT"
device = "mps"
tokenizer = T5Tokenizer.from_pretrained(model_id, legacy=False)
model = LlamaModelForCausalLM.from_pretrained(model_id, device_map="auto")

input_text = "tell me a joke"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))