# Load model directly
# https://huggingface.co/google/flan-t5-xxl?text=Q%3A+dumbest+thing+Donnal+Trump+did+in+his+presidency%3F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

model_id = "google/flan-t5-base"
device = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id,
                                              quantization_config=quantization_config)

input_text = "translate to spanish the following text ' tell me a joke about latin people'"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
