# Load model directly
# https://huggingface.co/google/flan-t5-xxl?text=Q%3A+dumbest+thing+Donnal+Trump+did+in+his+presidency%3F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "google/flan-t5-xxl"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

input_text = "Q:who was the last president of US?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))