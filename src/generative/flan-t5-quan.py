from transformers import T5Tokenizer, T5ForConditionalGeneration

model_id = "google/flan-t5-xxl"
tokenizer = T5Tokenizer.from_pretrained(model_id)
model = T5ForConditionalGeneration.from_pretrained(model_id, device_map="auto", load_in_8bit=True)

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cpu")

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))

