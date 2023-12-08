from transformers import T5Tokenizer, T5ForConditionalGeneration

model_id = "google/flan-ul2"
device = "mps"
tokenizer = T5Tokenizer.from_pretrained(model_id, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_id, device_map="auto", offload_folder="offload_flan_ul2")

input_text = "tell me a short joke about a rabbit and turtle "
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

outputs = model.generate(input_ids, max_length=200)
print(tokenizer.decode(outputs[0]))
