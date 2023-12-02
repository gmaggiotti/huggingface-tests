from transformers import T5Tokenizer, T5ForConditionalGeneration

model_id = "google/flan-ul2"
device = "mps"
tokenizer = T5Tokenizer.from_pretrained(model_id, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_id, device_map="auto",offload_folder="offload_flan_ul2")

input_text = "what is X^2 + Y^2 + z^2= r^2"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
