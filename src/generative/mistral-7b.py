from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mistral-7B-v0.1"
device = "cpu"  # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto", offload_folder="offload_mistral_7B")

prompt = "My favourite condiment is"
model_inputs = tokenizer([prompt], return_tensors="pt").input_ids.to(device)

generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
print(tokenizer.batch_decode(generated_ids)[0])
