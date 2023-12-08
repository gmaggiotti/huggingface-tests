import requests
import json

data = {
    "inputs": "tell me a joke about LLMs",
    "parameters": {
        "max_new_tokens": 200
    }
}

headers = {
    'Content-Type': 'application/json'
}

response = requests.post('https://llm-server-private.test.asapp.com/generic-flan-t5-large/generate', headers=headers,
                         data=json.dumps(data))

print(response.text)
