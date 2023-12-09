import requests
import json

input = '''summarize the following text: JetBlue Agent: Hello, thank you for reaching out to JetBlue. My name is Alex, how may I assist you today? \
Customer: Hi Alex, I am Bryan. I need to reschedule my flight due to some personal commitments. The booking reference is JB123456.\
JetBlue Agent: Hi Bryan, Ill be glad to assist with that. Could you please tell me your original flight date and the preferred new date? \
stomer: The original flight date was this Friday, 8th April. But, now, I want to reschedule it to next Friday, 15th April. \
JetBlue Agent: Thank you for the details, Bryan. Let me pulling up your booking and check the availability for you. Please bear with me for a moment.
Customer: Sure, take your time. \
JetBlue Agent: Thanks for waiting, Bryan. Im pleased to inform you that there is availability on 15th April. There is a morning flight at 9:30 AM and an evening flight at 6:30 PM. Which would you prefer? \
    Customer: I would prefer the evening flight at 6:30 PM. \
JetBlue Agent: Great! I just rescheduled your flight to 15th April 202 '''

input2 = "tell me a joke about LLMs"
data = {
    "inputs": input2,
    "parameters": {
        "max_new_tokens": 200
    }
}

headers = {
    'Content-Type': 'application/json'
}

# response = requests.post('https://llm-server-private.test.asapp.com/generic-flan-t5-large/generate', headers=headers, data=json.dumps(data))
response = requests.post('https://llm-server-private.test.asapp.com/generic-flan-ul2/generate', headers=headers, data=json.dumps(data))


print(response.text)
