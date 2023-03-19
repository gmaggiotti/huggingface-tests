from transformers import pipeline
import pandas as pd

text = """Dear Amazon, last week I ordered an Optimus Prime action figure \
from your online store in Germany. Unfortunately, when I opened the package, \
I discovered to my horror that I had been sent an action figure of Megatron \
instead! As a lifelong enemy of the Decepticons, I hope you can understand my \
dilemma. To resolve the issue, I demand an exchange of Megatron for the \
Optimus Prime figure I ordered. Enclosed are copies of my records concerning \
this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""

# # Sentiment classification
# classifier = pipeline("text-classification")
# outputs = classifier(text)
# print(outputs)
#
# # Name Entity Recognition
# ner_tagger = pipeline("ner", aggregation_strategy="simple")
# outputs = ner_tagger(text)
# [print(output) for output in outputs]
#
# # Question Answering
# reader = pipeline("question-answering")
# question = "Was the customer happy?"
# outputs = reader(question=question, context=text)
# print(outputs)
#
# # Summarization
# summarizer = pipeline("summarization")
# outputs = summarizer(text, max_length=56, clean_up_tokenization_spaces=True)
# print(outputs[0]['summary_text'])

# Translator

translator = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")
outputs = translator(text, clean_up_tokenization_spaces=True, min_length=100)
print(outputs[0]['translation_text'])

# using big opus spanish model
translator = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-tc-big-en-es")
outputs = translator(text, clean_up_tokenization_spaces=True, min_length=100)
print(outputs[0]['translation_text'])

print("EOF")