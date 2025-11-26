## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:
Natural Language Processing (NLP) requires identifying important entities such as names, places, organizations, and dates from text. Manually detecting these entities is time-consuming and error-prone.
Hence, the goal is to build an automated NER system using a pre-trained transformer model and a user-friendly web interface that:

Accepts text input from the user,

Detects named entities using a pre-trained model,

Highlights and classifies entities for visualization.
### DESIGN STEPS:



#### STEP 1: Import Libraries and Load Environment Variables

Import the necessary Python libraries: os, json, requests, gradio, and dotenv.

Load the .env file to access the Hugging Face API key and model endpoints securely.

#### STEP 2: Define Helper Function for API Calls

Create a get_completion() function that sends HTTP POST requests to the Hugging Face Inference API.

Include Authorization headers for secure access using the API token.

#### STEP 3: Define the Named Entity Recognition (NER) Function

Use the get_completion() function to send input text to the NER model endpoint.

Process the JSON response and extract named entities.

#### STEP 4: Token Merging (Optional Enhancement)

Implement a merge_tokens() helper function to merge subword tokens (e.g., “Cal” + “##ifornia” → “California”) for cleaner entity visualization.

#### STEP 5: Build Gradio Interface

Create a Gradio interface using gr.Interface() with:

Input: Textbox for entering text.

Output: HighlightedText for displaying entities.

Example texts for quick testing.

Launch the application using demo.launch(share=True) to generate a public link for access.

### PROGRAM:
```
import os
import json
import requests
import gradio as gr
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
hf_api_key = os.environ['HF_API_KEY']
API_URL = os.environ['HF_API_NER_BASE']

def get_completion(inputs, parameters=None, ENDPOINT_URL=API_URL):
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }
    data = {"inputs": inputs}
    if parameters:
        data.update({"parameters": parameters})

    response = requests.post(ENDPOINT_URL, headers=headers, data=json.dumps(data))
    text = response.content.decode("utf-8").strip()

    # Handle extra data safely
    try:
        # Try parsing as normal JSON
        return json.loads(text)
    except json.JSONDecodeError:
        # If response contains multiple JSON objects, take the first valid one
        parts = text.split("\n")
        for part in parts:
            try:
                return json.loads(part)
            except Exception:
                continue
        raise ValueError(f"Invalid JSON returned from model: {text}")

def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            last = merged_tokens[-1]
            last['word'] += token['word'].replace('##', '')
            last['end'] = token['end']
            last['score'] = (last['score'] + token['score']) / 2
        else:
            merged_tokens.append(token)
    return merged_tokens

def ner(input_text):
    output = get_completion(input_text)
    if not isinstance(output, list):
        raise ValueError(f"Unexpected model output: {output}")
    merged_tokens = merge_tokens(output)
    return {"text": input_text, "entities": merged_tokens}

gr.close_all()
demo = gr.Interface(
    fn=ner,
    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
    outputs=[gr.HighlightedText(label="Text with entities")],
    title="NER with dslim/bert-base-NER",
    description="Find named entities using the dslim/bert-base-NER model via Hugging Face Inference API.",
    allow_flagging="never",
    examples=[
        "My name is KARTHIK PADMANABAN R , works at HuggingFace in Chennai."
    ]
)

demo.launch(share=True, server_port=int(os.environ.get("PORT3",7860)))

```

### OUTPUT:

<img width="754" height="1010" alt="Screenshot 2025-11-26 091550" src="https://github.com/user-attachments/assets/ae6f0dfb-dde7-4e46-9321-2c2f21526995" />

<img width="856" height="364" alt="Screenshot 2025-11-26 091603" src="https://github.com/user-attachments/assets/b4616657-ed18-4d72-8f28-e114c91281ef" />


### RESULT:

The Named Entity Recognition (NER) prototype was successfully developed using the fine-tuned BERT model (dslim/bert-base-NER) and deployed through the Gradio interface.
The system efficiently identifies and highlights entities such as names, locations, and organizations from user-provided text input.
