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
import io
from IPython.display import Image, display, HTML
from PIL import Image
import base64 
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']

# Helper function
import requests, json

#Summarization endpoint
def get_completion(inputs, parameters=None,ENDPOINT_URL=os.environ['HF_API_SUMMARY_BASE']): 
    headers = {
      "Authorization": f"Bearer {hf_api_key}",
      "Content-Type": "application/json"
    }
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST",
                                ENDPOINT_URL, headers=headers,
                                data=json.dumps(data)
                               )
    return json.loads(response.content.decode("utf-8"))

API_URL = os.environ['HF_API_NER_BASE'] #NER endpoint
text = "My name is Andrew, I'm building DeepLearningAI and I live in California"
get_completion(text, parameters=None, ENDPOINT_URL= API_URL)

import gradio as gr
def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    return {"text": input, "entities": output}

gr.close_all()
demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    allow_flagging="never",
                    #Here we introduce a new tag, examples, easy to use examples for your application
                    examples=["My name is Andrew and I live in California", "My name is Poli and work at HuggingFace"])
demo.launch(share=True, server_port=int(os.environ['PORT4']))

def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            # If current token continues the entity of the last one, merge them
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            # Otherwise, add the token to the list
            merged_tokens.append(token)

    return merged_tokens

def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    merged_tokens = merge_tokens(output)
    return {"text": input, "entities": merged_tokens}

gr.close_all()
demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    allow_flagging="never",
                    examples=["My name is Andrew, I'm building DeeplearningAI and I live in California", "My name is Poli, I live in Vienna and work at HuggingFace"])

demo.launch(share=True, server_port=int(os.environ['PORT4']))

```

### OUTPUT:

<img width="1351" height="765" alt="Screenshot 2025-10-31 113443" src="https://github.com/user-attachments/assets/7edafbce-3e2e-4fb5-b9c4-e2bd4614a4f6" />

<img width="1314" height="775" alt="Screenshot 2025-10-31 113456" src="https://github.com/user-attachments/assets/0e1af52a-8a6f-4700-9cad-9e4850028f96" />

### RESULT:

The Named Entity Recognition (NER) prototype was successfully developed using the fine-tuned BERT model (dslim/bert-base-NER) and deployed through the Gradio interface.
The system efficiently identifies and highlights entities such as names, locations, and organizations from user-provided text input.
