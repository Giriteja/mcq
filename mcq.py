import os
import streamlit as st
import openai
import requests
import json
import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np
# Set your OpenAI API key
openai.api_key = os.getenv("openaikey")
chatgpt_url = "https://api.openai.com/v1/chat/completions"

chatgpt_headers = {
    "content-type": "application/json",
    "Authorization":"Bearer {}".format(os.getenv("openaikey"))}
    
tab1, tab2, tab3 = st.tabs(["MCQ", "Summary", "GIY"])
    
def generate_mcq(paragraph,url,headers):
    
    # Define the payload for the chat model
    messages = [
        {"role": "system", "content": f"""Given the following paragraph, generate multiple-choice questions that align with specific cognitive levels according to Bloom's Taxonomy. For each question, use the associated verbs as a guide to ensure the questions match the intended complexity and cognitive process.
.For each question classify it as Easy,Medium or Hard.

1. Remember (recall facts and basic concepts): Use verbs like "list," "define," "name." 
   - Example Question: "[Question based on 'remember' level]" [Easy]
     a) Option A
     b) Option B
     c) Option C 
     d) Option D
     Answer: C

2. Understand (explain ideas or concepts): Use verbs like "summarize," "describe," "interpret."
   - Example Question: "[Question based on 'understand' level]" [Hard]
     a) Option A
     b) Option B (Correct Answer) 
     c) Option C
     d) Option D
     Answer: A
     Level:Easy

3. Apply (use information in new situations): Use verbs like "use," "solve," "demonstrate."
   - Example Question: "[Question based on 'apply' level]" [Medium]
     a) Option A
     b) Option B
     c) Option C (Correct Answer)
     d) Option D
     Answer: D
     Level:Medium

4. Analyze (draw connections among ideas): Use verbs like "classify," "compare," "contrast."
   - Example Question: "[Question based on 'analyze' level]" [Hard]
     a) Option A
     b) Option B (Correct Answer)
     c) Option C
     d) Option D
     Answer: B
     Level:Hard

5. Evaluate (justify a stand or decision): Use verbs like "judge," "evaluate," "critique."
   - Example Question: "[Question based on 'evaluate' level]" [Medium]
     a) Option A
     b) Option B
     c) Option C (Correct Answer)
     d) Option D
     Answer: E
     Level:Medium

6. Create (produce new or original work): Use verbs like "design," "assemble," "construct."
   - Example Question: "[Question based on 'create' level]"
Please ensure the questions and options are closely related to the content of the provided text and reflect the cognitive level specified for every question.
"""
},
        {"role": "user", "content": paragraph}
    ]

    chatgpt_payload = {
        "model": "gpt-3.5-turbo-16k",
        "messages": messages,
        "temperature": 1.3,
        "max_tokens": 2000,
        "top_p": 1,
        "stop": ["###"]
    }

    # Make the request to OpenAI's API
    response = requests.post(url, json=chatgpt_payload, headers=headers)
    response_json = response.json()

    # Extract data from the API's response
    #st.write(response_json)
    output = response_json['choices'][0]['message']['content']
    return output


def generate_summary(paragraph,url,headers,prompt):
    
    # Define the payload for the chat model
    messages = [
        {"role": "system", "content": "Give Detail Summarization of the given content with headings.Also consider the following Instruction while generating Summary"+prompt},
        {"role": "user", "content": paragraph}
    ]

    chatgpt_payload = {
        "model": "gpt-3.5-turbo-16k",
        "messages": messages,
        "temperature": 1.3,
        "max_tokens": 10000,
        "top_p": 1,
        "stop": ["###"]
    }

    # Make the request to OpenAI's API
    response = requests.post(url, json=chatgpt_payload, headers=headers)
    response_json = response.json()

    # Extract data from the API's response
    #st.write(response_json)
    output = response_json['choices'][0]['message']['content']
    return output
# Streamlit app layout
# Function to read the image and extract text
def extract_text(image):
    # Convert the image to a format suitable for OCR
    opencv_image = np.array(image)  # Convert PIL image to OpenCV format
    gray_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

    # Apply Tesseract OCR
    extracted_text = pytesseract.image_to_string(gray_image)

    return extracted_text


with(tab1):
	# Upload image
	uploaded_image = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])
	#option = st.selectbox(
    	#'Choose Number of Questions:',
    	#('5', '10', '15', '20'))
	# If an image is uploaded, display and process it
	if uploaded_image is not None:
	    # Display the uploaded image
		image = Image.open(uploaded_image)
		st.image(image, caption="Uploaded Image", use_column_width=True)

		# Extract text
		text = extract_text(image)
		paragraph = st.text_area("Enter a paragraph:",text, height=200)
		#prompt = st.text_area("Enter the prompt:", height=200)
		if st.button("Generate MCQs"):
			if paragraph:
		  		mcqs = generate_mcq(paragraph,chatgpt_url,chatgpt_headers)
		  		st.write(mcqs)
		else:
			st.write("Please enter a paragraph to generate questions.")
			
		  
	if uploaded_image is None:         
		paragraph = st.text_area("Enter a paragraph:", height=200)
		#prompt = st.text_area("Enter the prompt:", height=200)
		if st.button("Generate MCQs via text"):
			if paragraph:
				mcqs = generate_mcq(paragraph,chatgpt_url,chatgpt_headers)
				st.write(mcqs)
			else:
				st.write("Please enter a paragraph to generate questions.")
				
with(tab2):
	paragraph = st.text_area("Enter the text:", height=200)
	promptsum = st.text_area("Enter the prompt:",key="sum", height=200)
	if st.button("Generate Summary via text"):
		if paragraph:
			summ = generate_summary(paragraph,chatgpt_url,chatgpt_headers,promptsum)
			st.write(summ)
		else:
			st.write("Please enter the text to generate Summary.")
