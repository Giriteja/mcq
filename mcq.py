import streamlit as st
import openai
import requests
import json
import os
# Set your OpenAI API key
openai.api_key = os.getenv("openaikey")
chatgpt_url = "https://api.openai.com/v1/chat/completions"

chatgpt_headers = {
    "content-type": "application/json",
    "Authorization":"Bearer {}".format(os.getenv("openaikey"))}
    
def generate_mcq(paragraph,url,headers):
    
    # Define the payload for the chat model
    messages = [
        {"role": "system", "content": "You are an expert In generating mcq questions from given paragraph"},
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

# Streamlit app layout
st.title("MCQ Generator")
# Function to read the image and extract text
def extract_text(image):
    # Convert the image to a format suitable for OCR
    opencv_image = np.array(image)  # Convert PIL image to OpenCV format
    gray_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

    # Apply Tesseract OCR
    extracted_text = pytesseract.image_to_string(gray_image)

    return extracted_text

# Upload image
uploaded_image = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

# If an image is uploaded, display and process it
if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Extract text
    text = extract_text(image)
    paragraph = st.text_area("Enter a paragraph:",text, height=200)
    if st.button("Generate MCQs"):
       if paragraph:
          mcqs = generate_mcq(paragraph,chatgpt_url,chatgpt_headers)
          st.write(mcqs)
       else:
          st.write("Please enter a paragraph to generate questions.")
          
if uploaded_image is None:         
	paragraph = st.text_area("Enter a paragraph:", height=200)
	if st.button("Generate MCQs via text"):
		if paragraph:
			mcqs = generate_mcq(paragraph,chatgpt_url,chatgpt_headers)
			st.write(mcqs)
		else:
			st.write("Please enter a paragraph to generate questions.")

