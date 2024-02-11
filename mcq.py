import openai
import os
import json
import streamlit as st
# Initialize the OpenAI client with your API key
from openai import OpenAI
import streamlit as st
import requests
from PIL import Image
import pytesseract
import cv2
import numpy as np

client = OpenAI(
  api_key=os.getenv("openaikey"),  # this is also the default, it can be omitted
)

chatgpt_url = "https://api.openai.com/v1/chat/completions"

chatgpt_headers = {
    "content-type": "application/json",
    "Authorization":"Bearer {}".format(os.getenv("openaikey"))}

tab1, tab2, tab3 = st.tabs(["MCQ", "Summary", "GIY"])

paragraph="""Food in the form of a soft slimy substance where some
proteins and carbohydrates have already been broken down
is called chyme. Now the food material passes from the
stomach to the small intestine. Here the ring like muscles
called pyloric sphincters relax to open the passage into
the small intestine. The sphincters are responsible for
regulating the opening of the passage such that only small
quantities of the food material may be passed into the
small intestine at a time."""

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def generateMCQs(questions,topic):
        return json.dumps({"questions": questions, "topic":topic})
        
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
    st.write(response_json)
    output = response_json['choices'][0]['message']['content']
    return output
        
        
def extract_text(image):
    # Convert the image to a format suitable for OCR
    opencv_image = np.array(image)  # Convert PIL image to OpenCV format
    gray_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

    # Apply Tesseract OCR
    extracted_text = pytesseract.image_to_string(gray_image)

    return extracted_text

def run_conversation(paragraph):
    # Step 1: send the conversation and available functions to the model
    messages = [{"role": "system", "content": """Given the following paragraph, generate multiple-choice questions that align with specific cognitive levels according to Bloom's Taxonomy. For each question, use the associated verbs as a guide to ensure the questions match 			the intended complexity and cognitive process.For each question classify it as Easy,Medium or Hard.
    
1. Remember (recall facts and basic concepts): Use verbs like "list," "define," "name." 
   - Example Question: "[Question based on 'remember' level]" 
     a) Option A
     b) Option B
     c) Option C 
     d) Option D

     Answer: C
     
     Level:Easy

2. Understand (explain ideas or concepts): Use verbs like "summarize," "describe," "interpret."
   - Example Question: "[Question based on 'understand' level]" 
     a) Option A
     b) Option B
     c) Option C
     d) Option D

     
     Answer: A
     
     Difficulty Level:Easy


3. Apply (use information in new situations): Use verbs like "use," "solve," "demonstrate."
   - Example Question: "[Question based on 'apply' level]" 
     a) Option A
     b) Option B
     c) Option C 
     d) Option D

     Answer: D
     
     Difficulty Level:Medium


4. Analyze (draw connections among ideas): Use verbs like "classify," "compare," "contrast."
   - Example Question: "[Question based on 'analyze' level]"
     a) Option A
     b) Option B
     c) Option C
     d) Option D

     Answer: B
     
    Difficulty  Level:Hard


5. Evaluate (justify a stand or decision): Use verbs like "judge," "evaluate," "critique."
   - Example Question: "[Question based on 'evaluate' level]"
     a) Option A
     b) Option B
     c) Option C
     d) Option D


     Answer: E
     
    Difficulty Level:Medium

6. Create (produce new or original work): Use verbs like "design," "assemble," "construct."
   - Example Question: "[Question based on 'create' level]"
Please ensure the questions and options are closely related to the content of the provided text and reflect the cognitive level specified for every question."""},{"role": "user", "content": paragraph}]
    tools = [
        {
            "type": "function",
            "function": {
            "name": "generateMCQs",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string"
                    },
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string"
                                },
                                "options": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "answer": {
                                    "type": "string"
                                },
                                "question_level": {
                                    "type": "string",
                                    "enum": ["easy", "medium","hard"]
                                },
                                "question_type": {
                                    "type": "string",
                                    "enum": ["Remember", "Understand","Apply","Analyze","Evaluate","Create"]
                                }
                            },
                            "required": ["question", "options", "answer","question_level","question_type"]
                        }
                    }
                },
                "required": ["topic", "questions"]
            }
        }
        }
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    #print("response------------",response)
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "generateMCQs": generateMCQs,
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        #print("tool_calls-----------------",tool_calls)
        if(1):
            function_name = tool_calls[0].function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_calls[0].function.arguments)
            function_response = function_to_call(
                questions=function_args.get("questions"),
                topic=function_args.get("topic"),
            )
            return function_response
            
            
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
				mcqs = run_conversation(paragraph)
				#st.write(mcqs)
				mcq_json=json.loads(mcqs)
				for j in mcq_json['questions']:
		  			st.write(j)
		else:
			st.write("Please enter a paragraph to generate questions.")
			
		  
	if uploaded_image is None:         
		paragraph = st.text_area("Enter a paragraph:", height=200)
		#prompt = st.text_area("Enter the prompt:", height=200)
		if st.button("Generate MCQs via text"):
			if paragraph:
				mcqs = run_conversation(paragraph)
				#st.write(type(mcqs))
				mcq_json=json.loads(mcqs)
				for j in mcq_json['questions']:
		  			st.write(j)
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
