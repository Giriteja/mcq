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
import pandas as pd
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



def highlight_max(s):
    is_max = s == s.max()
    return [
        "background-color: lightgreen" if v else "background-color: white"
        for v in is_max
    ]

EVALUATION_PROMPT_TEMPLATE = """
You will be given one summary written for an article. Your task is to rate the summary on one metric.
Please make sure you read and understand these instructions very carefully. 
Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

{criteria}

Evaluation Steps:

{steps}

Example:

Source Text:

{document}

Summary:

{summary}

Evaluation Form (scores ONLY):

- {metric_name}
"""

# Metric 1: Relevance

RELEVANCY_SCORE_CRITERIA = """
Relevance(1-5) - selection of important content from the source. \
The summary should include only important information from the source document. \
Annotators were instructed to penalize summaries which contained redundancies and excess information.
"""

RELEVANCY_SCORE_STEPS = """
1. Read the summary and the source document carefully.
2. Compare the summary to the source document and identify the main points of the article.
3. Assess how well the summary covers the main points of the article, and how much irrelevant or redundant information it contains.
4. Assign a relevance score from 1 to 5.
"""

# Metric 2: Coherence

COHERENCE_SCORE_CRITERIA = """
Coherence(1-5) - the collective quality of all sentences. \
We align this dimension with the DUC quality question of structure and coherence \
whereby "the summary should be well-structured and well-organized. \
The summary should not just be a heap of related information, but should build from sentence to a\
coherent body of information about a topic."
"""

COHERENCE_SCORE_STEPS = """
1. Read the article carefully and identify the main topic and key points.
2. Read the summary and compare it to the article. Check if the summary covers the main topic and key points of the article,
and if it presents them in a clear and logical order.
3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.
"""

# Metric 3: Consistency

CONSISTENCY_SCORE_CRITERIA = """
Consistency(1-5) - the factual alignment between the summary and the summarized source. \
A factually consistent summary contains only statements that are entailed by the source document. \
Annotators were also asked to penalize summaries that contained hallucinated facts.
"""

CONSISTENCY_SCORE_STEPS = """
1. Read the article carefully and identify the main facts and details it presents.
2. Read the summary and compare it to the article. Check if the summary contains any factual errors that are not supported by the article.
3. Assign a score for consistency based on the Evaluation Criteria.
"""

# Metric 4: Fluency

FLUENCY_SCORE_CRITERIA = """
Fluency(1-3): the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.
1: Poor. The summary has many errors that make it hard to understand or sound unnatural.
2: Fair. The summary has some errors that affect the clarity or smoothness of the text, but the main points are still comprehensible.
3: Good. The summary has few or no errors and is easy to read and follow.
"""

FLUENCY_SCORE_STEPS = """
Read the summary and evaluate its fluency based on the given criteria. Assign a fluency score from 1 to 3.
"""

def get_geval_score(
    criteria: str, steps: str, document: str, summary: str, metric_name: str
):
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        criteria=criteria,
        steps=steps,
        metric_name=metric_name,
        document=document,
        summary=summary,
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].message.content


evaluation_metrics = {
    "Relevance": (RELEVANCY_SCORE_CRITERIA, RELEVANCY_SCORE_STEPS),
    "Coherence": (COHERENCE_SCORE_CRITERIA, COHERENCE_SCORE_STEPS),
    "Consistency": (CONSISTENCY_SCORE_CRITERIA, CONSISTENCY_SCORE_STEPS),
    "Fluency": (FLUENCY_SCORE_CRITERIA, FLUENCY_SCORE_STEPS),
}

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
    #st.write(response_json)
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
			summaries = {"Summary 1": summ}

			data = {"Evaluation Type": [], "Summary Type": [], "Score": []}
			
			
			for eval_type, (criteria, steps) in evaluation_metrics.items():
			    for summ_type, summary in summaries.items():
			        data["Evaluation Type"].append(eval_type)
			        data["Summary Type"].append(summ_type)
			        result = get_geval_score(criteria, steps, paragraph, summary, eval_type)
			        score_num = int(result.strip())
			        data["Score"].append(score_num)
			
			pivot_df = pd.DataFrame(data, index=None).pivot(
			    index="Evaluation Type", columns="Summary Type", values="Score"
			)
			st.write(pivot_df)
		else:
			st.write("Please enter the text to generate Summary.")
