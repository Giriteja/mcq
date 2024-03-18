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
import uuid
#import pdfplumber
import PyPDF2


client = OpenAI(
  api_key=os.getenv("openaikey"),  # this is also the default, it can be omitted
)

chatgpt_url = "https://api.openai.com/v1/chat/completions"

chatgpt_headers = {
    "content-type": "application/json",
    "Authorization":"Bearer {}".format(os.getenv("openaikey"))}

tab1, tab2, tab3,tab4,tab5 = st.tabs(["MCQ", "Summary", "Lesson Plan","Assignments","Topic Segregation"])

paragraph="""Food in the form of a soft slimy substance where some
proteins and carbohydrates have already been broken down
is called chyme. Now the food material passes from the
stomach to the small intestine. Here the ring like muscles
called pyloric sphincters relax to open the passage into
the small intestine. The sphincters are responsible for
regulating the opening of the passage such that only small
quantities of the food material may be passed into the
small intestine at a time."""

def generateMCQs(questions,topic):
        return json.dumps({"questions": questions, "topic":topic})

def generate_long_short_questions(questions,topic):
        return json.dumps({"questions": questions, "topic":topic})

def chapter_topic_identification(questions,topic):
        return json.dumps({"questions": questions, "topic":topic})

def save_json_to_text(json_data, filename):
    with open(filename, 'w') as f:
        f.write(json.dumps(json_data, indent=4))

def extract_data(file):
	reader = PyPDF2.PdfReader(file)
	# Extract the content
	text = ""
	num_pages = len(reader.pages)
	for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text()
    	# Display the content
	return text

def topic_segregation(questions,url,headers,prompt):
	# Define the payload for the chat model
    messages = [
        {"role": "system", "content": "Divide the given text into list of questions"+prompt},
        {"role": "user", "content": paragraph}
    ]

    chatgpt_payload = {
        "model": "gpt-3.5-turbo-1106",
        "messages": messages,
        "temperature": 1.3,
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

def generate_assignment(paragraph,url,headers,prompt):
    # Step 1: send the conversation and available functions to the model
    messages = [{"role": "system", "content": """Given the following paragraph, generate Short questions,Long questions that should align with specific cognitive levels according to Bloom's Taxonomy. For each question, use the associated verbs as a guide to ensure the questions match the intended complexity and cognitive process.For each question classify it as Easy,Medium or Hard.
    
Please ensure the questions and options are closely related to the content of the provided text and reflect the cognitive level specified for every question.For short questions, focus on concise inquiries that can be answered in a sentence or two. These questions should aim to test the reader's understanding of key concepts and facts related to the topic.

For long questions, delve deeper into the topic and pose more complex inquiries that may require extended explanations or analysis. These questions should encourage critical thinking and provide opportunities for in-depth exploration of the subject matter.You may use the following example questions as a guide.

Here are examples of short and long questions, along with their answers:

Short Questions:

Question: What is the basic unit of life?

Answer: The cell.
Question: What is photosynthesis?

Answer: Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll.
Question: Define mitosis.

Answer: Mitosis is a type of cell division that results in two daughter cells each having the same number and kind of chromosomes as the parent nucleus.
Question: What is DNA?

Answer: DNA (deoxyribonucleic acid) is a molecule that carries the genetic instructions used in the growth, development, functioning, and reproduction of all known living organisms.
Question: Explain the function of the respiratory system.

Answer: The respiratory system is responsible for the exchange of gases (oxygen and carbon dioxide) between the body and the environment. It involves breathing, gas exchange in the lungs, and the transport of gases via the bloodstream.

Long Questions: Use words like Describe,Explain,Analyze

Question: Describe the process of protein synthesis.

Answer: Protein synthesis is the process by which cells generate new proteins. It involves two main stages: transcription and translation. During transcription, the DNA sequence of a gene is copied into messenger RNA (mRNA). Then, during translation, the mRNA sequence is decoded to assemble a corresponding amino acid sequence to form a protein.
Question: Discuss the structure and function of the human heart.

Answer: The human heart is a muscular organ that pumps blood throughout the body via the circulatory system. It consists of four chambers: two atria and two ventricles. The right side of the heart receives deoxygenated blood from the body and pumps it to the lungs for oxygenation, while the left side receives oxygenated blood from the lungs and pumps it to the rest of the body.
Question: Explain the process of meiosis and its significance.

Answer: Meiosis is a type of cell division that produces gametes (sperm and egg cells) with half the number of chromosomes as the parent cell. It involves two rounds of division (meiosis I and meiosis II) and results in four haploid daughter cells. Meiosis is significant because it generates genetic diversity through the shuffling and recombination of genetic material, which is essential for sexual reproduction and evolutionary adaptation.
Question: Discuss the role of enzymes in biological reactions.

Answer: Enzymes are biological catalysts that speed up chemical reactions in living organisms without being consumed in the process. They lower the activation energy required for reactions to occur, thereby increasing the rate of reaction. Enzymes are specific to particular substrates and often undergo conformational changes to facilitate substrate binding and catalysis.
Question: Explain the process of photosynthesis in detail.

Answer: Photosynthesis is a complex biochemical process by which green plants, algae, and some bacteria convert light energy, carbon dioxide, and water into glucose and oxygen. It occurs in chloroplasts and involves two main stages: the light-dependent reactions and the Calvin cycle. During the light-dependent reactions, light energy is used to split water molecules, producing oxygen and ATP. The Calvin cycle then uses ATP and NADPH generated in the light-dependent reactions to fix carbon dioxide and synthesize glucose."""},{"role": "user", "content": paragraph}]
    tools = [
	{
            "type": "function",
            "function": {
            "name": "generate_long_short_questions",
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
                                },
				"question_type_short_or_long": {
                                "type": "string",
                                "enum": ["Short Question", "Long Question"]
                                }
				    
                            },
                            "required": ["question", "answer","question_level","question_type","question_type_short_or_long"]
                        }
                    }
                },
                "required": ["topic", "questions"]
            }
        }
        }
	    
	    
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
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
	    "generate_long_short_questions":generate_long_short_questions
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        #print("tool_calls-----------------",tool_calls)
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                questions=function_args.get("questions"),
                topic=function_args.get("topic"),
            )
            return function_response

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
        model="gpt-3.5-turbo-1106",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
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

def generate_lessonplan(topic,url,headers,prompt):
    
    # Define the payload for the chat model
    messages = [
        {"role": "system", "content": """Generate a detailed lesson plan for a 45-minute high school class on the given topic . The lesson plan should include:

1. Learning Objectives: Clearly defined goals that students should achieve by the end of the lesson.
2. Introduction: A brief overview to engage students and introduce the topic.
3. Main Activity: A step-by-step guide for the main instructional activity, including any discussions, demonstrations, or hands-on activities.
4. Materials Needed: A list of all materials and resources required for the lesson.
5. Assessment: Methods for evaluating student understanding, such as questions to ask or short exercises.
6. Conclusion: A summary to reinforce key concepts and connect to future lessons.
7  Real world Exmaples : Include real world examples based on the topic if applicable.
8. Additional Resources: Optional extra online materials like youtube videos,weblinks for further exploration of the topic.

Ensure the lesson plan is structured, engaging, and suitable for high school students with a basic understanding of biology.
"""+prompt},
        {"role": "user", "content": topic}
    ]

    chatgpt_payload = {
        "model": "gpt-4-1106-preview",
        "messages": messages,
        "temperature": 1.3,
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
        {"role": "system", "content": "Summarize content you are provided with headings and bullet points.Also consider the following Instruction while generating Summary"+prompt},
        {"role": "user", "content": paragraph}
    ]

    chatgpt_payload = {
        "model": "gpt-3.5-turbo-1106",
        "messages": messages,
        "temperature": 1.3,
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
    messages = [{"role": "system", "content": """Generate multiple-choice questions (MCQs) on the given paragraph. Provide questions at different cognitive levels according to Bloom's Taxonomy. Include a variety of question types and encourage creativity in the question generation process. You may use the following example questions as a guide.For each question classify it as Easy,Medium or Hard.
    
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


   
Ensure the questions and options are closely related to the content of the provided text and reflect the cognitive level specified for every question. Generate as many questions as possible from the given content, incorporating diverse question types and encouraging creativity in the process."""},{"role": "user", "content": paragraph}]
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
        model="gpt-4-1106-preview",
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
	uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")
	final_data=[]
	#option = st.selectbox(
    	#'Choose Number of Questions:',
    	#('5', '10', '15', '20'))
	# If an image is uploaded, display and process it

	syllabus  = st.selectbox(
	   			"Select Subject",
		("CBSE", "SSC"),key="syllabus")
	class_name = st.selectbox(
	   			"Select class",
		("10", "9","8","7","6"),key="class")
	subject_name  = st.selectbox(
	   			"Select Subject",
		("Physics", "Social","Biology","Chemistry"),key="subject")
	lesson_name  = st.selectbox(
	   			"Select lesson",
		("1", "2","3","4","5","6","7","8","9","10"),key="lesson_name")
	#paragraph = st.text_area("Enter a paragraph:",text, height=200)
	
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
				mcq_json=json.loads(mcqs)
				for j in mcq_json['questions']:
					json_struct={}
					json_struct['class']=class_name
					json_struct['subject']=subject_name
					json_struct['lesson']=subject_name
					json_struct['question']=j['question']
					json_struct['options']=j['options']
					json_struct['answer']=j['answer']
					json_struct['level']=j['question_level']
					json_struct['question_type']=j['question_type']
					json_struct['type']='multi-choice'
					json_struct['lesson']=lesson_name
					json_struct['syllabus']=syllabus
					#st.write(json_struct)
					final_data.append(json_struct)
				#st.write(final_data)
				save_json_to_text(final_data, 'output.txt')
				download_button_id = str(uuid.uuid4())
				# Provide a download link for the text file
				st.download_button(
				        label="Download Text File",
				        data=open('output.txt', 'rb').read(),
				        file_name='output.txt',
				        mime='text/plain',
					key=download_button_id
				)
				
				

	
		else:
			st.write("Please enter a paragraph to generate questions.")
			
		  
	if not(uploaded_image and uploaded_pdf):         
		paragraph = st.text_area("Enter a paragraph:", height=200)
		#prompt = st.text_area("Enter the prompt:", height=200)
		
		
		if st.button("Generate MCQs via text"):
			if paragraph:
				mcqs = run_conversation(paragraph)
				mcq_json=json.loads(mcqs)
				for j in mcq_json['questions']:
					json_struct={}
					json_struct['class']=class_name
					json_struct['subject']=subject_name
					json_struct['lesson']=lesson_name
					json_struct['question']=j['question']
					json_struct['options']=j['options']
					json_struct['answer']=j['answer']
					json_struct['level']=j['question_level']
					json_struct['question_type']=j['question_type']
					json_struct['type']='multi-choice'
					json_struct['lesson']=lesson_name
					json_struct['syllabus']=syllabus
					#st.write(json_struct)
					final_data.append(json_struct)
				save_json_to_text(final_data, 'output.txt')
				download_button_id = str(uuid.uuid4())
				# Provide a download link for the text file
				st.download_button(
				        label="Download Text File",
				        data=open('output.txt', 'rb').read(),
				        file_name='output.txt',
				        mime='text/plain',
					key=download_button_id
				)
			else:
				st.write("Please enter a paragraph to generate questions.")

	if uploaded_pdf is not None:         
		pdf_file_path = uploaded_pdf  # Replace "example.pdf" with the path to your PDF file
		extracted_text = extract_data(pdf_file_path)
		paragraph = st.text_area("Enter a paragraph:",extracted_text, height=200)
		
		
		if st.button("Generate MCQs via text",key="123"):
			if paragraph:
				mcqs = run_conversation(paragraph)
				mcq_json=json.loads(mcqs)
				for j in mcq_json['questions']:
					json_struct={}
					json_struct['class']=class_name
					json_struct['subject']=subject_name
					json_struct['lesson']=lesson_name
					json_struct['question']=j['question']
					json_struct['options']=j['options']
					json_struct['answer']=j['answer']
					json_struct['level']=j['question_level']
					json_struct['question_type']=j['question_type']
					json_struct['type']='multi-choice'
					json_struct['lesson']=lesson_name
					json_struct['syllabus']=syllabus
					#st.write(json_struct)
					final_data.append(json_struct)
				save_json_to_text(final_data, 'output.txt')
				download_button_id = str(uuid.uuid4())
				# Provide a download link for the text file
				st.download_button(
				        label="Download Text File",
				        data=open('output.txt', 'rb').read(),
				        file_name='output.txt',
				        mime='text/plain',
					key=download_button_id
				)
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
			        score_num = float(result.strip())
			        data["Score"].append(score_num)
			
			pivot_df = pd.DataFrame(data, index=None).pivot(
			    index="Evaluation Type", columns="Summary Type", values="Score"
			)
			st.write(pivot_df)
		else:
			st.write("Please enter the text to generate Summary.")
with(tab3):
	topic = st.text_area("Enter the topic for lesson plan:", height=200)
	prompt_topic = st.text_area("Enter the prompt:",key="topic", height=200)
	if st.button("Generate Lesson Plan"):
		if topic:
			lp = generate_lessonplan(topic,chatgpt_url,chatgpt_headers,prompt_topic)
			st.write(lp)
			
		else:
			st.write("Please enter the text to generate Summary.")

with(tab4):
	
	topic_assign = st.text_area("Enter the topic for Assignment:", height=200)
	prompt_topic_assign = st.text_area("Enter the prompt:",key="topic_assign", height=200)
	json_struct={}
	final_data=[]
	if st.button("Generate Assignment"):
		if topic_assign:
			lp = generate_assignment(topic_assign,chatgpt_url,chatgpt_headers,prompt_topic_assign)
			lp_json=json.loads(lp)
			for j in lp_json['questions']:
					json_struct={}
					json_struct['class']=class_name
					json_struct['subject']=subject_name
					json_struct['lesson']=subject_name
					json_struct['options']=[]
					json_struct['question']=j['question']
					json_struct['answer']=j['answer']
					json_struct['level']=j['question_level']
					json_struct['question_type']=j['question_type']
					json_struct['type']=j['question_type_short_or_long']
					json_struct['lesson']=lesson_name
					json_struct['syllabus']=syllabus
					#st.write(json_struct)
					final_data.append(json_struct)
					#st.write(final_data)
			save_json_to_text(final_data, 'output.txt')
			download_button_id = str(uuid.uuid4())
			# Provide a download link for the text file
			st.download_button(
				        label="Download Text File12",
				        data=open('output.txt', 'rb').read(),
				        file_name='output.txt',
				        mime='text/plain',
					key=download_button_id
			)
			
		else:
			st.write("Please enter the text to generate Summary.")

with(tab5):
	
	prev_questions = st.text_area("Enter the topic for Assignment:", height=200,key="prev")
	final_data=[]
	if st.button("Generate topic questions"):
		if prev_questions:
			lp = topic_segregation(topic_assign,chatgpt_url,chatgpt_headers,prompt_topic_assign)
			lp_json=json.loads(lp)
			st.write(lp_json)
			
		else:
			st.write("Please enter the text to generate Summary.")
