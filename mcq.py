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
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


data_cred={"type": "service_account",
		"project_id": os.getenv("project_id"),
		"private_key_id": os.getenv("private_key_id"),
		"private_key": os.getenv("private_key").replace('\\n', '\n'),
		"client_email": os.getenv("client_email"),
		"client_id": os.getenv("client_id"),
		"auth_uri": "https://accounts.google.com/o/oauth2/auth",
		"token_uri": "https://oauth2.googleapis.com/token",
		"auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
		"client_x509_cert_url": os.getenv("client_x509_cert_url"),
	"universe_domain":"googleapis.com"}



if not firebase_admin._apps:
	cred = credentials.Certificate(data_cred) 
	app = firebase_admin.initialize_app(cred)
 
db = firestore.client()

client = OpenAI(
  api_key=os.getenv("openaikey"),  # this is also the default, it can be omitted
)

chatgpt_url = "https://api.openai.com/v1/chat/completions"

chatgpt_headers = {
	"content-type": "application/json",
	"Authorization":"Bearer {}".format(os.getenv("openaikey"))}

tab1, tab2, tab3,tab4,tab5,tab6,tab7 = st.tabs(["MCQ", "Summary", "Lesson Plan","Assignments","Topic Segregation","Brain Busters","Textbook Questions"])

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
	# Step 1: send the conversation and available functions to the model

	messages = [{"role": "system", "content": """Please format the provided set of questions and assign each question to the relevant chapter and subtopic from the given syllabus :
	
BIOLOGY - SYLLABUS
10th CLASS
1. Nutrition
1.1 Life process- Introduction
1.1.1 Autotrophic and heterotrophic nutrition
1.2 Photosynthesis
1.2.1 Understand the concept of photosynthesis
1.2.2 Raw materials required for photosynthesis - H2O, CO2
sunlight
1.2.3 Process of releasing oxygen in photosynthesis
1.2.4 Necessity of light for formation of carbohydrate
1.2.5 Chlorophyll - Photosynthesis
1.2.6 Where does photosynthesis takes place
1.2.7 Mechanism of photosynthesis :
(i) Light reaction, (ii) Dark reaction
1.3 Nutrition in organisms
1.3.1 How do the organisms obtain the food?
1.3.2 Cuctuta - Parasitic nutrition
1.4 Digestion in human beings
O Process of movement of food through alimentary canal
O Litmus paper test OEnzyme OFlow chart of Human
 digestive system
1.5 Healthy points about oesophagus
1.6 Malnutrition -disease OKwashiorkore OMarasmus OObesity
1.6.1 Diseases due to vitamin deficiency
2. Respiration
2.1 Respiration - discovery of gases involved in respiration
2.1.1 Different stages of respiration
2.1.2 Expiration, inspiration
2.1.3 Pathway of air
2.1.4 Epiglottis - Pathway of air.
2.2 Respirating system in human being
2.2.1 Exchange of gases (alveolies to Blood capillaries)
2.2.2 Mechanism of transport of gases
2.2.3 Transport of gases (Capillaries to cells, cells to back)
2.3 Cellular respiration
2.3.1 Anaerobic respiration
2.3.2 Aerobic respiration
2.3.3 Fermentation
2.4 Respiration - Combustion
O Liberating heat during respiration
2.5 Evolution of gaseous exchange
2.6 Plant respiration
2.6.1 Transportation of gases in plants
2.6.2 Respiration through roots
2.6.3 Photosynthesis - respiration
3. Transportation
3.1 Internal structure of Heart
3.1.1 Blood vessels and blood transport
OBlood capillaries OArteries veins
3.2 Cardiac cycle
3.2.1 Single circulation, double circulation
3.3 Lymphatic system
3.4 Evolution of transport system
3.5 Blood pressure
3.6 Blood clotting
3.7 Trasnportation in plants
3.7.1 How water is absorbed
3.7.2 Root hair absorbtion
3.7.3 What is root pressure?
3.7.4 Mechanism of transportation of water in plants -
Transportation, Root pressure, ascent of sap. Cohesive
adhesive pressure
3.7.5 Transportation of Minerals
3.7.6 Transportation of food material
4. Excretion
4.1 Excretion in Human beings
4.2 Excretory system
4.2.1 Kidney
4.2.2 Kidney internal structure
4.3 Structure of Nephron
O Malphigion tubules ONephron
4.4 Formation of urine
• Glomerular filtration
• Tubular reabsorption
• Tubular secretion
• Formation of hypertonic urine
4.4.1 Ureter
4.4.2 Urinary bladder
4.4.3 Urethra
4.4.4 Urine excretion
4.4.5 Urine composition
4.5 Dialysis - Artificial kidney
4.5.1 Kidney transportation
4.6 Accessory Excretery organs in human beeing (Lungs, skin,
liver large intestine)
4.7 Excretion in other organisms
4.8 Excretion in plants
4.8.1 Alkaloids
4.8.2 Tannin
4.8.3 Resin
4.8.4 Gums
4.8.5 Latex
4.9 Excretion, Secretion
5. Control & coordination
5.1 Stimulus and response
5.2 Integrated system - Nerves coordination
5.3 Nerve cell structure
5.4 Pathways from stimulus to response
5.4.1 Afferent nerves
5.4.2 Efferent nerves
5.5 Reflex arc
5.5.1 Reflex arc
5.6 Central nervous system
OBrain OSpinal nerves
5.7 Peripherial nervous system
5.8 Coordination without nerves
5.8.1 Story of insulin
5.8.2 Chemical coordination - endocrine glands
5.8.3 Feedback mechanism
5.9 Autonomous nervous system
5.10 Coordination in plants - Phytohormones
5.10.1 How plant shows responses to stimulus
5.10.2 Tropic movements in plants
6. Reproduction
6.1 Growth of bacteria in milk.
6.2 Asexual reproduction
6.2.1 fission, budding, fragmentation, parthenocarpy,
parthenogensis, regeneration
6.2.2 Vegetative propagation
ONatural vegetative propagation through roots, stem,
 leaves
OArtificial propagation - cuttings, layering and
 grafting
6.2.3 Formation of spores Sporophyll
6.3 Sexual reproduction
Reproduction in human beings
6.3.1 Male reproductive system
6.3.2 Female reproductive system
6.3.3 Child birth
6.4 Sexual reproduction in plants
6.4.1 Flower - reproductive parts, unisexual, bisexual flowers,
self and cross pollination.
6.4.2 Pollen grain
6.4.3 Structure of ovule, ovary; double fertilisation
6.4.4 Germination of seeds
6.5 Cell division - Cell cycle
6.5.1 Cell division in humn beings
6.5.2 Cell cycle - G1
, S, G2 and M phases
6.5.3 Mitosis
6.5.4 Meiosis
6.6 Reproductive health - HIV/ AIDS
6.6.1 Birth control methods
6.6.2 Fighting against social ills
6.6.3 Teenage motherhood, stop female foeticide
7. Coordination in Life Processes
7.1 Hunger
7.1.1 Effect of hunger stimulus
7.2 Relation between taste and smell
7.2.1 Relation between taste of tongue and palate
7.3 Mouth - a mastication machine
7.3.1 Action of Saliva on flour
7.3.2 Observing the pH of mouth
7.4 Passage of food through oesophagus
7.4.1 Peristaltic movement in oespaphagus
7.5 Stomach is mixer
7.5.1 Movement of food from stomach to intestion.
7.5.2 Excretion of waste material
8. Heredity
8.1 New Characters - variation
8.2 Experiments conducted by Mendal (F1 generation, F2 generation), Mendel's Laws
8.2.1 F1 generation self pollination
8.2.2 Phenotype
8.2.3 Genotype
8.3 Parents to offsprings
8.31 How the characters exhibit?
8.3.2 Sex determination in human beings
8.4 Evolution
8.4.1 Genetic drift
8.5 Theories of organic evolution
8.5.1 Lamarckism
8.5.2 Darwinism
8.5.3 Darwin theory in a nut shell
8.6 Origin of species
8.6.1 How the new species orginates
8.7 Evolution - Evidences
8.7.1 Homologous organs - analogous organs
8.7.2 Embrylogical Evidence
8.7.3 Fossils Evidences
8.8 Human Evolution
8.8.1 Human Beings: Museum of vestigial organs
9. Our Environment
9.1 Ecosystem - Food chain
9.1.1 Number Pyramid
9.1.2 Biomass Pyramid
9.1.3 Energy pyramid
9.2 Human activities - Their effect on ecosystem
9.2.1 Story of Kolleru lake
9.2.2 Edulabad resorvoir - Effect of heavy metals
9.2.3 Sparrow campaign
9.3 Biological pest control measures
O Crop rotation
O Knowing the history of pests
O Sterility
O Gene mutation
O Concern towards environment
10. Natural resources
10.1 Case study - Agricultural land (past and present)
10.2 Case study - Water management
O Community based particing
O Farmer based intervention
O Waste land cultivation
10.3 Water resources in the Telugu States
10.4 Natural resources around us
10.5 Forest Renewable resources
10.5.1 Soil
10.5.2 Bio-diversity
10.6 Fossil fuels
10.6.1 Minerals
10.7Conservation, Redue, Reuse, Recycle, Recover
10.7.1 Conservation groups

Provide the output in following json format:

questions=[{Question:"Write the functions of xylem and phloem?",
Chapter Name:"Transportation ",
Subtopic Name:"Trasnportation in plants",
question_type:"very short",
Marks:"2 marks"},
{Question:"Write the differences between mitosis and meiosis ?",
Chapter Name:"Reproduction ",
Subtopic Name:"Cell division - Cell cycle",
question_type:"short",
Marks:"4 marks"},
{Question:"What is Malnutrition ? Explain few nutritional deficiency diseases ?",
Chapter Name:"Nutrition",
Subtopic Name:"Malnutrition -disease",
question_type:"long",
Marks:"8 marks"}
{Question:"The energy currency of the cells is?",
options:[A)GTP B) ADP C) ATP D)GDP]
Chapter Name:"Respiration",
Subtopic Name:"Cellular respiration",
question_type:"mcq",
Marks:"8 marks"}

While giving Marks and question type consider the section in which that question was mentioned and assign.Marks should be given based on following values[2,4,8,1].
]

"""},{"role": "user", "content": questions}]

	chatgpt_payload = {
	"response_format": {"type": "json_object"},
		"model": "gpt-4-1106-preview",
		"messages": messages,
		"temperature": 1.3,
		"top_p": 1,
	"max_tokens":4096,
	}

	# Make the request to OpenAI's API
	response = requests.post(url, json=chatgpt_payload, headers=headers)
	response_json = response.json()

	# Extract data from the API's response
	st.write(response_json)
	output = response_json['choices'][0]['message']['content']
	return output

def generate_assignment(paragraph,url,headers,prompt):
	# Step 1: send the conversation and available functions to the model
	messages = [{"role": "system", "content": f"""Given the following paragraph and the following instructions{prompt}, generate Short questions,Long questions,assertion and reason based questions that should align with specific cognitive levels according to Bloom's Taxonomy. For each question, use the associated verbs as a guide to ensure the questions match the intended complexity and cognitive process.For each question classify it as Easy,Medium or Hard.
	
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
		tool_choice="required",	 # auto is default, but we'll be explicit
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
def generate_answers_for_textbookquestions(paragraph,url,headers,prompt):
	# Step 1: send the conversation and available functions to the model
	messages = [{"role": "system", "content": f""" 
You will be provided with a list of textbook questions in triple quotes. For each question, you need to generate an appropriate answer and classify the question as Easy, Medium, or Hard.


Think step by step and follow instructions: 

step 1: Extract each question without question numbers: Separate each question from the provided list(don't need question number).If the options are not there for a question then provide ' ' (empty space) for options.
step 2: Workout answers to all the questions(dont leave any question): Take the time to think and based on the type of question (short answer, multiple-choice question, long answer), workout an accurate and concise answer. For MCQ questions you have to provide answer from among the options of the question.
step 3: Classify the question: Determine the difficulty level of each question as Easy, Medium, or Hard.

 
Ensure that your answers are accurate and appropriate for the type of question asked, if given for the question.

Here's an example to guide you:

MCQ's
Example 1 type:
	Question: <given mcq question>
	a) option A
	b) option B
	c) option C
	d) option D

	options:
	['(a) and (b)'
	'(a) and (c)'
	'(a), (b) and (c)'
	'all']

	Answer:

	i) (a) and (b)
				
	Difficulty Level: Medium
	question_type_mcq_or_short_or_long: MCQ
			  
example 2:
	If the input question is like this: Two conducting wires of the same material and of equal lengths and equal diameters\nare first connected in series and then parallel in a circuit across the same potential difference. The ratio of heat produced in series and parallel combinations would be (a) 1:2 (b) 2:1 (c) 1:4 (d) 4:1
	then output should be
	question: Two conducting wires of the same material and of equal lengths and equal diameters\nare first connected in series and then parallel in a circuit across the same potential difference. The ratio of heat produced in series and parallel combinations would be
	options:[
	 '1:2',
	 '2:1',
	 '1:4', 
	 '4:1']
	
	answer: 1:2

	Difficuly Level: easy

	question_type_mcq_or_short_or_long: MCQ

example 3:
	If the input question is like this: 2. Which of the following terms does not represent electrical power in a circuit? (a) I**2R (b) IR**2 (c) VI (d) V**2/R
	then output should be
	question: Which of the following terms does not represent electrical power in a circuit?
	options:[
	I**2R 
	IR**2 
	VI 
	V**2/R]
	
	answer: IR**2

	Difficuly Level: easy

	question_type_mcq_or_short_or_long: MCQ
			  
short answer questions

example:
'''
question:
Translate the following statements into chemical equations and then balance them.
(a) Hydrogen gas combines with nitrogen to form ammonia.
(b) Hydrogen sulphide gas burns in air to give water and sulpur dioxide.
(c) Barium chloride reacts with aluminium sulphate to give aluminium chloride
and a precipitate of barium sulphate.
(d) Potassium metal reacts with water to give potassium hydroxide and hydrogen
gas.

answer: 
(a) Hydrogen gas combines with nitrogen to form ammonia.
3H2 + N2 → 2NH3
(b) Hydrogen sulphide gas burns in air to give water and sulfur dioxide.
2H2S + 3O2 → 2H2O + 2SO2
(c) Barium chloride reacts with aluminium sulphate to give aluminium chloride and a precipitate of barium sulphate.
3BaCl2 + Al2(SO4)3 → 2AlCl3 + 3BaSO4
(d) Potassium metal reacts with water to give potassium hydroxide and hydrogen gas.
2K + 2H2O → 2KOH + H2

options: []
			  
Difficuly Level: medium
			  
question_type_mcq_or_short_or_long: Short Question
'''

long answer questions examples: 


Question: Describe the process of protein synthesis.

Answer: Protein synthesis is the process by which cells generate new proteins. It involves two main stages: transcription and translation. During transcription, the DNA sequence of a gene is copied into messenger RNA (mRNA). Then, during translation, the mRNA sequence is decoded to assemble a corresponding amino acid sequence to form a protein.
options: []
Difficuly Level: hard
			  
question_type_mcq_or_short_or_long: Long Question
			  	  
Question: Discuss the structure and function of the human heart.

Answer: The human heart is a muscular organ that pumps blood throughout the body via the circulatory system. It consists of four chambers: two atria and two ventricles. The right side of the heart receives deoxygenated blood from the body and pumps it to the lungs for oxygenation, while the left side receives oxygenated blood from the lungs and pumps it to the rest of the body.

options = []
			  
Difficuly Level: medium
			  
question_type_mcq_or_short_or_long: Long Question
			  
Important notes:
			  You must give options key for every question. If question don't have options then value for this key will be empty space else options given.
			  
"""},{"role": "user", "content": f"{prompt}:{paragraph}"}]
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
								"options": {
									"type": "array",
									"items": {
										"type": "string",
										"description": "You must give options key for every question."
									}
								},
								"question_type": {
									"type": "string",
									"enum": ["Remember", "Understand","Apply","Analyze","Evaluate","Create"]
								},
				"question_type_mcq_or_short_or_long": {
								"type": "string",
								"enum": ["MCQ", "Short Question", "Long Question"]
								}
					
							},
							"required": ["question", "answer","question_level","question_type","question_type_mcq_or_short_or_long","options"]
						}
					}
				},
				"required": ["topic", "questions"]
			}
		}
		}
		
		
	]
	response = client.chat.completions.create(
		model="gpt-4",
		messages=messages,
		tools=tools,
		tool_choice="required",
		seed=100	 # auto is default, but we'll be explicit
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
	opencv_image = np.array(image)	# Convert PIL image to OpenCV format
	gray_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

	# Apply Tesseract OCR
	extracted_text = pytesseract.image_to_string(gray_image)

	return extracted_text

def run_conversation(paragraph,prompt):
	# Step 1: send the conversation and available functions to the model
	messages = [{"role": "system", "content": f"""Generate multiple-choice questions (MCQs) on the given paragraph and consider the following instructions {prompt}. Provide questions at different cognitive levels according to Bloom's Taxonomy. Include a variety of question types and encourage creativity in the question generation process. You may use the following example questions as a guide.For each question classify it as Easy,Medium or Hard.
	
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
	 
	Difficulty	Level:Hard


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
		model="gpt-4",
		messages=messages,
		tools=tools,
		tool_choice="required",	 # auto is default, but we'll be explicit
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
				"Select Syllabus",
		("CBSE", "SSC"),key="syllabus")
	class_name = st.selectbox(
				"Select class",
		('VI', 'VII', 'VIII', 'IX','X'),key="class")
	subject_name  = st.selectbox(
				"Select Subject",
		("PHYSICS","SCIENCE","BIOLOGY","CHEMISTRY", "SOCIAL", "HISTORY", "GEOGRAPHY", "CIVICS", "ECONOMICS", "MATHEMATICS", "TELUGU", "HINDI", "ENGLISH"),key="subject")
	lesson_name	 = st.selectbox(
				"Select lesson",
		("LESSON1", "LESSON2","LESSON3","LESSON4","LESSON5","LESSON6","LESSON7","LESSON8","LESSON9","LESSON10","LESSON11","LESSON12","LESSON13"),key="lesson_name")
	#paragraph = st.text_area("Enter a paragraph:",text, height=200)

	
	
	if uploaded_image is not None:
		# Display the uploaded image
		image = Image.open(uploaded_image)
		st.image(image, caption="Uploaded Image", use_column_width=True)

		# Extract text
		text = extract_text(image)
		paragraph = st.text_area("Enter a paragraph:",text, height=200)
		prompt_mcq = st.text_area("Enter a paragraph:",text, height=200)
		#prompt = st.text_area("Enter the prompt:", height=200)
		if st.button("Generate MCQs"):
				if paragraph:
			
					if syllabus == "CBSE":
						subject_collection = db.collection('cbse_subjects')
					elif syllabus == "SSC":
						subject_collection = db.collection('ssc_subjects')
					else:
						raise Exception("Wrong Syllabus")
					 
					subject_data = subject_collection.where("subject_name", "==", subject_name).limit(1).get()[0].to_dict()
					subject_id = subject_data['subject_id']
					 
					lesson_collection = db.collection('lessons')
					lesson_document = lesson_collection.where("lesson_name", "==", lesson_name).where("subject_id", "==", subject_id).where("class", "==", class_name).limit(1)
					lesson_id = lesson_document.get()[0].id
					mcqs = run_conversation(paragraph,prompt_mcq)
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
						json_struct['subject_id']=subject_id
						json_struct['lesson_id']=lesson_id
						json_struct['access']="public"
						json_struct['marks']='1'
						json_struct['metadata']={"tags":[class_name,lesson_name,subject_name,j['question_type']]}
						#st.write(json_struct)
						final_data.append(json_struct)
					#st.write(final_data)
					save_json_to_text(final_data, 'output.txt')
					collection = db.collection("question-library")
					for item in final_data:
						doc = collection.document()
						item['question_id'] = doc.id
						doc.set(item)
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
			prompt_mcq = st.text_area("Enter the prompt:", height=200)
		
		
			if st.button("Generate MCQs via text"):
				if syllabus == "CBSE":
					subject_collection = db.collection('cbse_subjects')
				elif syllabus == "SSC":
					subject_collection = db.collection('ssc_subjects')
				else:
					raise Exception("Wrong Syllabus")
				 
				subject_data = subject_collection.where("subject_name", "==", subject_name).limit(1).get()[0].to_dict()
				subject_id = subject_data['subject_id']
				 
				lesson_collection = db.collection('lessons')
				lesson_document = lesson_collection.where("lesson_name", "==", lesson_name).where("subject_id", "==", subject_id).where("class", "==", class_name).limit(1)
				lesson_id = lesson_document.get()[0].id
				if paragraph:
					mcqs = run_conversation(paragraph,prompt_mcq)
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
						json_struct['syllabus']=syllabus
						json_struct['subject_id']=subject_id
						json_struct['lesson_id']=lesson_id
						json_struct['access']="public"
						json_struct['marks']='1'
						json_struct['metadata']={"tags":[class_name,lesson_name,subject_name,j['question_type']]}
						#st.write(json_struct)
						final_data.append(json_struct)
					save_json_to_text(final_data, 'output.txt')
					collection = db.collection("question-library")
					for item in final_data:
						doc = collection.document()
						item['question_id'] = doc.id
						doc.set(item)
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
		prompt_mcq = st.text_area("Enter the prompt:", height=200)
		
		if st.button("Generate MCQs via text",key="123"):
			if paragraph:
				if st.button("Generate MCQs via text"):
					if syllabus == "CBSE":
						subject_collection = db.collection('cbse_subjects')
					elif syllabus == "SSC":
						subject_collection = db.collection('ssc_subjects')
					else:
						raise Exception("Wrong Syllabus")
					 
					subject_data = subject_collection.where("subject_name", "==", subject_name).limit(1).get()[0].to_dict()
					subject_id = subject_data['subject_id']
					 
					lesson_collection = db.collection('lessons')
					lesson_document = lesson_collection.where("lesson_name", "==", lesson_name).where("subject_id", "==", subject_id).where("class", "==", class_name).limit(1)
					lesson_id = lesson_document.get()[0].id
					mcqs = run_conversation(paragraph,prompt_mcq)
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
						json_struct['syllabus']=syllabus
						json_struct['subject_id']=subject_id
						json_struct['lesson_id']=lesson_id
						json_struct['access']="public"
						json_struct['marks']='1'
						json_struct['metadata']={"tags":[class_name,lesson_name,subject_name,j['question_type']]}
						#st.write(json_struct)
						final_data.append(json_struct)
					save_json_to_text(final_data, 'output.txt')
					collection = db.collection("question-library")
					for item in final_data:
						doc = collection.document()
						item['question_id'] = doc.id
						doc.set(item)
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
	# Upload image
	final_data=[]
	#option = st.selectbox(
		#'Choose Number of Questions:',
		#('5', '10', '15', '20'))
	# If an image is uploaded, display and process it

	syllabus  = st.selectbox(
				"Select Syllabus",
		("CBSE", "SSC"),key="syllabus1")
	class_name = st.selectbox(
				"Select class",
		('VI', 'VII', 'VIII', 'IX','X'),key="class1")
	subject_name  = st.selectbox(
				"Select Subject",
		("PHYSICS","SCIENCE","BIOLOGY","CHEMISTRY", "SOCIAL", "HISTORY", "GEOGRAPHY", "CIVICS", "ECONOMICS", "MATHEMATICS", "TELUGU", "HINDI", "ENGLISH"),key="subject1")
	lesson_name	 = st.selectbox(
				"Select lesson",
		("LESSON1", "LESSON2","LESSON3","LESSON4","LESSON5","LESSON6","LESSON7","LESSON8","LESSON9","LESSON10","LESSON11","LESSON12","LESSON13"),key="lesson_name1")
	#paragraph = st.text_area("Enter a paragraph:",text, height=200)

	
	topic_assign = st.text_area("Enter the topic for Assignment:", height=200)
	prompt_topic_assign = st.text_area("Enter the prompt:",key="topic_assign", height=200)
	json_struct={}
	
	if st.button("Generate Assignment"):
		if topic_assign:
			if syllabus == "CBSE":
				subject_collection = db.collection('cbse_subjects')
			elif syllabus == "SSC":
				subject_collection = db.collection('ssc_subjects')
			else:
				raise Exception("Wrong Syllabus")
			 
			subject_data = subject_collection.where("subject_name", "==", subject_name).limit(1).get()[0].to_dict()
			subject_id = subject_data['subject_id']
			 
			lesson_collection = db.collection('lessons')
			lesson_document = lesson_collection.where("lesson_name", "==", lesson_name).where("subject_id", "==", subject_id).where("class", "==", class_name).limit(1)
			lesson_id = lesson_document.get()[0].id
			lp = generate_assignment(topic_assign,chatgpt_url,chatgpt_headers,prompt_topic_assign)
			lp_json=json.loads(lp)
			for j in lp_json['questions']:
					json_struct={}
					json_struct['class']=class_name
					json_struct['subject']=subject_name
					json_struct['lesson']=lesson_name
					json_struct['options']=[]
					json_struct['question']=j['question']
					json_struct['answer']=j['answer']
					json_struct['level']=j['question_level']
					json_struct['question_type']=j['question_type']
					json_struct['type']='single-line'
					if(j['question_type_short_or_long']=='short'):
						json_struct['marks']='2'
					else:
						json_struct['marks']='6'
					json_struct['syllabus']=syllabus
					json_struct['subject_id']=subject_id
					json_struct['lesson_id']=lesson_id
					json_struct['access']="public"
					json_struct['metadata']={"tags":[class_name,lesson_name,subject_name,j['question_type_short_or_long']]}
					#st.write(json_struct)
					final_data.append(json_struct)
					#st.write(final_data)
			save_json_to_text(final_data, 'output.txt')
			collection = db.collection("question-library")
			for item in final_data:
				doc = collection.document()
				item['question_id'] = doc.id
				doc.set(item)
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
			lp = topic_segregation(prev_questions,chatgpt_url,chatgpt_headers,prompt_topic_assign)
			#lp_json=json.loads(lp)
			st.write(lp)
			
		else:
			st.write("Please enter the text to generate Summary.")

with(tab6):

	
	st.title("Syllabus Explorer")
 
	
	
 
	# Create a dropdown for syllabus
	syllabus_options = [doc.id for doc in db.collection("syllabus-db").stream()]
	syllabus_option_ids = [doc.id for doc in db.collection("syllabus-db").stream()]
	syllabus_options = []
	for item in syllabus_option_ids:
		syllabus_options.append(db.collection("syllabus-db").document(item).get().to_dict()['syllabus'])
	syllabus_brain = st.selectbox("Select Syllabus", syllabus_options)
   
 
	# Create a dropdown for class
	if syllabus_brain:
		classes_option_ids = [doc.id for doc in db.collection("classes").stream()]
		classes_options = []
		for item in classes_option_ids:
			classs = db.collection("classes").document(item).get().to_dict()['display_name']
			classes_options.append(classs)
		class_brain = st.selectbox("Select Class", classes_options)
		
 
	# Create a dropdown for subject
	if class_brain:
		subject_option_ids = [doc.id for doc in db.collection("subjects").where("class.display_name", "==", class_brain).where("syllabus.syllabus", "==", syllabus).stream()]
		subjects_options = []
		subjects_id_mapping = {}
		for item in subject_option_ids:
			subject = db.collection("subjects").document(item).get().to_dict()['subject']
			subjects_id_mapping[subject] = item
			subjects_options.append(subject)
		subject_brain = st.selectbox("Select Subject", subjects_options)
	   
 
	# Create a dropdown for lesson
	if subject_brain:
		lessons_data = db.collection("lessons").where("subject_details.subject_id", "==", subjects_id_mapping[subject_brain]).get()
		lesson_options = []
		lesson_id_mapping = {}
		for item in lessons_data:
			lesson = item.to_dict()['lesson_name']
			lesson_id_mapping[lesson] = item.id
			lesson_options.append(lesson)
		# lesson_options
		# lesson_options = [doc.id for doc in db.collection("lessons").where("subject", "==", subject_brain).stream()]
		# lesson_options = ["LESSON1", "LESSON2"]
		lesson_brain = st.selectbox("Select Lesson", lesson_options)
		
 
	# Create a dropdown for topic/activity
	if lesson_brain:
		section_selected = st.selectbox("Select Section Type", ["topics", "activities"])
		
 
		# Create a dropdown for lesson
		#st.write(lesson_id_mapping)
		if section_selected:
			topics_data = db.collection("lessons").document(lesson_id_mapping[lesson_brain]).collection(section_selected).get()
			topic_options = []
			topic_id_mapping = {}
			for item in topics_data:
				try:
					topic = item.to_dict()['topic_name']
				except:
					topic = item.to_dict()['activity_name']
				topic_id_mapping[topic] = item.id
				topic_options.append(topic)
			# lesson_options
			# lesson_options = [doc.id for doc in db.collection("lessons").where("subject", "==", st.session_state["subject_brain"]).stream()]
			# lesson_options = ["LESSON1", "LESSON2"]
			topic_selected = st.selectbox("Select Topic", topic_options)
			paragraph_brain = st.text_area("Enter a paragraph:",key="bain_para", height=200)
			prompt_brain = st.text_area("Enter the prompt:",key="brain_prompt", height=200)
			if(paragraph_brain or prompt_brain):
				mcqs_brain = run_conversation(paragraph_brain,prompt_brain)
				mcq_json=json.loads(mcqs_brain)
				cards=[]
				json_struct={}

				for idx,j in enumerate(mcq_json['questions']):
					json_struct_inter={}
					options = j['options']  # Assuming j['options'] is a list of strings

					# Create a formatted string with a, b, c, ... in front of each option
					result = '\n'.join(f"{chr(97 + i)}. {option}" for i, option in enumerate(options))
					json_struct_inter['front_text']=str(idx+1)+")"+j['question']+'\n'+result
			
					json_struct_inter['back_text']=j['answer']
					json_struct_inter['back_image']=None
					json_struct_inter['front_image']=None
					#st.write(json_struct)
					cards.append(json_struct_inter)
				json_struct['cards']=cards
				json_struct['topic_id']=topic_id_mapping[topic_selected]
				#st.write(json_struct)
				brain_buster_query = db.collection('brain_busters').where('topic_id', '==', json_struct['topic_id']).stream()
				bb_docs = list(brain_buster_query)
				if bb_docs:
					doc_ref = bb_docs[0].id
					db.collection('brain_busters').document(doc_ref).set(json_struct)
				else:
					db.collection('brain_busters').document().set(json_struct)
				save_json_to_text(json_struct, 'output.txt')
				download_button_id = str(uuid.uuid4())
				# Provide a download link for the text file
				st.download_button(
				label="Download Text File",
				data=open('output.txt', 'rb').read(),
				file_name='output.txt',
				mime='text/plain',
				key=download_button_id
						)
		
		

with(tab7):
	# Upload image
	final_data=[]
	#option = st.selectbox(
		#'Choose Number of Questions:',
		#('5', '10', '15', '20'))
	# If an image is uploaded, display and process it

	syllabus  = st.selectbox(
				"Select Syllabus",
		("CBSE", "SSC"),key="syllabus2")
	class_name = st.selectbox(
				"Select class",
		('VI', 'VII', 'VIII', 'IX','X'),key="class2")
	subject_name  = st.selectbox(
				"Select Subject",
		("PHYSICS","SCIENCE","BIOLOGY","CHEMISTRY", "SOCIAL", "HISTORY", "GEOGRAPHY", "CIVICS", "ECONOMICS", "MATHEMATICS", "TELUGU", "HINDI", "ENGLISH"),key="subject2")
	lesson_name	 = st.selectbox(
				"Select lesson",
		("LESSON1", "LESSON2","LESSON3","LESSON4","LESSON5","LESSON6","LESSON7","LESSON8","LESSON9","LESSON10","LESSON11","LESSON12","LESSON13"),key="lesson_name2")
	#paragraph = st.text_area("Enter a paragraph:",text, height=200)

	
	textbook_questions = st.text_area("Enter the textbook questions here:", height=200)
	prompt_textbook_questions = st.text_area("Enter the prompt:",key="textbook questions", height=200,value="workout answers for these textbook questions")
	json_struct={}
	
	if st.button("Get Textbook Questions"):
		if textbook_questions:
			if syllabus == "CBSE":
				subject_collection = db.collection('cbse_subjects')
			elif syllabus == "SSC":
				subject_collection = db.collection('ssc_subjects')
			else:
				raise Exception("Wrong Syllabus")
			 
			subject_data = subject_collection.where("subject_name", "==", subject_name).limit(1).get()[0].to_dict()
			subject_id = subject_data['subject_id']
			 
			lesson_collection = db.collection('lessons')
			lesson_document = lesson_collection.where("lesson_name", "==", lesson_name).where("subject_id", "==", subject_id).where("class", "==", class_name).limit(1)
			lesson_id = lesson_document.get()[0].id
			lp = generate_answers_for_textbookquestions(textbook_questions,chatgpt_url,chatgpt_headers,prompt_textbook_questions)
			print("lp----->",lp)
			lp_json=json.loads(lp)

			for j in lp_json['questions']:
					json_struct={}
					json_struct['class']=class_name
					json_struct['subject']=subject_name
					json_struct['lesson']=lesson_name
					json_struct['question']=j['question']
					json_struct['options']=j['options']
					json_struct['answer']=j['answer']
					json_struct['level']=j['question_level']
					json_struct['question_type']=j['question_type']
					json_struct['type']='single-line'
					if(j['question_type_mcq_or_short_or_long']=='MCQ'):
						json_struct['marks']='1'
					elif(j['question_type_mcq_or_short_or_long']=='Short Question'):
						json_struct['marks']='2'
					else:
						json_struct['marks']='6'
					json_struct['syllabus']=syllabus
					json_struct['subject_id']=subject_id
					json_struct['lesson_id']=lesson_id
					json_struct['access']="public"
					json_struct['metadata']={"tags":[class_name,lesson_name,subject_name,j['question_type_mcq_or_short_or_long'],"textbook-question"]}
					#st.write(json_struct)
					final_data.append(json_struct)
					#st.write(final_data)
			save_json_to_text(final_data, 'output.txt')
			collection = db.collection("question-library")
			for item in final_data:
				doc = collection.document()
				item['question_id'] = doc.id
				doc.set(item)
			download_button_id = str(uuid.uuid4())
			# Provide a download link for the text file
			st.download_button(
						label="Download Textbook Questions",
						data=open('output.txt', 'rb').read(),
						file_name='output.txt',
						mime='text/plain',
					key=download_button_id
			)
			
		else:
			st.write("Please enter the text to generate Summary.")
