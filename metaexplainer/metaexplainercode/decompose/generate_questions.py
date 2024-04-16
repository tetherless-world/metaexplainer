from openai import OpenAI
client = OpenAI()

from jinja2 import Template
import pandas as pd

import sys
sys.path.append('../')
from metaexplainercode import codeconstants

def retrieve_prompt():
	#certain kind of prediction and features in certain ways, and that fact that it uses confidence 
	prompt_template_text = '''
 Generate 5 questions for {{explanation}} for the Diabetes domain. {{explanation}} questions are of the form, 
 {%- for question in questions %}
	{{question}} \n
	{%- endfor %}


	Your features could be:
	 {%- for feature in features %}
	{{feature}} \n
	{%- endfor %}.
	 and the prediction is {{prediction}}. 

	 The feature values range from 
	 {%- for feature_range in feature_ranges %}
	{{feature_range}} \n
	{%- endfor %}.

	For each question, 
	also generate a machine interpretation which includes:
	 predicate logic translation of the question, 
	 action mentioned in the question,
	 explanation type: {{explanation}}, 
	 and the target variable in the question and if there is a low / high likelihood.
	'''

	return prompt_template_text


def create_prompt_record(explanation_type, questions, features, prediction, feature_ranges):
	#print('Trial ', sample)
	return {
		"explanation" : explanation_type,
		"questions" : questions,
		"features" : features,
		"prediction": prediction,
		"feature_ranges" : feature_ranges
	}

if __name__=="__main__":

	conts = ''
	prompt_template_text = retrieve_prompt()
	print(prompt_template_text)
	prompt_template = Template(prompt_template_text)

	#define the features and feature ranges on a per dataset basis - can have a common util function for this
	features = ['Age', 'Sex', 'Diabetes Pedigree Function', 'BMI']
	feature_ranges = ['Age: 30 - 70', 'Sex: Male or Female', 'Diabetes Pedigree Function: 0.2 - 0.1', 'BMI: 18 - 30']

	selected_explanation_types = ['Case Based Explanation', 'Contrastive Explanation', 'Data Explanation', 'Rationale Explanation', 'Contextual Explanation', 'Counterfactual Explanation']
	questions_file = pd.read_csv(codeconstants.OUTPUT_FOLDER + '/prototypical_questions_explanations_eo.csv')

	for explanation_type in selected_explanation_types:
		questions_explanation = (questions_file[questions_file['explanation'] == explanation_type]['question']).to_list()
		prompt = create_prompt_record(explanation_type, questions_explanation, features, 'whether a patient has Diabetes or not',feature_ranges)
		filled_prompt = prompt_template.render(prompt)
		print(filled_prompt)

		for i in range(0, 1):
			completion = client.chat.completions.create(
				model="gpt-3.5-turbo",
				messages=[
					# {"role": "system",
					#  "content": "You are an explanation assistant skilled at understanding the needs of users and the questions they could ask for which they require explanations. Anticipate what their needs might be based on the user's requirements and generate questions."},
					 {"role": "system",
					 "content": "Always answer in English."},
					 {"role": "system",
					 "content": "Always keep the target variable as Diabetes prediction."},
					{"role": "user",
					 "content": filled_prompt}
					 #recognizing in the question - all instances, all features - their filters and the values that are being compared, all datasets and the target variable and whether it is a low or high likelihood.
				],
				temperature=0
				#above part of prompt 2 can be used across question types
			)

			output = completion.choices[0].message
			conts += output.content + '\n \n'

			print("Run ", str(i), 'for explanation ', explanation_type)

	with open(codeconstants.OUTPUT_FOLDER + '/gpt-questions.txt', 'w') as f:
		f.write(conts)