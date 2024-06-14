'''
Run RAG on outputs from explainer methods with prompts for identified explanation type from EO
Base it off of: https://towardsai.net/p/machine-learning/query-your-dataframes-with-powerful-large-language-models-using-langchain
Or https://docs.llamaindex.ai/en/stable/examples/query_engine/pandas_query_engine/
'''

import logging
import sys
from IPython.display import Markdown, display

import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine

from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import OpenAI

import sys
sys.path.append('../')

from metaexplainercode import metaexplainer_utils
from metaexplainercode import codeconstants
from metaexplainercode import ontology_utils
from metaexplainercode.synthesis.handle_explainer_outputs import ParseExplainerOutput

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from jinja2 import Template

def edit_results(prompt_record): #will be crucial to edit this for better feature groupsgi
	'''
	Don't create entries for multiple sub-results if they are not needed, they could be created because of a parsing error.
	- If the length of feature groups is less than length of sub_dirs - use just length of feature groups
	- If record subsets are same across result sets, use non-duplicates 
	'''
	len_fg = len(prompt_record['feature_groups'])

	#also drop feature groups that have only patient and 
	if len_fg < len(prompt_record['Results']):
		prompt_record['Results'] = prompt_record['Results'][:len_fg]
		prompt_record['Metrics'] = prompt_record['Metrics'][:len_fg]
		prompt_record['Subsets'] = prompt_record['Subsets'][:len_fg]
	
	#edit feature groups to remove empty features 
	for fg in prompt_record['feature_groups']:
		fg.pop('Diabetes', 'No key found')


	duplicates = []

	for i in range(0, len(prompt_record['Results'])):
		for j in range(i + 1, len(prompt_record['Results'])):
			if metaexplainer_utils.are_dfs_equal(prompt_record['Results'][i], prompt_record['Results'][j]):
				duplicates.append(j)
	
	prompt_record['Results'] = [prompt_record['Results'][i] for i in range(0, len(prompt_record['Results'])) if i not in duplicates]
	prompt_record['Metrics'] = [prompt_record['Metrics'][i] for i in range(0, len(prompt_record['Results'])) if i not in duplicates]
	prompt_record['Subsets'] = [prompt_record['Subsets'][i] for i in range(0, len(prompt_record['Results'])) if i not in duplicates]

	if len(prompt_record['Results']) > len(prompt_record['Subsets']):
		prompt_record['Results'] = prompt_record['Results'][: len(prompt_record['Subsets'])]
		prompt_record['Metrics'] = prompt_record['Metrics'][: len(prompt_record['Subsets'])]

	return prompt_record
	
def construct_prompt_record(output_folder):
	dir_ep = output_folder.split('/')[-1]
	sub_dirs = delegate_output_folders[output_folder]

	record_dets = metaexplainer_utils.read_delegate_records_df(output_folder + '/record.csv')

	prompt_record = {}
	prompt_record['Record'] = record_dets

	prompt_record['Explanation_Type'] = record_dets.iloc[0]['Explanation type']
	prompt_record['Definition'] = explanation_definition_pd[explanation_definition_pd['Explanation type'] == prompt_record['Explanation_Type']]['Definition'].item()
	prompt_record['Explainer_Method'] = dir_ep.split('_')[2]
	prompt_record['Modality'] = explainer_modality_pd[explainer_modality_pd['Instances'].str.lower() == prompt_record['Explainer_Method']].iloc[0]['Modality']
	prompt_record['Question'] = record_dets['Question'].item()
	prompt_record['feature_groups'] = eval(record_dets['Feature groups'].item())

	parse_explainer_obj = ParseExplainerOutput(prompt_record['Modality'], prompt_record['Explanation_Type'])
	

	results_list = []
	evaluations_list = []
	originals_list = []
	subset_list = []
	modality_func_finder = prompt_record['Modality'].lower().replace(' ', '_')

	for sub_dir in sub_dirs:
		(edited_result, use_question) = getattr(parse_explainer_obj,'parse' + '_' + modality_func_finder)(pd.read_csv(sub_dir + '/Results.csv'))
		#print('Debug ', edited_result)
		results_list.append(edited_result)
		evaluations_list.append(pd.read_csv(sub_dir + '/Evaluations.csv'))
		if prompt_record['Explanation_Type'] == 'Counterfactual Explanation':
			originals_list.append(pd.read_csv(sub_dir + '/Original.csv'))
		subset_list.append(pd.read_csv(sub_dir + '/Subset.csv'))
		
	prompt_record['Results'] = results_list
	prompt_record['Subsets'] = subset_list
	prompt_record['Metrics'] = evaluations_list
	if prompt_record['Explanation_Type'] == 'Counterfactual Explanation':
		prompt_record['Originals'] = originals_list
	prompt_record['use_question'] = use_question

	prompt_record = edit_results(prompt_record)

	#don't group at this stage
	# evaluations_df = pd.concat([x.head(10) for x in evaluations_list], ignore_index=True)


	# grouped_mean = evaluations_df.groupby('Metric')['Value'].mean().reset_index()

	# 	# Renaming the columns for better readability
	# grouped_mean.columns = ['Metric', 'mean_value']

	# prompt_record['Metrics'] = grouped_mean

	return prompt_record

def retrieve_prompt_subset():
	prompt_template_text = '''
	Find a match in the data based on feature group: {{feature_group}}. 
	Context: The outcome variable is {{outcome_variable}}.
	If there are no full matches, summarize the dataset.'''
	# There are several dataframes in the answer that answer the feature groups identified:
	# {%- for group in feature_groups %}
	# {{group}} \n
	# {%- endfor %}
	# Results were generated by {{Explainer_Method}}.
	# '''
	return prompt_template_text


def retrieve_prompt_explanation():
	#certain kind of prediction and features in certain ways, and that fact that it uses confidence 
	prompt_template_text = '''
	Summarize the data in English. Structure your response based on the expected format for: {{Explanation_Type}} being that {{Definition}}.
	'''

	return prompt_template_text

def save_explanations(explanation_type_explainer_timestamp_dir, records, 
					  explanations_aux, explanations, queries,
					  explanation_type):
	'''
	Use the same folder name as delegate so that it is easy to retrieve from delegate
	'''
	synthesis_results_folder = codeconstants.SYNTHESIS_FOLDER + '/results/'
	synthesis_instance_folder = synthesis_results_folder + '/' + explanation_type_explainer_timestamp_dir

	metaexplainer_utils.create_folder(synthesis_results_folder)
	metaexplainer_utils.create_folder(synthesis_instance_folder)

	ctr = 0
	curr_folder = synthesis_instance_folder + '/' + str(ctr)

	'''
	Persist explanation, subsets, results and metrics for each set
	'''
	for explanation in explanations_aux:
		metaexplainer_utils.create_folder(curr_folder)
		explanation['Results'].to_csv(curr_folder + '/' + 'Results.csv')
		explanation['Subsets'].to_csv(curr_folder + '/' + 'Subsets.csv')

		if explanation_type == 'Counterfactual Explanation':
			#print(explanation['Originals'])
			explanation['Originals'].to_csv(curr_folder + '/' + 'Original.csv')

		explanation['Metrics'].to_csv(curr_folder + '/' + 'Metrics.csv')

		ctr += 1
		curr_folder = synthesis_instance_folder + '/' + str(ctr)
	
	records.to_csv(synthesis_instance_folder + '/Record.csv')
	
	pd.DataFrame(explanations).to_csv(synthesis_instance_folder + '/Explanations.csv')
	pd.DataFrame(queries).to_csv(synthesis_instance_folder + '/Queries.csv')

	explan_text = ''
	group_ctr = 0
	for explan_record in explanations:
		explan_text += 'Group ' + str(group_ctr) + '\n'
		explan_text += 'Matched subset ' + explan_record['Subset'] + '\n'
		explan_text += 'Explainer explanation ' + explan_record['Explanation'] + '\n'
		explan_text += '---------\n'
	
	with open(synthesis_instance_folder + '/Explanations.txt', 'w') as f:
		f.write(explan_text)
	
	print('Created folder for ', synthesis_instance_folder)


def run_rag_on_record(output_folder, domain_name):
	prompt_record = construct_prompt_record(output_folder)
	print(prompt_record)
	print('Lengths ', len(prompt_record['Results']), len(prompt_record['Subsets']))
	explans_list = []

	explanation_aux = []
	queries_list = []
	
	for i in range(0, len(prompt_record['Results'])):	
		if not prompt_record['use_question']:
			feature_group = prompt_record['feature_groups'][i]
		else:
			feature_group = prompt_record['Question']

		if not (len(feature_group) == 1 and 'patient' in feature_group.keys()):
			to_add_in_aux = {'Results': prompt_record['Results'][i], 'Subsets': prompt_record['Subsets'][i], 'Metrics': prompt_record['Metrics'][i]}
					
			if len(feature_group) == 0:
				feature_group = prompt_record['Question']

			cols_in_question = metaexplainer_utils.find_labels_in_sentence(prompt_record['Question'], domain_name)
			cols_in_question = list(set(cols_in_question))
			print('Debug ', cols_in_question)

			filled_prompt = prompt_template.render(prompt_record)
			filled_prompt_subset = prompt_template_subset.render({'feature_group': feature_group, 'outcome_variable': 'Outcome'})

			if len(cols_in_question) > 0:
				filled_prompt += 'If features from question which are: ' + str(cols_in_question) + 'in dataframe, tailor your answers to them.'
			
			query_engine = PandasQueryEngine(df=prompt_record['Subsets'][i], verbose=False, synthesize_response=True)

			if prompt_record['Explanation_Type'] == 'Counterfactual Explanation':
				filled_prompt += 'The dataframe contains changes and not actual values.'
					
			response = query_engine.query(
				filled_prompt_subset,
			)
			print('Matched subset from data: ',str(i),' ', response)


			query_engine_explan = PandasQueryEngine(df=prompt_record['Results'][i], verbose=False, synthesize_response=True)
			response_explan = query_engine_explan.query(
				filled_prompt
			)

			explans_list.append({'Subset': str(response), 'Explanation': str(response_explan)})
			queries_list.append({'Subset Query': filled_prompt_subset, 'Explanation Query': filled_prompt, 'Feature group': feature_group})

			if prompt_record['Explanation_Type'] == 'Counterfactual Explanation':
				to_add_in_aux['Originals'] = prompt_record['Originals'][i]

			explanation_aux.append(to_add_in_aux)
			print('Explanations for group: ',str(i),' ', response_explan)
			
	print('-----')
	save_explanations(output_folder.split('results/')[-1], prompt_record['Record'], explanation_aux, explans_list, queries_list, prompt_record['Explanation_Type'])
		

if __name__=='__main__':
	'''
	Pass into prompt:
	- EO template
	- Metrics 
	- Question 
	- Modality 
	Dataframes (all will have same columns so that is good):
	- Explainer results
	'''
	delegate_output_folders = metaexplainer_utils.read_delegate_explainer_outputs()

	explainer_modality_pd = pd.read_csv(codeconstants.DELEGATE_FOLDER + '/explanation_type_methods.csv')
	explanation_definition_pd = pd.read_csv(codeconstants.SYNTHESIS_FOLDER + '/explanation_type_definition.csv')

	prompt_template_text_explan = retrieve_prompt_explanation()
	prompt_subset = retrieve_prompt_subset()

	
	prompt_template = Template(prompt_template_text_explan)
	prompt_template_subset = Template(prompt_subset)

	domain_name = 'Diabetes'
	

	for output_folder in delegate_output_folders.keys():
		run_rag_on_record(output_folder, domain_name)

	
	'''
	Trials
	'''
	# langchain_agent_subsets = create_pandas_dataframe_agent(OpenAI(temperature=0, seed=3), prompt_record['Subsets'], max_iterations = 50, verbose=True)
		# out = langchain_agent_subsets.invoke(filled_prompt_subset)
		# print('Subsets', out)

		# langchain_agent = create_pandas_dataframe_agent(OpenAI(temperature=0, seed=3), prompt_record['Results'], max_iterations = 50, verbose=True)
		# out = langchain_agent.invoke(filled_prompt)
		# print('Summary of results',out)
		

	# df = pd.DataFrame(
	# {
	# 	"city": ["Toronto", "Tokyo", "Berlin"],
	# 	"population": [2930000, 13960000, 3645000],
	# })
	# df1 = pd.DataFrame(
	# 	{
	# 		"city": ["Bangalore", "Bombay", "Berlin"],
	# 	"population": [2930000, 13960000, 3645000],
	# 	}
	# )
	
	# query_city = "Answer question: What are the cities with the highest population? Give both the city and population? "\
	# 		"Structure of response: Answer in natural-language inlcuding details of what value addresses the question."

	# langchain_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), [df, df1], verbose=False)
	# out = langchain_agent.invoke(query_city)
	# print(out)

	


