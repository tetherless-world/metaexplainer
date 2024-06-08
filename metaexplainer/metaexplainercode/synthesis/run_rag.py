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

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

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

	for output_folder in delegate_output_folders.keys():
		dir_ep = output_folder.split('/')[-1]
		sub_dirs = delegate_output_folders[output_folder]

		record_dets = metaexplainer_utils.read_delegate_records_df(output_folder + '/record.csv')

		prompt_record = {}

		prompt_record['Explanation Type'] = record_dets.iloc[0]['Explanation type']
		prompt_record['Explainer Method'] = dir_ep.split('_')[2]
		prompt_record['Explanation Modality'] = explainer_modality_pd[explainer_modality_pd['Instances'].str.lower() == prompt_record['Explainer Method']]['Modality'].item()
		prompt_record['Question'] = record_dets['Question'].item()
		prompt_record['Feature groups'] = record_dets['Feature groups'].item()

		results_list = []
		evaluations_list = []

		metrics = []

		for sub_dir in sub_dirs:
			results_list.append(pd.read_csv(sub_dir + '/Results.csv'))
			evaluations_list.append(pd.read_csv(sub_dir + '/Evaluations.csv'))
		
		evaluations_df = pd.concat([x.head(1) for x in evaluations_list], ignore_index=True)

		print(evaluations_df)
		grouped_mean = evaluations_df.groupby('Metric')['Value'].mean().reset_index()

		# Renaming the columns for better readability
		grouped_mean.columns = ['Metric', 'mean_value']

		prompt_record['Metrics'] = grouped_mean
		print(prompt_record['Metrics'])
		break






	df = pd.DataFrame(
	{
		"city": ["Toronto", "Tokyo", "Berlin"],
		"population": [2930000, 13960000, 3645000],
	})
	df1 = pd.DataFrame(
		{
			"city": ["Bangalore", "Bombay", "Berlin"],
		"population": [2930000, 13960000, 3645000],
		}
	)
	
	query_city = "Answer question: What are the cities with the highest population? Give both the city and population? "\
			"Structure of response: Answer in natural-language inlcuding details of what value addresses the question."

	langchain_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), [df, df1], verbose=False)
	out = langchain_agent.invoke(query_city)
	print(out)

	# query_engine = PandasQueryEngine(df=df, verbose=False, synthesize_response=True)
	
	# #need to replace this query with EO templates - check how to add the templates
	# response = query_engine.query(
	# 	query_city,
	# )
	# print(response)


