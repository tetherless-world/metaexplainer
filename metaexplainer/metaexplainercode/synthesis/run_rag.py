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
	query_engine = PandasQueryEngine(df=df, verbose=False, synthesize_response=True)
	query_city = "Answer question: What are the cities with the highest population? Give both the city and population? "\
			"Structure of response: Answer in natural-language inlcuding details of what value addresses the question."
	#need to replace this query with EO templates - check how to add the templates
	response = query_engine.query(
		query_city,
	)
	print(response)

	langchain_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), [df, df1], verbose=False)
	out = langchain_agent.invoke(query_city)
	print(out)


