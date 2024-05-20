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


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

if __name__=='__main__':
	df = pd.DataFrame(
	{
		"city": ["Toronto", "Tokyo", "Berlin"],
		"population": [2930000, 13960000, 3645000],
	})
	query_engine = PandasQueryEngine(df=df, verbose=True, synthesize_response=True)
	response = query_engine.query(
		"What is the city with the highest population? Give both the city and population",
	)
	print(str(response))

