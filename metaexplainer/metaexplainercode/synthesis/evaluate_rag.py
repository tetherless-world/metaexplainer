from ragas import evaluate
from ragas import evaluate

from langchain.text_splitter import CharacterTextSplitter, NLTKTextSplitter, RecursiveCharacterTextSplitter

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
   context_utilization
)


import sys
sys.path.append('../')

from metaexplainercode import metaexplainer_utils
from metaexplainercode import codeconstants

import pandas as pd
from datasets import Dataset

'''
Construct the question, contexts and answer fields for each result frame so that they can be evaluated.
'''

def chunk_contexts(context): 
    if len(context) > 16000:
        context = context[:15000]

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ' ', ''], chunk_size = 200, chunk_overlap = 0)
    #print(len(context))
    docs = text_splitter.split_text(context)
    #print(docs)
    docs = [doc for doc in docs]
    #print(len(docs))
    return docs

def construct_eval_record(dir_path, sub_dirs):
    '''
    read a result folder and generate the record 
    '''
    explanations_res = pd.read_csv(dir_path + '/Explanations.csv')

    record = pd.read_csv(dir_path + '/Record.csv')
    question = record['Question'].item()

    questions = []
    explanations = []
    contexts = []
    
    ctr = 0

    for sub_dir in sub_dirs:
        explan_record = explanations_res.iloc[ctr]

        explanations.append(explan_record['Subset'])
        subset_txt = str(metaexplainer_utils.drop_unnamed_cols(pd.read_csv(sub_dir + '/Subsets.csv')).to_string(index=False))
        #print('Debug - chunk len', len(chunk_contexts(subset_txt)))
        contexts.append(chunk_contexts(subset_txt))
        questions.append(question)

        explanations.append(explan_record['Explanation'])
        explanation_txt = str(metaexplainer_utils.drop_unnamed_cols(pd.read_csv(sub_dir + '/Results.csv')).to_string(index=False))
        contexts.append(chunk_contexts(explanation_txt))

        questions.append(question)

        ctr+=1

    per_record_result_df = {}
    per_record_result_df['answer'] = explanations
    per_record_result_df['question'] = questions
    per_record_result_df['contexts'] = contexts

    results_df = pd.DataFrame(per_record_result_df)

    #print(per_record_result_df['answers'])
    #print('sample ', results_df.head(10))

    return results_df

def eval_metrics(eval_dataset):
    result = evaluate(
    dataset = eval_dataset, 
    metrics=[
        faithfulness,
        answer_relevancy,
    ],
    )

    df = result.to_pandas()
    return df

if __name__ == '__main__':
    synthesis_dirs = metaexplainer_utils.read_delegate_explainer_outputs(stage='synthesis')
    result_datasets = []

    for result_dir in synthesis_dirs.keys():
        eval_dataset = construct_eval_record(result_dir, synthesis_dirs[result_dir])
        result_datasets.append(eval_dataset)
    
    result_df = pd.concat(result_datasets)
    #result_df['contexts'] = result_df['contexts'].astype(str)

    result_datasets = Dataset.from_pandas(result_df)
    print(result_df.head(10))
    eval_metrics = eval_metrics(result_datasets)
    print(eval_metrics.head(10))