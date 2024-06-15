from ragas import evaluate
from ragas import evaluate

import os

from langchain.text_splitter import CharacterTextSplitter, NLTKTextSplitter, RecursiveCharacterTextSplitter

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
   context_utilization
)

from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	BitsAndBytesConfig,
	TrainingArguments,
	pipeline
)

from langchain import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings


import sys
sys.path.append('../')

from metaexplainercode import metaexplainer_utils
from metaexplainercode import codeconstants
from metaexplainercode.decompose.fine_tune import LLM_ExplanationInterpretor

import pandas as pd
from datasets import Dataset

'''
Construct the question, contexts and answer fields for each result frame so that they can be evaluated.
'''

def chunk_contexts(context, context_size): 
    if len(context) > context_size:
        context = context[:context_size]

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ' ', ''], chunk_size = 200, chunk_overlap = 0)
    #print(len(context))
    docs = text_splitter.split_text(context)
    #print(docs)
    docs = [doc for doc in docs]
    #print(len(docs))
    return docs

def construct_eval_record(dir_path, sub_dirs, context_size):
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
        contexts.append(chunk_contexts(subset_txt, context_size))
        questions.append(question)

        explanations.append(explan_record['Explanation'])
        explanation_txt = str(metaexplainer_utils.drop_unnamed_cols(pd.read_csv(sub_dir + '/Results.csv')).to_string(index=False))
        contexts.append(chunk_contexts(explanation_txt, context_size))

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

def load_llm_model():
    base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    refined_model_name = "llama-3-8b-charis-explanation" #You can give it your own name

    llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, use_auth_token=True)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "left"  # Fix for fp16
    print('Tokenizer setup ')
	
	#instantiating class 
    llm_explanation_interpreter = LLM_ExplanationInterpretor(llama_tokenizer, base_model_name, refined_model_name)
    #llm_explanation_interpreter.set_refined_model()

    

    llama_llm = pipeline(
    tokenizer=llama_tokenizer,
    model=llm_explanation_interpreter.set_base_model(),
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    temperature=0.1, 
    repetition_penalty=1.1
    )

    evaluator = HuggingFacePipeline(pipeline=llama_llm)

    return evaluator


def eval_metrics(eval_dataset):
    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    # bnb_4bit_compute_dtype=torch.float16,
    # bnb_4bit_quant_type="nf4",
    # bnb
    #_4bit_use_double_quant=True,
    )
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # tiny_llm = pipeline(
    # tokenizer=AutoTokenizer.from_pretrained(model_name),
    # model=AutoModelForCausalLM.from_pretrained(model_name),
    # return_full_text=True,  # langchain expects the full text
    # task='text-generation',
    # temperature=0.1, 
    # repetition_penalty=1.1
    # )

    # embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    evaluator = load_llm_model()

    result = evaluate(
    dataset = eval_dataset, 
    llm = evaluator,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_utilization
    ],
    )

    df = result.to_pandas()
    return df

if __name__ == '__main__':
    synthesis_dirs = metaexplainer_utils.read_delegate_explainer_outputs(stage='synthesis')
    result_datasets = []
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    context_size = 2048

    for result_dir in synthesis_dirs.keys():
        eval_dataset = construct_eval_record(result_dir, synthesis_dirs[result_dir], context_size)
        result_datasets.append(eval_dataset)
    
    result_df = pd.concat(result_datasets)
    #result_df['contexts'] = result_df['contexts'].astype(str)

    result_datasets = Dataset.from_pandas(result_df)
    print(result_df.head(10))
    eval_metrics = eval_metrics(result_datasets)
    eval_metrics.to_csv(codeconstants.SYNTHESIS_FOLDER + '/Evaluations_rag.csv')
    print(eval_metrics.head(10))