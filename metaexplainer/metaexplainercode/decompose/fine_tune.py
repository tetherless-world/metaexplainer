import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig

from datasets import load_dataset

import json

import yaml

import sys
sys.path.append('../')
from metaexplainercode import codeconstants

from metaexplainercode import metaexplainer_utils

class LLM_ExplanationInterpretor():
	def __init__():
		self.tokenizer = tokenizer
		self.base_model_name = base_model_name
		self.refined_model_name = refined_model_name
		self.template = yaml.safe_load(open(codeconstants.PROMPTS_FOLDER + '/question_decompose.yaml'))

	def format_instruction(row):
	    #print(row)
	    prompt = '### System:\n'
	    prompt += template['instruction']
	    prompt += '\n\n'
	    prompt += '### User:\n'
	    #print(row)
	    prompt += template['task'].replace(
	        '{question}', row['Question']
	    )
	    prompt += '\n\n'
	    prompt += '### Response:\n'
	    #print(row)
	    response = template['response'].replace(
	        '{parse}', row['Machine interpretation']
	    ).replace(
	        '{action}', row['Action'] if not pd.isna(row['Action']) else ''
	    ).replace(
	        '{explanation type}', row['Explanation type']
	    ).replace(
	        '{target variable}', row['Target variable']
	    )
	    prompt += response

	    prompt += '\n\n'
	    #Could use this for explanation type later
	    # prompt += '### Response:\n'
	    # prompt += template['label'][label]
	    return prompt, response

	def dump_jsonl(data, output_path, append=False):
	    """
	    Write list of objects to a JSON lines file.
	    Taken from: https://galea.medium.com/how-to-love-jsonl-using-json-line-format-in-your-workflow-b6884f65175b
	    """
	    mode = 'a+' if append else 'w'
	    with open(output_path, mode, encoding='utf-8') as f:
	        for line in data:
	            json_record = json.dumps(line, ensure_ascii=False)
	            f.write(json_record + '\n')
	    print('Wrote {} records to {}'.format(len(data), output_path))

    def process_jsonl_file(output_file_path, dataset):
	    list_objects = []

	    for idx, item in dataset.iterrows():
	      (prompt, response) = format_instruction(item)
	      json_object = {
	                "text": prompt,
	                "instruction": template['instruction'],
	                "input": item["Question"],
	                "output": response,
	                "label": item["Explanation type"]}
	      list_objects.append(json_object)

	    dump_jsonl(list_objects, output_file_path)


	def load_datasets(domain_name):
		domain_dir_path = codeconstants.DECOMPOSE_QUESTIONS_FOLDER + '/' + domain
		dataset_file = domain_dir_path '/' + domain_name + '_dataset.jsonl' 

		if os.file_exists(dataset_file):
			process_jsonl_file(dataset_file, pd.read_csv(domain_dir_path + '/finetune_questions.csv'))
		
		dataset = load_dataset('json', data_files= dataset_file)
		splits = dataset['train'].train_test_split(test_size=0.2) #https://huggingface.co/docs/datasets/v1.8.0/package_reference/main_classes.html#datasets.Dataset.train_test_split
		train_dataset = splits['train']
		test_dataset = splits['test']
		return train_dataset, test_dataset

	def create_model():
		pass

	def train():
		pass


	def inference():
		pass

	def run(mode):
		if mode == 'train':
			train()
		elif mode == 'test':
			inference


if __name__== "__main__":
	pass

