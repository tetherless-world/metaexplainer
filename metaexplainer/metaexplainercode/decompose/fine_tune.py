import torch

from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	BitsAndBytesConfig,
	TrainingArguments,
	pipeline
)
import os
import os.path
from pathlib import Path

from peft import LoraConfig
from trl import SFTTrainer

from datasets import load_dataset

import json

import yaml
import pandas as pd

import sys
sys.path.append('../')
from metaexplainercode import codeconstants

from metaexplainercode import metaexplainer_utils

class LLM_ExplanationInterpretor():
	def __init__(self, tokenizer, base_model_name, refined_model_name):
		self.tokenizer = tokenizer
		self.base_model_name = base_model_name
		self.refined_model_name = refined_model_name
		self.template = yaml.safe_load(open(codeconstants.PROMPTS_FOLDER + '/question_decompose.yaml'))
		self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
		#defining variables that get set through setter functions later on
		self.base_model = None
		self.train_dataset = None
		self.test_dataset = None

	def format_instruction(self, row):
		#print(row)
		prompt = '### System:\n'
		prompt += self.template['instruction']
		prompt += '\n\n'
		prompt += '### User:\n'
		#print(row)
		prompt += self.template['task'].replace(
			'{question}', row['Question']
		)
		prompt += '\n\n'
		prompt += '### Response:\n'
		#print(row)
		response = self.template['response'].replace(
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

	def dump_jsonl(self, data, output_path, append=False):
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

	def process_jsonl_file(self, output_file_path, dataset):
		list_objects = []

		for idx, item in dataset.iterrows():
		  (prompt, response) = self.format_instruction(item)
		  json_object = {
					"text": prompt,
					"instruction":self.template['instruction'],
					"input": item["Question"],
					"output": response,
					"label": item["Explanation type"]}
		  list_objects.append(json_object)

		self.dump_jsonl(list_objects, output_file_path)


	def set_datasets(self, domain_name):
		domain_dir_path = codeconstants.DECOMPOSE_QUESTIONS_FOLDER + '/' + domain_name
		dataset_file_path = domain_dir_path + domain_name + '_dataset.jsonl' 

		try:
			my_abs_path = Path(dataset_file_path).resolve(strict=True)
		except FileNotFoundError:
			self.process_jsonl_file(dataset_file_path, pd.read_csv(domain_dir_path + '/finetune_questions.csv'))
		
		dataset = load_dataset('json', data_files= dataset_file_path)
		splits = dataset['train'].train_test_split(test_size=0.2) #https://huggingface.co/docs/datasets/v1.8.0/package_reference/main_classes.html#datasets.Dataset.train_test_split
		self.train_dataset = splits['train']
		self.test_dataset = splits['test']
		print('Datasets loaded ')
		#return train_dataset, test_dataset
	
	def set_base_model(self):
		# Quantization Config
		quant_config = BitsAndBytesConfig(
			load_in_4bit=True,
			bnb_4bit_quant_type="nf4",
			bnb_4bit_compute_dtype=torch.float16,
			bnb_4bit_use_double_quant=False
		)

		# Model
		self.base_model = AutoModelForCausalLM.from_pretrained(
			self.base_model_name,
			quantization_config=quant_config,
			device_map={"": 0}
		)
		self.base_model.config.use_cache = True
		self.base_model.config.pretraining_tp = 1
		print('Base model downloaded', self.base_model_name)

	def train(self, train_dataset):
		#LORA config
		peft_parameters = LoraConfig(
			lora_alpha=16,
			lora_dropout=0.1,
			r=8,
			bias="none",
			task_type="CAUSAL_LM"
		)

		# Training Params
		train_params = TrainingArguments(
			output_dir= codeconstants.OUTPUT_FOLDER + "/llm_results/decompose_results_modified",
			num_train_epochs=5,
			per_device_train_batch_size=4,
			gradient_accumulation_steps=1,
			optim="paged_adamw_32bit",
			save_steps=25,
			logging_steps=25,
			learning_rate=2e-4,
			weight_decay=0.001,
			fp16=False,
			bf16=False,
			max_grad_norm=0.3,
			max_steps=-1,
			warmup_ratio=0.03,
			group_by_length=True,
			lr_scheduler_type="constant",
			report_to="tensorboard"
		)

		# Trainer
		fine_tuning = SFTTrainer(
			model=self.base_model,
			train_dataset = train_dataset,
			dataset_text_field="text",
			#eval_dataset = test_dataset,
			peft_config=peft_parameters,
			tokenizer= self.tokenizer,
			#compute_metrics= compute_metrics,
			#preprocess_logits_for_metrics = preprocess_logits_for_metrics,
			args=train_params
		)

		#fine_tuning.add_callback(CustomCallback(fine_tuning))
		print('Parameters configured, starting to finetune ', self.refined_model_name)

		# Training
		train_result = fine_tuning.train()

		# Save Model
		fine_tuning.model.save_pretrained(codeconstants.OUTPUT_FOLDER + '/llm_results/decompose_refined_model')
		print('Saved the trained model ')


	def inference(self, test_dataset):
		pass

	def run(self, mode):
		if mode == 'train':
			self.train(self.train_dataset)
		elif mode == 'test':
			self.inference(self.test_dataset)


if __name__== "__main__":
	# Model and tokenizer names
	base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
	refined_model_name = "llama-3-8b-charis-explanation" #You can give it your own name

	#defining variables necessary for instantiation
	llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, use_auth_token=True)
	llama_tokenizer.pad_token = llama_tokenizer.eos_token
	llama_tokenizer.padding_side = "right"  # Fix for fp16

	print('Tokenizer setup ')
	
	#instantiating class 
	llm_explanation_interpreter = LLM_ExplanationInterpretor(llama_tokenizer, base_model_name, refined_model_name)
	llm_explanation_interpreter.set_base_model()
	
	llm_explanation_interpreter.set_datasets('Diabetes')

	llm_explanation_interpreter.run('train')

