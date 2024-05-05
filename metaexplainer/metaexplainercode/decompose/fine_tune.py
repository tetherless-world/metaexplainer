import torch

from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	BitsAndBytesConfig,
	TrainingArguments,
	pipeline
)

import re
import os
import os.path
from pathlib import Path

from peft import LoraConfig, PeftModel, PeftConfig

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
		self.refined_model = None
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
		dataset_file_path = domain_dir_path + '/' + domain_name + '_dataset.jsonl' 

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

		

		#efficient way that doesn't work with pipeline framework
		fine_tuning.model.save_pretrained(codeconstants.OUTPUT_FOLDER + '/llm_results/decompose_refined_model')

		# Save Model - pipeline way
		## change this to include the refined model name
		#fine_tuning.save_model(codeconstants.OUTPUT_FOLDER + '/llm_results/' + self.refined_model_name + '_decompose_refined_model')
		#fine_tuning.model.config.save_pretrained(codeconstants.OUTPUT_FOLDER + '/llm_results/' + self.refined_model_name + '_decompose_refined_model')

		print('Saved the trained model ')


	def get_base_model(self):
		base_model = AutoModelForCausalLM.from_pretrained(
				self.base_model_name,
				torch_dtype=torch.float16,
				device_map="auto",
				#quantization_config=quant_config,
			)
		return base_model

	def set_refined_model(self):
		'''
		Might be unnecessary
		'''
		refined_model = None
		with torch.no_grad():
			# Load the Peft configuration from the saved location
			## change this to include the refined model name
			peft_config = codeconstants.OUTPUT_FOLDER + '/llm_results/decompose_refined_model'

			device_map = {
				"base_model": self.device,
			}


			# Create the PeftModel instance
			refined_model = PeftModel.from_pretrained(self.get_base_model(), peft_config,
														adapter_name="sft", device_map=device_map)

			refined_model.merge_adapter()

			# refined_model = AutoModelForCausalLM.from_pretrained(gdrive_path + 'llama')

			refined_model  = refined_model.to(self.device)

			torch.cuda.empty_cache()
			print(' Refined model ', self.refined_model_name, ' and base model ', self.base_model_name, ' merged.')
		
		self.refined_model = refined_model
	
	def get_refined_model(self):
		return self.refined_model

	def format_instruction_record(self, record, mode = 'test'):
		#print(row)
		prompt = '### System:\n'
		prompt += record['instruction']
		prompt += '\n\n'
		prompt += '### User:\n'
		#print(row)
		prompt += record['input'].strip()
		prompt += '\n\n'
		prompt += '### Response:\n'
		response = ''

		if mode=='train':
			response += record['output']

		prompt += response

		prompt += '\n\n'
		#Could use this for explanation type later
		# prompt += '### Response:\n'
		# prompt += template['label'][label]
		return prompt
	
	def inference(self, passed_dataset, mode='test'):
		inputs = []
		outputs = []
		
		with torch.no_grad():
			for dataset_record in passed_dataset:
				prompt = self.format_instruction_record(dataset_record)
				inputs.append(prompt)

			print('# of Prompts generated', len(inputs), ' and a sample is \n', inputs[0])
			#based on https://discuss.huggingface.co/t/llama2-pad-token-for-batched-inference/48020
			inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).to(self.device)
			#inputs = {k: v.to("cuda") for k, v in inputs.items()}
			print(inputs['input_ids'].shape)

			generate_ids = self.refined_model.generate(**inputs, max_length=500, do_sample=True, top_p=0.9)
			outputs = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
		
			#trying this approach - https://anirbansen2709.medium.com/finetuning-llms-using-lora-77fb02cbbc48
			# self.refined_model = AutoModelForCausalLM.from_pretrained(codeconstants.OUTPUT_FOLDER + '/llm_results/' + self.refined_model_name + '_decompose_refined_model')
			# pipe = pipeline('text-generation', model= self.refined_model, tokenizer=self.tokenizer, 
			# trust_remote_code=True, 
			# max_length = 300,
			# return_full_text=False, device=self.device)

			# for input in inputs:
			# 	out_sample = pipe(input)
			# 	print(out_sample)
			# 	outputs.append(out_sample)
			# 	break

			
			metaexplainer_utils.write_list(outputs, codeconstants.OUTPUT_FOLDER + '/llm_results/' + self.refined_model_name + '_' + mode + '_outputs.txt')

			torch.cuda.empty_cache()

		print('Inference ran on ', mode, 'dataset ')
		return outputs

	def extract_key_value_from_string(self, response_str, find_key):
		extract_str = ''
		extracted_val = re.split('(' + find_key + '):\n?', response_str)[1:3]

		if len(extracted_val) > 1:
				extract_str = extracted_val[1].split('\\n')[0].strip()
		
		return extract_str

	def post_process_results(self, mode='test'):
		'''
		Need to remove instruction from the responses and retain the top-1 alone
		'''
		result_file_name = codeconstants.OUTPUT_FOLDER + '/llm_results/' + self.refined_model_name + '_' + mode + '_outputs.txt'
		keys = ['Question', 'Explanation type', 'Machine interpretation', 'Action', 'Target variable']
		

		read_content = metaexplainer_utils.read_list_from_file(result_file_name)

		regex = re.compile(r'''
			[\S]+:                # a key (any word followed by a colon)
			(?:
			\s                    # then a space in between
				(?!\S+:)\S+       # then a value (any word not followed by a colon)
			)+                    # match multiple values if present
			''', re.VERBOSE)


		for result_str in read_content:
			#only get response onward 
			split_at_response = result_str.split('### Response:')
			rest_of_string = split_at_response[0]
			response = split_at_response[1]
			#print(response)
			val_keys = {field_key: '' for field_key in keys}

			for field in keys:
				val_keys[field] = self.extract_key_value_from_string(response, field)

			if val_keys['Question'] == '':
				#if question is empty - extract from the head string
				val_keys['Question'] = self.extract_key_value_from_string(rest_of_string, 'Question')


			#need to write this to a file
			print(val_keys)
			#would need to add this to a list and do some post-processing on the input too - to make them comparable


	def run(self, mode):
		'''
		Caution: don't run train and test at same time
		'''
		if mode == 'train':
			self.set_base_model()
			self.train(self.train_dataset)
		elif mode == 'test':
			print('Running inference on test datatset')
			self.set_refined_model()
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
	#llm_explanation_interpreter.set_base_model()
	
	#llm_explanation_interpreter.set_datasets('Diabetes')

	#llm_explanation_interpreter.run('train')
	#llm_explanation_interpreter.run('test')

	#post-processing
	llm_explanation_interpreter.post_process_results()

