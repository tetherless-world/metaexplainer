import torch

from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	BitsAndBytesConfig,
	TrainingArguments,
	pipeline
)


from sklearn.metrics import classification_report

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

	def get_datasets(self, domain_name):
		domain_dir_path = codeconstants.DECOMPOSE_QUESTIONS_FOLDER + '/' + domain_name
		dataset_file_path = domain_dir_path + '/' + domain_name + '_dataset.jsonl' 
		train_dataset_path = domain_dir_path + '/' + domain_name + '_train_dataset.jsonl' 
		test_dataset_path = domain_dir_path + '/' + domain_name + '_test_dataset.jsonl'

		self.train_dataset = load_dataset('json', data_files= train_dataset_path)
		self.test_dataset = load_dataset('json', data_files= test_dataset_path)
		print('Split used for training loaded.')
	
	def set_datasets(self, domain_name):
		domain_dir_path = codeconstants.DECOMPOSE_QUESTIONS_FOLDER + '/' + domain_name
		dataset_file_path = domain_dir_path + '/' + domain_name + '_dataset.jsonl' 
		train_dataset_path = domain_dir_path + '/' + domain_name + '_train_dataset.jsonl' 
		test_dataset_path = domain_dir_path + '/' + domain_name + '_test_dataset.jsonl' 

		try:
			my_abs_path = Path(dataset_file_path).resolve(strict=True)
		except FileNotFoundError:
			self.process_jsonl_file(dataset_file_path, pd.read_csv(domain_dir_path + '/finetune_questions.csv'))
		
		dataset = load_dataset('json', data_files= dataset_file_path)
		splits = dataset['train'].train_test_split(test_size=0.2) #https://huggingface.co/docs/datasets/v1.8.0/package_reference/main_classes.html#datasets.Dataset.train_test_split
		self.train_dataset = splits['train']
		self.test_dataset = splits['test']
		self.dump_jsonl(self.train_dataset, train_dataset_path)
		self.dump_jsonl(self.test_dataset, test_dataset_path)
		print('Dataset split generated and datasets loaded. Train: ', len(self.train_dataset),
		'And test: ', len(self.test_dataset))
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
			num_train_epochs=12,
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
			for dataset_record in passed_dataset['train']:
				#print('Dataset record ', dataset_record)
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
			parts = extracted_val[1].split('\\n')
			for str_val in parts:
				if str_val != '':
					#find first non empty string and set that!
					extract_str = str_val.strip()
					break
		
		return extract_str

	def post_process_results(self, mode='test'):
		'''
		Need to remove instruction from the responses and retain the top-1 alone
		This would need to be called at the compute-F1
		'''
		result_file_name = codeconstants.OUTPUT_FOLDER + '/llm_results/' + self.refined_model_name + '_' + mode + '_outputs.txt'
		keys = ['Explanation type', 'Machine interpretation', 'Action', 'Target variable']
		
		#loads content in decoded form - while writing or returning it back need to use encode.
		read_content = metaexplainer_utils.read_list_from_file(result_file_name)

		result_dict = []

		for result_str in read_content:
			#only get response onward 
			split_at_response = result_str.split('### Response:')
			rest_of_string = split_at_response[0]
			response = split_at_response[1]
			#print(response)
			#print(rest_of_string)
			val_keys = {field_key: '' for field_key in keys}

			for field in keys:
				val_keys[field] = self.extract_key_value_from_string(response, field).encode('utf-8')

			
			val_keys['Question'] = self.extract_key_value_from_string(str(rest_of_string), 'User')
			
			result_dict.append(val_keys)
			#print(val_keys)
		
		result_dictionary = {}

		for record in result_dict:
			question = record['Question']
			result_dictionary[question] = {}

			del record['Question']
			
			result_dictionary[question] = record
		
		return result_dictionary

	
	def post_process_input(self, mode='test'):
		'''
		Return a dicitonary version of the input to easily compare against prediction output
		{<question>}:{'Machine interpretation':<>, 'Action':<>, 'Explanation type':<>, 'Prediction': <>}
		'''
		input_dict = {}

		dataset_iter = self.test_dataset
		#neeed to fix the output loading 
		#print(dataset_iter['train'])

		if mode == 'train':
			dataset_iter = self.train_dataset

		for record in dataset_iter['train']:
			#print(record)
			input_dict[record['input'].strip()] = {output.split(':')[0]: output.split(':')[1].strip() for output in record['output'].split('\n')}

		return input_dict
	
	def compute_metrics(self, domain_name, mode='test'):
		'''
		Get the results in a dictionary and then use the input to find outputs based on input
		'''
		#Call get dataset if files exist
		results = self.post_process_results(mode)
		inputs = self.post_process_input(mode)
		print('Length of results ', len(results.keys()))

		

		#Defining hash key retrievers
		not_found = []
		found_questions_results = []
		found_questions_references = []
		question_comparisons = list(results.keys())

		for result_question in results.keys():
			#trying to find non matches 
			if type(result_question) != str:
				result_question = result_question.decode('utf-8')

			if not result_question in inputs.keys():
				not_found.append(result_question)
			else:
				found_questions_results.append(results[result_question])
				found_questions_references.append(inputs[result_question])

		#converting the results and references to pandas since it is easier to load
		found_questions_results = pd.DataFrame(found_questions_results)
		found_questions_references = pd.DataFrame(found_questions_references)
		
		print('Computing accuracy by explanation type')
		reference_explanation_types = found_questions_references['Explanation type']
		unique_explanation_types = list(reference_explanation_types.unique())
		print('Labels for explanation types are ', unique_explanation_types)
		#print('Results ', list(found_questions_results['Explanation type']))
		#print('References ', list(found_questions_references['Explanation type']))
		metaexplainer_utils.generate_confusion_matrix_and_visualize(list(found_questions_results['Explanation type'].astype(str)),
															   list(reference_explanation_types), 
															   unique_explanation_types, 
															   'llm_results/' + refined_model_name + '_' + domain_name + '_explanation_type_accuracy.png')

		cm_explanation_types = classification_report(list(found_questions_results['Explanation type'].astype(str)), 
										  list(reference_explanation_types), labels=unique_explanation_types)
		
		cm_confusion_explanation_str = 'Confusion matrix for explanation types is \n' + str(cm_explanation_types)
		print(cm_confusion_explanation_str)

		(f1_pred, precision_pred, recall_pred) = metaexplainer_utils.compute_f1(list(found_questions_references['Machine interpretation']),
															list(found_questions_results['Machine interpretation'].astype(str)))
		
		results_pred = '\nF1 on Machine interpretation ' + str(round(f1_pred*100, 2)) + '% Precision ' + str(round(precision_pred*100, 2)) + '% Recall ' + str(round(recall_pred*100,2)) + '%'
		print(results_pred)

		(f1_action, precision_action, recall_action) = metaexplainer_utils.compute_f1(list(found_questions_references['Action']), list(found_questions_results['Action'].astype(str)))
		
		results_action = '\nF1 on Action ' + str(round(f1_action*100, 2)) + '% Precision ' + str(round(precision_action*100, 2)) + '% Recall ' + str(round(recall_action*100,2))	+ '%'											
		print(results_action)

		(f1_likelihood, precision_likelihood, recall_likelihood) = metaexplainer_utils.compute_f1(list(found_questions_references['Target variable']),
															list(found_questions_results['Target variable'].astype(str)))
		
		results_likelihood = '\nF1 on Likelihood ' + str(round(f1_likelihood*100, 2)) + '% Precision ' + str(round(precision_likelihood*100, 2)) + '% Recall ' + str(round(recall_likelihood*100,2))	+ '%'											

		print(results_likelihood)

		error_str = '\nNon-matches between result and input ' + str(not_found) + '\nThese will be skipped.'
		#return label level F1 and F1s for other output fields - Machine Interpretation, Action and Likelihood
		print(error_str)

		with open(codeconstants.OUTPUT_FOLDER + '/llm_results/' + str(self.refined_model_name) + '_' + str(domain_name) + '_results.txt', 'w') as f:
			f.write(cm_confusion_explanation_str + '\nF1 scores on text fields: \n' 
				+ results_pred + results_action + results_likelihood + '\n\nErrors: ' + error_str)

	def run(self, mode):
		'''
		Caution: don't run train and test at same time
		'''
		if mode == 'train':
			self.set_base_model()
			self.set_datasets('Diabetes')
			self.train(self.train_dataset)
		elif mode == 'test':
			print('Running inference on test datatset')
			self.get_datasets('Diabetes')
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
	

	#llm_explanation_interpreter.run('train')
	#llm_explanation_interpreter.run('test')

	#if the compute metrics is called outside of test / train - then call get_datasets
 
	if llm_explanation_interpreter.test_dataset == None or llm_explanation_interpreter.train_dataset == None:
		#maybe the train doesn't have to be domain-specific
		llm_explanation_interpreter.get_datasets('Diabetes')

	llm_explanation_interpreter.compute_metrics('Diabetes')

