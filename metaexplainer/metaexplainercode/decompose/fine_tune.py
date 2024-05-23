import torch

from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	BitsAndBytesConfig,
	TrainingArguments,
	pipeline
)

from accelerate import Accelerator

from jinja2 import Template
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

		accelerator = Accelerator() #when using cpu: cpu=True

		device = accelerator.device
		self.device = device

		#self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		#defining variables that get set through setter functions later on
		self.batch_size = 15
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
			train_dataset_path_check = Path(train_dataset_path).resolve(strict=True)
			test_dataset_path_check = Path(test_dataset_path).resolve(strict=True)
			self.train_dataset = load_dataset('json', data_files= train_dataset_path)
			self.test_dataset = load_dataset('json', data_files= test_dataset_path)
			print('Dataset split found and datasets loaded. Train: ', len(self.train_dataset), 'And test: ', len(self.test_dataset))
		except FileNotFoundError:
			try:
				my_abs_path = Path(dataset_file_path).resolve(strict=True)
			except FileNotFoundError:
				self.process_jsonl_file(dataset_file_path, pd.read_csv(domain_dir_path + '/finetune_questions.csv'))
			
				dataset = load_dataset('json', data_files= dataset_file_path)

				#https://huggingface.co/docs/datasets/v1.8.0/package_reference/main_classes.html#datasets.Dataset.train_test_split
				splits = dataset['train'].train_test_split(test_size=0.2)
				self.train_dataset = splits['train']
				self.test_dataset = splits['test']
				self.dump_jsonl(self.train_dataset, train_dataset_path)
				self.dump_jsonl(self.test_dataset, test_dataset_path)
				print('Dataset split generated and datasets loaded. Train: ', len(self.train_dataset),
				'And test: ', len(self.test_dataset))
	
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
			#device_map="{"": 0}",
			#device_map = "auto"
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
			output_dir= codeconstants.OUTPUT_FOLDER + '/llm_results/' + self.refined_model_name + '_decompose_results_modified',
			num_train_epochs=12,
			per_device_train_batch_size= self.batch_size,
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

		train_source = self.train_dataset

		if 'train' in self.train_dataset.keys():
			train_source = self.train_dataset['train']

		# Trainer
		fine_tuning = SFTTrainer(
			model=self.base_model,
			train_dataset = train_source,
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
		fine_tuning.model.save_pretrained(codeconstants.OUTPUT_FOLDER + '/llm_results/' + self.refined_model_name + '_decompose_refined_model')

		# Save Model - pipeline way
		## change this to include the refined model name
		#fine_tuning.save_model(codeconstants.OUTPUT_FOLDER + '/llm_results/' + self.refined_model_name + '_decompose_refined_model')
		#fine_tuning.model.config.save_pretrained(codeconstants.OUTPUT_FOLDER + '/llm_results/' + self.refined_model_name + '_decompose_refined_model')

		print('Saved the trained model ')


	def get_base_model(self):
		base_model = AutoModelForCausalLM.from_pretrained(
				self.base_model_name,
				torch_dtype=torch.float16,
				#device_map="auto",
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
			peft_config = codeconstants.OUTPUT_FOLDER + '/llm_results/' + self.refined_model_name + '_decompose_refined_model'

			device_map = {
				"base_model": self.device,
			}


			# Create the PeftModel instance
			refined_model = PeftModel.from_pretrained(self.get_base_model(), peft_config,
														adapter_name="sft", device_map=device_map)

			refined_model.merge_adapter()

			# refined_model = AutoModelForCausalLM.from_pretrained(gdrive_path + 'llama')
			#use multiple GPUs if available
			#refined_model = torch.nn.DataParallel(refined_model)
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

		
		accelerator = Accelerator()
		model = accelerator.prepare(self.refined_model)
		is_main_process = accelerator.is_main_process
		
		with torch.no_grad():
			for dataset_record in passed_dataset['train']:
				#print('Dataset record ', dataset_record)
				prompt = self.format_instruction_record(dataset_record)
				inputs.append(prompt)

			print('# of Prompts generated', len(inputs), ' and a sample is \n', inputs[0])

			generate_ids = []
			generate_batch_size = 4 # you can also say generate_batch_size = self.batch_size

			#pseudo-batching because otherwise it crashes during inference
			#this batch thing might be need to be based on the dataset size, for now - leave as-is
			for i in range(0, len(inputs), generate_batch_size):
				batch_inputs = inputs[i: i + generate_batch_size]
				print(' Handling batch ', str(i), 'to ', str(i + generate_batch_size))
				#based on https://discuss.huggingface.co/t/llama2-pad-token-for-batched-inference/48020
				tok_inputs = self.tokenizer(batch_inputs, return_tensors="pt", padding=True).to(self.device)
				#print('Debug ', self.device)
				generate_ids += model.generate(**tok_inputs, max_length=500, do_sample=True, top_p=0.9).to(self.device)


			print('Len of generate IDs ', len(generate_ids))
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

	

	def post_process_results(self, mode='test'):
		'''
		Need to remove instruction from the responses and retain the top-1 alone
		This would need to be called at the compute-F1
		'''
		result_dictionary = metaexplainer_utils.process_decompose_llm_result(self.refined_model_name, 'Diabetes', mode)
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

		print(' Length of inputs ', len(input_dict))
		return input_dict
	
	def define_result_f1_record(self, field, f1, precision, recall):
		f1_template = '''
			\n{{field}}, F1: {{ (f1*100) | round(2) }}%, Precision: {{ (precision*100) | round(2) }}% and Recall: {{ (recall*100) | round(2) }}%
		'''
		f1_template = Template(f1_template)
		return f1_template.render({'field': field,
							 'f1': f1,
							 'precision': precision,
							 'recall': recall
							 })
	
	def compute_metrics(self, domain_name, mode='test'):
		'''
		Get the results in a dictionary and then use the input to find outputs based on input
		'''
		#Call get dataset if files exist
		results = self.post_process_results(mode)
		inputs = self.post_process_input(mode)

		#Defining hash key retrievers
		not_found = []
		found_questions_results = []
		found_questions_references = []
		question_comparisons = list(results.keys())

		for result_question in results.keys():
			#trying to find non matches 
			if type(result_question) != str:
				result_question = result_question.decode('utf-8')

			edited_result_question = result_question.replace('\\', '')

			if not edited_result_question  in inputs.keys():
				not_found.append(result_question)
			else:
				found_questions_results.append(results[result_question])
				found_questions_references.append(inputs[edited_result_question])

		#converting the results and references to pandas since it is easier to load
		found_questions_results = pd.DataFrame(found_questions_results)
		found_questions_references = pd.DataFrame(found_questions_references)
		
		print('Computing accuracy by explanation type')
		reference_explanation_types = found_questions_references['Explanation type']
		results_explanation_types = found_questions_results['Explanation type'].astype(str)

		unique_explanation_types = list(reference_explanation_types.unique())
		print('Labels for explanation types are ', unique_explanation_types)
		unique_explanation_types_results = list(results_explanation_types.unique())
		print('Unique explanation types - to see the variety of answers ', unique_explanation_types_results)

		#defining comparison lists
		references_predicates = list(found_questions_references['Machine interpretation'])
		results_predicates = list(found_questions_results['Machine interpretation'].astype(str))

		references_actions = list(found_questions_references['Action'])
		results_actions = list(found_questions_results['Action'].astype(str))

		references_likelihoods = list(found_questions_references['Target variable'])
		results_likelihoods = list(found_questions_results['Target variable'].astype(str))

		
		
		metaexplainer_utils.generate_confusion_matrix_and_visualize(list(reference_explanation_types),
															  list(results_explanation_types), 
															   unique_explanation_types, 
															   'llm_results/' + refined_model_name + '_' + domain_name + '_' + mode + '_explanation_type_accuracy.png')

		cm_explanation_types = classification_report(list(reference_explanation_types), 
											   list(results_explanation_types), 
											   labels=unique_explanation_types)
		
		cm_confusion_explanation_str = '---Confusion matrix for explanation types--- \n' + str(cm_explanation_types)
		print(cm_confusion_explanation_str)

		print('---F1 on Exact Match---')
		(f1_pred, precision_pred, recall_pred, exact_match_pred) = metaexplainer_utils.compute_f1(references_predicates, results_predicates)
		
		results_pred_str = self.define_result_f1_record('Machine interpretation', f1_pred, precision_pred, recall_pred)
		print(results_pred_str)

		(f1_action, precision_action, recall_action, exact_match_action) = metaexplainer_utils.compute_f1(references_actions, results_actions)
		
		results_action_str = self.define_result_f1_record('Action', f1_action, precision_action, recall_action)
		print(results_action_str)

		(f1_likelihood, precision_likelihood, recall_likelihood, exact_match_likelihood) = metaexplainer_utils.compute_f1(references_likelihoods, results_likelihoods)
		
		results_likelihood_str = self.define_result_f1_record('Likelihood', f1_likelihood, precision_likelihood, recall_likelihood)											
		print(results_likelihood_str)
		#F1s on Levenshtein

		print('---F1 on Levenshtein distances---')
		(f1_pred_lev, precision_pred_lev, recall_pred_lev) = metaexplainer_utils.compute_f1_levenshtein(references_predicates, results_predicates)
		
		results_lev_pred_str = self.define_result_f1_record('Machine interpretation', f1_pred_lev, precision_pred_lev, recall_pred_lev)
		print(results_lev_pred_str)

		(f1_action_lev, precision_action_lev, recall_action_lev) = metaexplainer_utils.compute_f1_levenshtein(references_actions, results_actions)
		
		results_lev_action_str = self.define_result_f1_record('Action', f1_action_lev, precision_action_lev, recall_action_lev)
		print(results_lev_action_str)

		(f1_likelihood_lev, precision_likelihood_lev, recall_likelihood_lev) = metaexplainer_utils.compute_f1_levenshtein(references_likelihoods, results_likelihoods)
		
		results_lev_likelihood_str = self.define_result_f1_record('Likelihood', f1_likelihood_lev, precision_likelihood_lev, recall_likelihood_lev)											
		print(results_lev_likelihood_str)

		print('---Exact match ratios---')
		results_exact_match_pred_str = '\nMachine interpretation, Exact match: ' + str(round(100*exact_match_pred, 2)) +'%'
		print(results_exact_match_pred_str)

		results_exact_match_action_str = '\nAction, Exact match: ' + str(round(100*exact_match_action, 2)) +'%'
		print(results_exact_match_action_str)

		results_exact_match_likelihood_str = '\nLikelihood, Exact match: ' + str(round(100*exact_match_likelihood, 2)) +'%'
		print(results_exact_match_likelihood_str)
		

		error_str = '\nNon-matches between result and input ' + str(not_found) + '\nThese will be skipped.'
		#return label level F1 and F1s for other output fields - Machine Interpretation, Action and Likelihood
		print(error_str)

		with open(codeconstants.OUTPUT_FOLDER + '/llm_results/' + str(self.refined_model_name) + '_' + str(domain_name) + '_' +
			mode + '_results.txt', 'w') as f:
			f.write(cm_confusion_explanation_str + '\nF1 Exact Match scores on text fields: \n' 
				+ results_pred_str + results_action_str + results_likelihood_str + 
				'\n---F1 Levenshtein scores on text fields---\n'
				+ results_lev_pred_str + results_lev_action_str + results_lev_likelihood_str +
				'\n---Exact match on text fields---\n'
				+ results_exact_match_pred_str + results_exact_match_action_str + results_exact_match_likelihood_str + 
				'\n\n---Errors---' + error_str)

	def run(self, mode, infer_mode='test'):
		'''
		Caution: don't run train and test at same time
		'''
		if mode == 'train':
			self.set_base_model()
			#this shouldn't be called for each model, if there don't call this!
			self.set_datasets('Diabetes')
			self.train(self.train_dataset)
		elif mode == 'test':
			print('Running inference on, ',infer_mode,' datatset')
			self.get_datasets('Diabetes')
			self.set_refined_model()

			if infer_mode == 'train':
				self.inference(self.train_dataset, mode='train')
			else:
				self.inference(self.test_dataset)




if __name__== "__main__":
	# Model and tokenizer names
	base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
	refined_model_name = "llama-3-8b-charis-explanation" #You can give it your own name

	#LLama2

	# base_model_name = 'NousResearch/Nous-Hermes-Llama2-13b'
	# refined_model_name = "llama-2-13b-charis-explanation" #You can give it your own name

	# #IBM Granite
	# base_model_name = 'ibm-granite/granite-8b-code-instruct'
	# refined_model_name = "ibm-granite-8b-charis-explanation" #You can give it your own name

	#defining variables necessary for instantiation
	llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, use_auth_token=True)
	llama_tokenizer.pad_token = llama_tokenizer.eos_token
	llama_tokenizer.padding_side = "left"  # Fix for fp16

	print('Tokenizer setup ')
	
	#instantiating class 
	llm_explanation_interpreter = LLM_ExplanationInterpretor(llama_tokenizer, base_model_name, refined_model_name)
	

	#llm_explanation_interpreter.run('train')
	#llm_explanation_interpreter.run('test', infer_mode='train')
	#to run inference on test
	#llm_explanation_interpreter.run('test')

	##if the compute metrics is called outside of test / train - then call get_datasets
 
	if llm_explanation_interpreter.test_dataset == None or llm_explanation_interpreter.train_dataset == None:
		#maybe the train doesn't have to be domain-specific
		llm_explanation_interpreter.get_datasets('Diabetes')

	llm_explanation_interpreter.compute_metrics('Diabetes', mode='train')

