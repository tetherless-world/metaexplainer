from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
import Levenshtein


#visualization
import matplotlib.pyplot as plt
import seaborn as sns

import random


from nltk.tokenize import word_tokenize

import re

#data stack
import numpy as np
import pandas as pd

import os

import torch
import collections 
from pathlib import Path

import sys
sys.path.append('../')
from metaexplainercode import codeconstants

from pandas.testing import assert_frame_equal

def are_dfs_equal(df1, df2):
	try:
		assert_frame_equal(df1, df2)
		return True
	except:  # appeantly AssertionError doesn't catch all
		return False


def find_list_difference(l1, l2):
	'''
	When order doesn't matter convert l1 and l2 to sets and then return difference
	'''
	return list(set(l1) - set(l2))

def is_cat(column_dtype):
	'''
	In this approach string columns also get 1-hot encoded - but that is ok
	'''
	categorical_dtypes = ['object', 'category', 'bool']

	if column_dtype in categorical_dtypes:
		return True
	else:
		return False   

def drop_unnamed_cols(df):
	'''
	Return a dataframe without unnamed cols
	'''
	df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
	return df

def get_random_samples_in_list(passed_list, no_samples):
	rand_ranges = random.sample(range(len(passed_list)), no_samples)
	return [passed_list[i] for i in rand_ranges]

def find_cosine_similarity(s1, s2):
	'''
	return score for sklearn's cosine similarity
	'''
	comparisons = (s1, s2)
	#print(comparisons)

	tfidf_vectorizer = TfidfVectorizer()
	tfidf_matrix = tfidf_vectorizer.fit_transform(comparisons)

	result_cos = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
	return result_cos[0][1]

def print_list(list_n):
	'''
	Print contents of list on separate lines
	'''
	for val in list_n:
		print(val)

def write_list(list_n, file_name):
	with open(file_name, 'w') as f:
		for line in list_n:
			f.write(f"{line.encode('utf-8')}\n")


def read_list_from_file(file_name):
	f = open(file_name, 'r', encoding='utf-8')
	lines = f.read().splitlines()
	return lines

def read_delegate_parsed_instruction_file(file_name):
	'''
	read delegate instructions - which is dictionary separated by ---------
	'''
	f = open(file_name, 'r', encoding='utf-8')
	lines = f.read().splitlines()

	parses = []
	record = {}

	for line in lines:
		if line == '---------':
			parses.append(record)
			record = {}
		else:
			split_line = line.split(' : ')
			first_half = split_line[0].strip()
			second_half = split_line[1].strip()

			if first_half == 'Feature groups':
				second_half = eval(second_half)
			
			if first_half == 'Question' or (first_half == 'Machine interpretation') or (first_half == 'Explanation type'):
				record[split_line[0].strip()] = second_half
			elif first_half in record.keys():
				record[first_half].append(second_half)
			else:
				record[first_half] = [second_half]
	
	return parses

def get_subfolders_in_folder(folder_path):
	return [ f.path for f in os.scandir(folder_path) if f.is_dir() ]


def read_delegate_explainer_outputs(mode='generated', stage='delegate'):
	'''
	Return directories to read - since this code is used by delegate and synthesis 
	'''
	stage_dir = codeconstants.DELEGATE_FOLDER

	if stage=='synthesis':
		stage_dir = codeconstants.SYNTHESIS_FOLDER

	delegate_results_folder = stage_dir + '/results/'

	dirs_explainer_delegate = {}
	
	if mode == 'fine-tuned':
		delegate_results_folder = stage_dir + '/fine-tuned_results/'
	
	for dir_ep in os.listdir(delegate_results_folder):
		dir_ep_path = delegate_results_folder + '/' + dir_ep
		sub_dirs_dir = []

		sub_dirs = os.listdir(dir_ep_path)
		
		for subset_dir in sub_dirs:
			subset_dir_path = dir_ep_path + '/' + subset_dir
			
			if os.path.isdir(subset_dir_path):
				sub_dirs_dir.append(subset_dir_path)

		dirs_explainer_delegate[dir_ep_path] = sub_dirs_dir

	return dirs_explainer_delegate

def read_delegate_records_df(record_path):
    df = pd.read_csv(record_path, skiprows=1, header=None).T   # Read csv, and transpose
    df.columns = df.iloc[0]                                 # Set new column names
    df.drop(0,inplace=True)
    return df

def load_delegate_parses(domain_name, mode='generated', usable=True):
	parse_file = codeconstants.DELEGATE_FOLDER + domain_name + '_parsed_' + mode + '_delegate_instructions.txt'

	if not usable:
		parse_file = codeconstants.DELEGATE_FOLDER + '/unusable/' + domain_name + '_parsed_' + mode + '_delegate_instructions.txt'

	parses = read_delegate_parsed_instruction_file(parse_file)
	parses_df = pd.DataFrame(parses)

	return parses_df

def load_selected_explanation_types():
	loaded_explanations = [x.strip() for x in open(codeconstants.EXPLANATIONS_LOADED_FROM_EO, 'r').readlines()]
	return loaded_explanations

def load_dataset(domain_name):
	'''
	Dataset location is DATA_FOLDER + '/Domain_name' 
	'''
	dataset = pd.read_csv(codeconstants.DATA_FOLDER + '/' + domain_name + '/' + domain_name.lower() + '_val_corrected.csv')
	dataset = drop_unnamed_cols(dataset)
	return dataset

def load_column_names(domain_name):
	'''
	Return list of column names for dataset
	'''
	file_name = '_'.join(domain_name.lower().split(' '))
	domain_dataset = pd.read_csv(codeconstants.DATA_FOLDER + '/' + domain_name + '/' + file_name + '.csv')
	return list(domain_dataset.columns)

def get_multi_word_phrase_from_capitalized_string(capitalized_word):
	'''
	Generate a multi-word phrase from capitalized words
	E.g., DairyFreeMilk -> Dairy Free Milk
	'''
	subbed_string = re.sub('([A-Z])', r' \1', capitalized_word).strip()
	return subbed_string

def create_folder(path):
	Path(path).mkdir(parents=True, exist_ok=True)


def generate_acronyms_possibilities(list_of_conts):
	list_edited = {ele.replace(' ', '').lower():ele for ele in list_of_conts}
	list_expanded = {}
	acronyms_list = {}

	#based off of: https://stackoverflow.com/questions/4355201/creating-acronyms-in-python
	for ele_l in list_of_conts:
		if ' ' in ele_l:
			acronyms_list[''.join(w[0].upper() for w in ele_l.split(' '))] = ele_l
		else:
			acronyms_list[''.join(list(filter(str.isupper, ele_l)))] = ele_l

		if '_' in ele_l:
			list_expanded[ele_l.replace('_', ' ')] = ele_l
		else:
			list_expanded[re.sub('([A-Z])', r' \1', ele_l)] = ele_l
		
	return (list_edited, list_expanded, acronyms_list)

def check_if_label(field_key, labels):
	'''
	Check if field is in labels / columns
	Ignore case and spaces
	Need to see how to handle abbreviations
	'''
	field_key = field_key.replace(' ','').replace('_','').lower()
	(labels_edited, labels_expanded, acronyms_labels) = generate_acronyms_possibilities(labels)

	if (field_key in labels_edited.keys()):
		return (True, labels_edited[field_key])
	elif (field_key in labels_expanded.keys()):
		return (True, labels_expanded[field_key])
	elif (field_key.upper() in acronyms_labels.keys()):
		return (True, acronyms_labels[field_key.upper()])
	return (False, '')

def find_labels_in_sentence(sentence, domain_name):
	tokenized_sent = word_tokenize(sentence)
	col_names_domain = load_column_names(domain_name)
	#print(col_names_domain)
	cols_matched = []
	
	for token in tokenized_sent:
		matched_or_not, matched_col = check_if_label(token, col_names_domain)

		if matched_or_not:
			cols_matched.append(matched_col)

	return cols_matched

def is_valid_number(string):
	try:
		val = float(string)
		return True
	except Exception:
		return False

'''
Result generation functions
'''
def generate_confusion_matrix_and_visualize(y_true, y_pred, labels, output_file_path): 
	cm = confusion_matrix(y_true, y_pred, labels=labels)
	cm_array_df = pd.DataFrame(cm, index=labels, columns=labels)
	#print('Confusion matrix ', cm_array_df)
	cm_heatmap = sns.heatmap(cm_array_df, annot=True, annot_kws={"size": 12})
	#based on https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
	fig = cm_heatmap.get_figure()
	fig.savefig(codeconstants.OUTPUT_FOLDER + '/' + output_file_path, bbox_inches='tight')
	fig.show()

def compute_f1(reference_strs, result_strs):
	'''
	Assume you have a list of strings and return F1s for each of those 
	Inspired by: https://cohere.com/blog/evaluating-llm-outputs
	'''
	

	gold_toks = sum([sent.split(' ') for sent in reference_strs], [])
	#print('Sample ', gold_toks[:20])
	pred_toks = sum([sent_res.split(' ') for sent_res in result_strs], [])
	#print('Sample pred ', pred_toks[:20])

	common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
	num_same = sum(common.values())

	exact_match = num_same / (len(gold_toks) + len(pred_toks))

	if len(gold_toks) == 0 or len(pred_toks) == 0:
		#if either is no answer, then F1 is 1 if they agree, 0 otherwise
		return int(gold_toks == pred_toks)
	if num_same == 0:
		return 0
	precision = 1.0 * num_same/len(pred_toks)
	recall = 1.0*num_same/len(gold_toks)
	f1 = (2*precision*recall)/(precision + recall)
	return (f1, precision, recall, exact_match)

def compute_f1_levenshtein(reference_strs, result_strs, threshold=0.6):
	'''
	Assume you have a list of strings and return F1s for each of those 
	Inspired by: https://cohere.com/blog/evaluating-llm-outputs
	'''
	tp, fp, fn = 0, 0, 0

	for label, pred in zip(reference_strs, result_strs):
		if not (len(label) == 0 and len(pred) == 0):
			similarity_score = 1 - Levenshtein.distance(label, pred) / max(len(label), len(pred))
			if similarity_score >= threshold:
				tp += 1
			else:
				fp += 1

	fn = len(reference_strs) - tp

	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	f1_score = 2 * (precision * recall) / (precision + recall)

	return f1_score, precision, recall

'''
Loading records from Llama outputs
'''
def extract_key_value_from_string(response_str, find_key):
		extract_str = ''

		extracted_val = re.split('(' + find_key + '):?[\n=]?', response_str)[1:3]
		#print(type(extracted_val), extracted_val, len(extracted_val))
		
		if len(extracted_val) > 1:
			part_to_split = extracted_val[1]

			#fixer for getting more responses, because sometimes \n is encoded as \\n
			splitter = '\\n'

			if not splitter in part_to_split:
				if '\n' in part_to_split:
					splitter = '\n'
				elif '=' in part_to_split:
					splitter = '='
				print('Changing split', splitter)

			#print(part_to_split, splitter)
			parts = part_to_split.split(splitter)
			#print(parts)
			
			#print(parts)

			for str_val in parts:
				str_edited = str_val.strip()

				if str_edited != '':
					#find first non empty string and set that!
					extract_str = str_edited
					break
		
		#print('Extracted', extract_str)
		return extract_str

def process_decompose_llm_result(model_name, domain_name, mode, output_mode='dictionary'):
		'''
		Need to remove instruction from the responses and retain the top-1 alone
		In the future, will use domain_name - for now it is redundant		
		'''

		result_file_name = codeconstants.OUTPUT_FOLDER + '/llm_results/' + model_name + '_' + mode + '_outputs.txt'

		loaded_explanations = [x.strip() for x in open(codeconstants.EXPLANATIONS_LOADED_FROM_EO, 'r').readlines()]

		keys = ['Explanation type', 'Machine interpretation', 'Action', 'Target variable']

		print('Output mode ', output_mode)
		
		#loads content in decoded form - while writing or returning it back need to use encode.
		read_content = read_list_from_file(result_file_name)
		

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
				val = extract_key_value_from_string(response, field)

				if field == 'Explanation type':
					matched_explanation = list(filter(lambda x: x.lower() in val.lower(), loaded_explanations))

					if len(matched_explanation) > 0:
						val = matched_explanation[0]

					#print(str(val))

				val_keys[field] = val


			
			val_keys['Question'] = extract_key_value_from_string(str(rest_of_string), 'User')


			# if val_keys['Question'] == 'Why focus on the BMI range of 18-25 for predicting Diabetes instead of 26-30?':
			# 	print(response)
			# 	print(val_keys)
			
			result_dict.append(val_keys)
			#print(val_keys)
		
		result_datastructure = {}

		if output_mode == 'list':
			result_datastructure = []

		#print('Length of results before creating Question: rest dictionary', len(result_dict))
		already_seen = []
		duplicates = []

		for record in result_dict:
			question = record['Question']

			if question in already_seen:
				#print('Duplicate question ', question)
				duplicates.append(question)

			#result_datastructure[question] = {}

			del record['Question']

			already_seen.append(question)

			if output_mode == 'dictionary':
				result_datastructure[question] = record
			elif output_mode == 'list':
				quest_dict = {'Question': question}
				quest_dict.update(record)
				result_datastructure.append(quest_dict)
		
		print('Length of results ', len(result_datastructure), 'and duplicates removed ', len(duplicates))
		return result_datastructure
	
