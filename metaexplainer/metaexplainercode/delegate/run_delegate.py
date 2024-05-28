'''
Main entry-point for delegate; so given a machine interpretation; the steps are:
- Call parse machine interpretation (parse_machine_interpretation)
- Get the corresponding explainer method for explanation type (extract_explainer_methods)
- Run the explainer method (run_explainers)
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import pickle

import joblib
import time 

import sys
sys.path.append('../')
from metaexplainercode import codeconstants

from metaexplainercode import metaexplainer_utils
from metaexplainercode.delegate.train_models.run_model_tabular import *

from metaexplainercode.delegate.run_explainers import filter_records
from metaexplainercode.delegate.run_explainers import TabularExplainers

import random

def retrieve_sample_decompose_passes(dataset, domain_name, mode='fine-tuned'):
	'''
	Read from delegate output folder, if not running the method in real-time
	'''
	parse_file = codeconstants.DELEGATE_FOLDER + domain_name + '_parsed_' + mode + '_delegate_instructions.txt'

	parses = metaexplainer_utils.read_delegate_parsed_instruction_file(parse_file)
	parses_df = pd.DataFrame(parses)

	sample_record = parses_df.iloc[random.randrange(0, len(parses))]

	print(sample_record)

	explanation_methods = pd.read_csv(codeconstants.DELEGATE_FOLDER + '/explanation_type_methods.csv')
	implemented_methods = [attr for attr in dir(TabularExplainers) if not attr.startswith('__')]
	explanation_instance_for_record = explanation_methods[explanation_methods['Explanation Type'] == sample_record['Explanation type']]['Instances']
	matched_method = [im_method for im_method in implemented_methods for instance_method in explanation_instance_for_record if instance_method.lower() in im_method]

	action_list = []
	feature_groups = sample_record['Feature groups']

	if 'Action' in sample_record:
		action_list = sample_record['Action']

	subsets = filter_records(dataset, feature_groups, action_list)
	print(explanation_instance_for_record, 'Match found ', matched_method)

	for subset_i in range(len(subsets)):
		print('Filtered subsets for feature group ', feature_groups[subset_i], 'is \n', subsets[subset_i])

	return (subsets, action_list, matched_method, sample_record['Explanation type'])

	#need to extract and call run explainers based on feature selectors

def run_explainer(domain_name, feature_subsets, actions, explainer_method, explanation_type):
	'''
	Call corresponding explainer with feature group filters and actions 
	Need to implement this 
	'''
	tabular_explainer = TabularExplainers(domain_name)
	results = []

	if explainer_method != []:
		explainer = explainer_method[0]

		for feature_subset in feature_subsets:
			results.append(getattr(tabular_explainer, explainer)(passed_dataset=feature_subset))
	
	return results

def save_results(feature_subsets, results, explainer, explanation_type):
	metaexplainer_utils.create_folder(codeconstants.DELEGATE_RESULTS_FOLDER)

	if len(explainer) and len(results) > 0:
		explanation_substring = explanation_type.replace(' ', '') + '_' + explainer[0]
		result_folder = codeconstants.DELEGATE_RESULTS_FOLDER + '/' + explanation_substring + '_' + str(int(time.time()))
		metaexplainer_utils.create_folder(result_folder)

		res_ctr = 0

		for result in results:
			result_folder_curr = result_folder + '/' + str(res_ctr)

			metaexplainer_utils.create_folder(result_folder_curr)

			if explanation_type == 'Counterfactual Explanation':
				result['Changed'].to_csv(result_folder_curr + '/' + explanation_substring + '_Results.csv')
				result['Queries'].to_csv(result_folder_curr + '/' + explanation_substring + '_Original.csv')
			else:
				result.to_csv(result_folder_curr + '/' + explanation_substring + '_Results.csv')
		
			feature_subsets[res_ctr].to_csv(result_folder_curr + '/' + explanation_substring + '_Subset.csv')

		print('Created result folder ', result_folder, ' and added files for ', len(feature_subsets), ' with results from explainer ', explainer, ' for explanation ', explanation_type)
	else:
		print('Skipped for ', explanation_type)
		

if __name__=='__main__':
	domain_name = 'Diabetes'
	domain_dataset = metaexplainer_utils.load_dataset(domain_name)

	(subsets, action_list, explainer_method, explanation_type) = retrieve_sample_decompose_passes(domain_dataset, domain_name, mode='generated')

	# explainer_method = get_corresponding_explainer()
	method_results = run_explainer(domain_name, subsets, action_list, explainer_method, explanation_type)

	save_results(subsets, method_results, explainer_method, explanation_type)

