'''
Main entry-point for delegate; so given a machine interpretation; the steps are:
- Call parse machine interpretation (parse_machine_interpretation)
- Get the corresponding explainer method for explanation type (extract_explainer_methods)
- Run the explainer method (run_explainers)
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


import time 
import itertools

import sys
sys.path.append('../')
from metaexplainercode import codeconstants

from metaexplainercode import metaexplainer_utils

from metaexplainercode import ontology_utils

from metaexplainercode.delegate.train_models.run_model_tabular import *

from metaexplainercode.delegate.run_explainers import filter_records
from metaexplainercode.delegate.run_explainers import TabularExplainers
from metaexplainercode.delegate.evaluate_explainers import EvaluateExplainer

import random

def retrieve_sample_decompose_passes(sample_record, dataset):
	'''
	Read from delegate output folder, if not running the method in real-time
	'''
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

def get_metrics(explainer_method):
	'''
	Evaluate ouput of explainers by modality and metrics 
	Need to get modality for explainer and then metric and then pass results or the explainer itself 
	'''
	explanation_methods = pd.read_csv(codeconstants.DELEGATE_FOLDER + '/explanation_type_methods.csv')
	modality_metrics = pd.read_csv(codeconstants.DELEGATE_FOLDER + '/explanation_modality_metrics.csv')
	metrics_modality = []

	
	for explainer in explainer_method:
		explainer_instance = explainer.replace('run_', '').upper()

		print(explainer_method)
			
		modality_explainer = list(set(explanation_methods[explanation_methods['Instances'].str.strip().str.match(explainer_instance.strip(),case=False)]['Modality']))[0]
		#mapping between explainer and modality is typically 1:1
		metrics_modality = list(modality_metrics[modality_metrics['Modality'].str.strip().str.match(modality_explainer.strip(),case=False)]['Metric'])
		print(modality_explainer, metrics_modality)

	return metrics_modality

def run_and_evaluate_explainer(domain_dataset, model_details, feature_subsets, actions, explainer_method):
	'''
	Call corresponding explainer with feature group filters and actions 
	Need to implement this 
	'''
	tabular_explainer = TabularExplainers(model_details['model'], model_details['transformations'], model_details['results'], domain_dataset)
	evaluate_explainer = EvaluateExplainer(domain_dataset, model_details['model'], model_details['transformations'])
	implemented_evaluations = [attr for attr in dir(EvaluateExplainer) if not attr.startswith('__')]

	results = []
	evaluations = []

	metrics = get_metrics(explainer_method)

	if explainer_method != []:
		explainer = explainer_method[0]

		for feature_subset in feature_subsets:
			(result, explainer_obj) = getattr(tabular_explainer, explainer)(passed_dataset=feature_subset)
			evaluation_feature = []

			if len(metrics) > 0:
				samples = result

				if type(result) == dict:
					#It is a counterfactual
					samples = result['Changed']

				combinations_funcs = ['evaluate_' + '_and_'.join(list(comb)).lower().replace(' ','_') for i in range(1, len(metrics) + 1) 
						  for comb in list(itertools.permutations(metrics, i))]
				intersection_funcs = list(set(combinations_funcs).intersection(set(implemented_evaluations)))
				print('Debugging ', combinations_funcs)

				for eval_func in intersection_funcs:
					evaluation_feature += getattr(evaluate_explainer, eval_func)(explainer_obj, passed_dataset=feature_subset, results=samples)

			evaluations.append(evaluation_feature)
			results.append(result)
			#need to persist evaluations too
	#print(evaluations)
	return (results, evaluations)

def save_results(sample_record, feature_subsets, results, evaluations, explainer, explanation_type):
	metaexplainer_utils.create_folder(codeconstants.DELEGATE_RESULTS_FOLDER)

	if len(explainer) and len(results) > 0:
		explanation_substring = explanation_type.replace(' ', '') + '_' + explainer[0]
		result_folder = codeconstants.DELEGATE_RESULTS_FOLDER + '/' + explanation_substring + '_' + str(int(time.time()))
		metaexplainer_utils.create_folder(result_folder)

		#this will directly be used for prompt in synthesis
		sample_record.T.to_csv(result_folder + '/record.csv')

		res_ctr = 0
		print('Debugging ', len(results))

		for result in results:
			result_folder_curr = result_folder + '/' + str(res_ctr)

			metaexplainer_utils.create_folder(result_folder_curr)

			if len(evaluations) > 0 and len(evaluations[res_ctr]) > 0:
				#print(evaluations[res_ctr])
				pd.DataFrame(evaluations[res_ctr]).to_csv(result_folder_curr + '/Evaluations.csv')

			if explanation_type == 'Counterfactual Explanation':
				result['Changed'].to_csv(result_folder_curr + '/Results.csv')
				result['Queries'].to_csv(result_folder_curr + '/Original.csv')
			else:
				result.to_csv(result_folder_curr + '/Results.csv')
		
			feature_subsets[res_ctr].to_csv(result_folder_curr + '/Subset.csv')
			res_ctr += 1

		print('Created result folder ', result_folder, ' and added files for ', len(feature_subsets), ' with results from explainer ', explainer, ' for explanation ', explanation_type)
	else:
		print('Skipped for ', explanation_type)


	#matched_metric = modality_metrics[modality_metrics['Modality'] == modality_explainer]['Metric']


if __name__=='__main__':
	domain_name = 'Diabetes'
	domain_dataset = metaexplainer_utils.load_dataset(domain_name)

	(training_model, transformations, method_results) = get_domain_model(domain_name)
	mode = 'generated'

	model_details = {'model': training_model,
				  'transformations': transformations,
				  'results': method_results}

	decompose_parses = metaexplainer_utils.load_delegate_parses(domain_name, mode=mode)

	len_parses = len(decompose_parses)

	print('Length of parses to run ', len_parses)

	for i in range(0, len_parses + 1):
		sample_record = decompose_parses.iloc[i]
		
		#need to make this run in a loop to run across all parses 
		(subsets, action_list, explainer_method, explanation_type) = retrieve_sample_decompose_passes(sample_record, domain_dataset)

		# explainer_method = get_corresponding_explainer()
		(method_results, evaluations) = run_and_evaluate_explainer(domain_dataset, model_details, subsets, action_list, explainer_method)

		save_results(sample_record, subsets, method_results, evaluations, explainer_method, explanation_type)

		print('Finished delegate for record ', str(i), ' of ', str(len_parses))

