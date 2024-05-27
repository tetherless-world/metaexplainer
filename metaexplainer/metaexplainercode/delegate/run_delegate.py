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

import sys
sys.path.append('../')
from metaexplainercode import codeconstants

from metaexplainercode import metaexplainer_utils
from metaexplainercode.delegate.train_models.run_model_tabular import *

from metaexplainercode.delegate.run_explainers import filter_records

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

	explanation_instance_for_record = explanation_methods[explanation_methods['Explanation Type'] == sample_record['Explanation type']]['Instances']
	action_list = []
	feature_groups = sample_record['Feature groups']

	if 'Action' in sample_record:
		action_list = sample_record['Action']

	subsets = run_explainers.filter_records(dataset, feature_groups, action_list)
	print(explanation_instance_for_record)

	for subset_i in range(len(subsets)):
		print('Filtered subsets for feature group ', feature_groups[subset_i], 'is \n', subsets[subset_i])

	

	#need to extract and call run explainers based on feature selectors


def get_corresponding_explainer():
	'''
	This could be just reading from delegate output folder 
	'''
	pass

def run_explainer(feature_groups, actions, explainer_method):
	'''
	Call corresponding explainer with feature group filters and actions 
	'''
	pass

if __name__=='__main__':
	domain_name = 'Diabetes'
	domain_dataset = metaexplainer_utils.load_dataset(domain_name)

	parse = retrieve_sample_decompose_passes(domain_dataset, domain_name)

	# explainer_method = get_corresponding_explainer()
	# method_results = run_explainer(parse['feature_groups'], parse['actions'], explainer_method)