'''
Main entry-point for delegate; so given a machine interpretation; the steps are:
- Call parse machine interpretation (parse_machine_interpretation)
- Get the corresponding explainer method for explanation type (extract_explainer_methods)
- Run the explainer method (run_explainers)
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import joblib

import sys
sys.path.append('../')
from metaexplainercode import codeconstants

from metaexplainercode import metaexplainer_utils
from metaexplainercode.delegate.train_models.run_model_tabular import *

def retrieve_sample_decompose_passes(mode='fine-tuned'):
	'''
	Read from delegate output folder, if not running the method in real-time
	'''
	pass

def get_model(domain_name):
	'''
	Train model if not present 
	'''
	pima_diabetes = metaexplainer_utils.load_dataset(domain_name)

	(X, Y) = generate_X_Y(pima_diabetes, 'Outcome')
	x_train, x_test, y_train, y_test = generate_train_test_split(pima_diabetes, 'Outcome', 0.30)

	transformations = transform_data(pima_diabetes, [], ['Outcome'])

	models = get_models()

	
	evaluate_model(models, transformations, x_train, y_train)
	model_output = {}

	for mod_num in models.keys():
		model = models[mod_num]

		(model, mod_classification_report) = fit_and_predict_model(mod_num, transformations, model, x_train, y_train, x_test, y_test)

		#this is where you get the model 

		model_output[mod_num] = (model, mod_classification_report)

	best_model = get_best_model(model_output, 0)
	
	print('Stats on test dataset for best model ', best_model[0])	

	#getting best output here - objective is to retrain
	#print(model_output_print[1])
	(model_to_save, mod_classification_report_save) = fit_and_predict_model(mod_num, transformations, best_model, X, Y, x_test, y_test, save_model=True)
	print('Stats on entire dataset for the same best model ',mod_classification_report_save)
	best_model_name = mod_classification_report_save['model']

	#create model folders and for domain
	metaexplainer_utils.create_folder(codeconstants.DELEGATE_SAVED_MODELS_FOLDER)
	metaexplainer_utils.create_folder(codeconstants.DELEGATE_SAVED_MODELS_FOLDER + domain_name)

	model_save_path = codeconstants.DELEGATE_SAVED_MODELS_FOLDER + domain_name + '/' + best_model_name + '.pkl'
	transform_save_path = codeconstants.DELEGATE_SAVED_MODELS_FOLDER + domain_name + '/transformations.pkl'

	if not os.path.isfile(model_save_path):
		joblib.dump(model_to_save, model_save_path)
		joblib.dump(transformations, transform_save_path)

	print('Saved model', best_model_name,' and transformations.')

	return (best_model[0], transformations, x_train, x_test, y_train, y_test)


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
	get_model(domain_name)

	# parse = retrieve_sample_decompose_passes()
	# explainer_method = get_corresponding_explainer()
	# method_results = run_explainer(parse['feature_groups'], parse['actions'], explainer_method)