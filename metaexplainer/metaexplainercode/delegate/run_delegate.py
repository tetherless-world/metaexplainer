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

def retrieve_sample_decompose_passes(mode='fine-tuned'):
	'''
	Read from delegate output folder, if not running the method in real-time
	'''
	pass

def get_domain_model(domain_name):
	'''
	Train model if not present 
	'''
	domain_dataset = metaexplainer_utils.load_dataset(domain_name)
	
	model_save_path = codeconstants.DELEGATE_SAVED_MODELS_FOLDER + domain_name + '/model.pkl'

	if not os.path.isfile(model_save_path):
		(X, Y) = generate_X_Y(domain_dataset, 'Outcome')
		x_train, x_test, y_train, y_test = generate_train_test_split(domain_dataset, 'Outcome', 0.30)

		transformations = transform_data(domain_dataset, [], ['Outcome'])

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

		
		transform_save_path = codeconstants.DELEGATE_SAVED_MODELS_FOLDER + domain_name + '/transformations.pkl'
		results = mod_classification_report_save
		
		joblib.dump(model_to_save, model_save_path)
		pickle.dump(mod_classification_report_save, open(codeconstants.DELEGATE_SAVED_MODELS_FOLDER + domain_name + '/results.pkl', "wb"))
		joblib.dump(transformations, transform_save_path)
		print('Saved model', best_model_name,' and transformations.')
	else:
		model_to_save = joblib.load(model_save_path)
		transformations = joblib.load(codeconstants.DELEGATE_SAVED_MODELS_FOLDER + domain_name + '/transformations.pkl')
		results = pickle.load(open(codeconstants.DELEGATE_SAVED_MODELS_FOLDER + domain_name + '/results.pkl', 'rb'))
		print('Retrieved model and results.')


	return (model_to_save, transformations, results)


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
	get_domain_model(domain_name)

	# parse = retrieve_sample_decompose_passes()
	# explainer_method = get_corresponding_explainer()
	# method_results = run_explainer(parse['feature_groups'], parse['actions'], explainer_method)