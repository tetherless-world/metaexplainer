import pandas as pd

import sys
sys.path.append('../')
from metaexplainercode import codeconstants

from metaexplainercode import metaexplainer_utils
from train_models.run_model import *


# Importing shap KernelExplainer (aix360 style)
from aix360.algorithms.shap import KernelExplainer

# the following import is required for access to shap plotting functions and datasets
import shap

def run_shap(model, X_train, X_test, single_instance=True):
	shapexplainer = KernelExplainer(model.predict_proba, X_train)
	print(type(shapexplainer))

	if single_instance:
		shap_values = shapexplainer.explain_instance(X_test.iloc[0,:])
		print(shap_values)


def run_on_diabetes(diabetes_path):
	'''
	Here the trainer is run for diabetes
	'''
	pima_diabetes = pd.read_csv(diabetes_path, index_col=0)
	transformedDF = transform_data(pima_diabetes)

	models = get_models()

	x_train, x_test, y_train, y_test = generate_train_test_split(transformedDF, 0.30)
	evaluate_model(models, x_train, y_train)
	model_output = {}

	for mod_num in models.keys():
		model = models[mod_num]

		(model, mod_classification_report) = fit_and_predict_model(mod_num, model, x_train, y_train, x_test, y_test)

		#this is where you get the model 

		model_output[mod_num] = (model, mod_classification_report)

	model_output_print = get_model_output(model_output, True, 0)
	print('Testing proba ', model_output_print[0].predict_proba)
	print(model_output_print[1])

	return (model_output_print[0], x_train, x_test, y_train, y_test)

if __name__=='__main__':
	'''
	This stage would take as input user question, reframed question and identify explainers relevant for the explanation type
	'''

	'''
	Trial - run SHAP on best model for PIMA Indians Diabetes dataset 
	Need to have access to:
	- Dataset splits
	- Model 
	'''
	(trained_model, x_train, x_test, y_train, y_test) = run_on_diabetes(codeconstants.DATA_FOLDER + '/diabetes/diabetes_val_corrected.csv')
	run_shap(trained_model, x_train, x_test)
