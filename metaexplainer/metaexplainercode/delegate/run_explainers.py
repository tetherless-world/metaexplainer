import pandas as pd
import numpy as np

import sys
sys.path.append('../')
from metaexplainercode import codeconstants

from metaexplainercode import metaexplainer_utils
from train_models.run_model import *

import matplotlib as plt

# Importing shap KernelExplainer (aix360 style)
from aix360.algorithms.shap import KernelExplainer

# the following import is required for access to shap plotting functions and datasets
import shap

from aix360.algorithms.protodash import ProtodashExplainer
import dice_ml

def run_protodash(dataset, X_train, X_test):
	'''
	Protodash helps find representative cases in the data 
	'''
	# convert pandas dataframe to numpy
	data = X_train.to_numpy()

	#sort the rows by sequence numbers in 1st column 
	idx = np.argsort(data[:, 0])  
	data = data[idx, :]

	# replace nan's (missing values) with 0's
	original = data
	original[np.isnan(original)] = 0

	# delete 1st column (sequence numbers)
	original = original[:, 1:]

	protodash_explainer = ProtodashExplainer()
	(W, S, _) = protodash_explainer.explain(original, original, m=10)

	inc_prototypes = dataset.iloc[S, :].copy()

	# Compute normalized importance weights for prototypes
	inc_prototypes["Weights of Prototypes"] = np.around(W/np.sum(W), 2) 
	print(inc_prototypes)

	print('Running protodash ')

def run_brcg():
	'''
	Derive rules for prediction 
	'''
	pass

def run_dice(model, dataset, x_train, y_train, x_test, y_test, mode='genetic'):
	'''
	Generate counterfactuals 
	Can pass conditions here too 
	mode can be genetic / random
	'''
	dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
	dataset = dataset.drop(['Sex'], axis=1) 

	backend = 'sklearn'
	m = dice_ml.Model(model=model, backend=backend)
	#can automate this somehow - so that even pipeline can use it
	d = dice_ml.Data(dataframe= dataset, 
				  continuous_features=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'], 
				  outcome_name='Outcome')

	#you can specify ranges in counterfactuals - which is also nice! - https://github.com/interpretml/DiCE/blob/main/docs/source/notebooks/DiCE_model_agnostic_CFs.ipynb
	#set some instances for sampling

	print('# where outcome = 1 ',len(dataset[dataset['Outcome'] == 1.0]), '# where outcome = 0 ',len(dataset[dataset['Outcome'] == 0.0]))
	selection_range = (120, 123)

	query_instances = dataset.drop(columns="Outcome")[selection_range[0]: selection_range[1]]
	#y_queries = y_train[selection_range[0]: selection_range[1]]
	print('Query', query_instances)
	#print('Outcomes ', y_queries)

	exp_genetic = dice_ml.Dice(d, m, method='genetic')
	dice_exp_genetic = exp_genetic.generate_counterfactuals(query_instances, total_CFs=2, desired_class="opposite", verbose=True)
	dice_exp_genetic.visualize_as_dataframe(show_only_changes=True)


def generate_fnames_shap(shap_values, cols):
	vals= np.abs(shap_values.values).mean(0)
	feature_importance = pd.DataFrame(list(zip(cols,vals)),columns=['col_name','feature_importance_vals'])
	#print(feature_importance.head())
	feature_importance.sort_values(by=['feature_importance_vals'], key=lambda col: col.map(lambda x: x[1]), ascending=False, inplace=True)
	return feature_importance

def run_shap(model, X_train, X_test, single_instance=True):
	'''
	Based on: 
	https://aix360.readthedocs.io/en/latest/lbbe.html#shap-explainers
	https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html
	'''
	shapexplainer = KernelExplainer(model.predict_proba, X_test, feature_names=X_test.columns) 

	if single_instance:
		print(X_test.iloc[0,:])
		shap_values = shapexplainer.explain_instance(X_test.iloc[0,:])
		print(shap_values)
	else:
		print(model.classes_, 'Column names', X_test.columns)
		sampled_dist = shap.sample(X_test,10)
		shap_values = shapexplainer.explainer(sampled_dist)
		
		feature_importances = generate_fnames_shap(shap_values, X_test.columns)
		print(feature_importances)
		
		#shap.plots.bar(shap_values, class_names=model.classes_)
		#shap.summary_plot(shap_values, sampled_dist, class_names=model.classes_)

def run_on_diabetes(diabetes_path):
	'''
	Here the trainer is run for diabetes
	'''
	pima_diabetes = pd.read_csv(diabetes_path, index_col=0)
	pima_diabetes = pima_diabetes.loc[:, ~pima_diabetes.columns.str.contains('^Unnamed')]

	#need to define transforms here for categorical and numeric columns
	pima_diabetes = pima_diabetes.drop(['Sex'], axis=1) 

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

	domain_name = 'Diabetes'

	if domain_name == 'Diabetes':
		dataset = pd.read_csv(codeconstants.DATA_FOLDER + '/Diabetes/diabetes_val_corrected.csv')
		print('Columns of loaded dataset ', dataset.columns)

		(trained_model, x_train, x_test, y_train, y_test) = run_on_diabetes(codeconstants.DATA_FOLDER + '/Diabetes/diabetes_val_corrected.csv')
		#run_shap(trained_model, x_train, x_test, single_instance=False)
		run_protodash(dataset, x_train, x_test)

		run_dice(trained_model, dataset, x_train, y_train, x_test, y_test)
