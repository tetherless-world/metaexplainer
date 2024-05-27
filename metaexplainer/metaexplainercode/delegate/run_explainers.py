import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.append('../')
from metaexplainercode import codeconstants

from metaexplainercode import metaexplainer_utils
from metaexplainercode.delegate.train_models.run_model_tabular import *

import matplotlib as plt

# Importing shap KernelExplainer (aix360 style)
from aix360.algorithms.shap import KernelExplainer

# the following import is required for access to shap plotting functions and datasets
import shap

from aix360.algorithms.protodash import ProtodashExplainer
import dice_ml

from metaexplainercode.delegate.run_delegate import get_domain_model
from metaexplainercode.delegate.parse_machine_interpretation import replace_feature_string_with_col_names

def filter_records(dataset, feature_groups, actions):
	'''
	Filter records based on feature groups before passing it to the explainers
	'''
	subset_dataset = dataset
	column_names = dataset.columns

	for feature_group in feature_groups:
		for feature in feature_group.keys():
			feature_val = feature_group[feature]

			if not feature in column_names:
				feature = replace_feature_string_with_col_names(feature, column_names)

			if feature_val != '' and metaexplainer_utils.is_valid_number(feature_val):
				print('Applying ', feature, 'fitler for vals ', feature_val)
				subset_dataset = dataset.iloc[(dataset[feature]- float(feature_val)).abs().argsort()[:2]]

				print(subset_dataset.head())



def run_protodash(dataset, transformations, X):
	'''
	Protodash helps find representative cases in the data 
	'''
	# convert pandas dataframe to numpy
	X_train = transformations.transform(X)

	data = X_train

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
	inc_prototypes = metaexplainer_utils.drop_unnamed_cols(inc_prototypes)

	# Compute normalized importance weights for prototypes
	inc_prototypes["Weights of Prototypes"] = np.around(W/np.sum(W), 2) 
	print('Running protodash ')
	print(inc_prototypes)

	

def run_brcg():
	'''
	Derive rules for prediction 
	'''
	pass

def run_dice(model, dataset, transformations, X, Y, mode='genetic'):
	'''
	Generate counterfactuals 
	Can pass conditions here too 
	mode can be genetic / random
	'''	
	#dataset = dataset.drop('Sex', axis=1)

	backend = 'sklearn'
	m = dice_ml.Model(model=model, backend=backend)
	m.transformer.func = 'ohe-min-max'
	#m.transformer = transformations

	#can automate this somehow - so that even pipeline can use it
	d = dice_ml.Data(dataframe= dataset, 
				  continuous_features=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'],
				  categorical_features=['Sex'],
				  outcome_name='Outcome')

	#you can specify ranges in counterfactuals - which is also nice! - https://github.com/interpretml/DiCE/blob/main/docs/source/notebooks/DiCE_model_agnostic_CFs.ipynb
	#set some instances for sampling

	print('# where outcome = 1 ',len(dataset[dataset['Outcome'] == 1.0]), '# where outcome = 0 ',len(dataset[dataset['Outcome'] == 0.0]))
	selection_range = (140, 143)

	#query_instances = dataset.drop(columns="Outcome")[selection_range[0]: selection_range[1]]
	query_instances = X[selection_range[0]: selection_range[1]]
	
	print('Query \n', query_instances)

	# LE = LabelEncoder()
	# query_instances['Sex'] = LE.fit_transform(query_instances['Sex'])

	y_queries = Y[selection_range[0]: selection_range[1]]
	
	print('Outcomes \n', y_queries)

	exp_genetic = dice_ml.Dice(d, m, method='random')
	dice_exp_genetic = exp_genetic.generate_counterfactuals(query_instances, 
														 total_CFs=2, 
														 desired_class="opposite",
														 random_seed=9, 
														 features_to_vary= [col.replace('num__', '') for col in metaexplainer_utils.find_list_difference(transformations.get_feature_names_out(), ['cat__Sex_Female'])],
														 verbose=False)
	
	dice_exp_genetic.visualize_as_dataframe(show_only_changes=True)


def generate_fnames_shap(shap_values, cols):
	vals= np.abs(shap_values.values).mean(0)
	feature_importance = pd.DataFrame(list(zip(cols,vals)),columns=['col_name','feature_importance_vals'])
	#print(feature_importance.head())
	feature_importance.sort_values(by=['feature_importance_vals'], key=lambda col: col.map(lambda x: x[1]), ascending=False, inplace=True)
	return feature_importance

def run_shap(model, transformations, X, single_instance=True):
	'''
	Based on: 
	https://aix360.readthedocs.io/en/latest/lbbe.html#shap-explainers
	https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html
	'''
	X_test = transformations.transform(X)
	shapexplainer = KernelExplainer(model.predict_proba, X_test, feature_names=transformations.get_feature_names_out()) 

	if single_instance:
		print(X_test.iloc[0,:])
		shap_values = shapexplainer.explain_instance(X_test.iloc[0,:])
		print(shap_values)
	else:
		#print(model.classes_, 'Column names', X_test.columns)
		sampled_dist = shap.sample(X_test,10)
		shap_values = shapexplainer.explainer(sampled_dist)
		
		feature_importances = generate_fnames_shap(shap_values, transformations.get_feature_names_out())
		print(feature_importances)
		
		#shap.plots.bar(shap_values, class_names=model.classes_)
		#shap.summary_plot(shap_values, sampled_dist, class_names=model.classes_)

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
	dataset = metaexplainer_utils.load_dataset(domain_name)
	(X, Y) = generate_X_Y(dataset, 'Outcome')

	
	(trained_model, transformations, results) = get_domain_model(domain_name)

	run_shap(trained_model, transformations, X, single_instance=False)
		
	run_protodash(dataset, transformations, X)

	run_dice(trained_model, dataset, transformations, X, Y)
