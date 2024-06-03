import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.append('../')
from metaexplainercode import codeconstants

from metaexplainercode import metaexplainer_utils
from metaexplainercode.delegate.train_models.run_model_tabular import *

import matplotlib as plt
import joblib 
import pickle
import random

# Importing shap KernelExplainer (aix360 style)
from aix360.algorithms.shap import KernelExplainer
from rulexai.explainer import Explainer

from aix360.metrics import faithfulness_metric, monotonicity_metric

# the following import is required for access to shap plotting functions and datasets
import shap

from aix360.algorithms.protodash import ProtodashExplainer
import dice_ml

from metaexplainercode.delegate.parse_machine_interpretation import replace_feature_string_with_col_names


def filter_records(dataset, feature_groups, actions):
	'''
	Filter records based on feature groups before passing it to the explainers
	'''
	
	column_names = dataset.columns
	subsets = []

	for feature_group in feature_groups:
		subset_dataset = dataset

		for feature in feature_group.keys():
			feature_val = feature_group[feature]
			
			if not feature in column_names:
				feature = replace_feature_string_with_col_names(feature, column_names).strip()

			if feature != '' and feature_val != '' and metaexplainer_utils.is_valid_number(feature_val):
				#print('Applying ', feature, 'fitler for vals ', feature_val)
				subset_dataset = subset_dataset.iloc[(subset_dataset[feature]- float(feature_val)).abs().argsort()[:10]]

		subsets.append(subset_dataset)
	
	return subsets

class TabularExplainers():
	'''
	A wrapper class to encapsulate all the explainer methods so that they can be easily retrieved for each explanation method
	'''
	def __init__(self, model, transformations, results, dataset) -> None:
		#self.domain_name = domain_name
		self.model = model
		self.transformations = transformations
		self.results = results 

		self.dataset = dataset
		(self.X, self.Y) = generate_X_Y(self.dataset, 'Outcome')

	def run_protodash(self, passed_dataset=None):
		'''
		Protodash helps find representative cases in the data 
		'''
		# convert pandas dataframe to numpy
		if passed_dataset is None:
			X_train = self.transformations.transform(self.X)
		else:
			(X, y) = generate_X_Y(passed_dataset, 'Outcome')
			X_train = self.transformations.transform(X)

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

		inc_prototypes = self.dataset.iloc[S, :].copy()
		inc_prototypes = metaexplainer_utils.drop_unnamed_cols(inc_prototypes)

		# Compute normalized importance weights for prototypes
		inc_prototypes["Weights of Prototypes"] = np.around(W/np.sum(W), 2) 
		print('Running protodash ')
		print(inc_prototypes, type(inc_prototypes))
		return (inc_prototypes, protodash_explainer)

		

	def run_rulexai(self, passed_dataset=None):
		'''
		Derive rules for prediction - https://github.com/adaa-polsl/RuleXAI
		'''
		y_train = self.Y

		if passed_dataset is None:
			X_train = self.X
		else:
			(X_train, y_train) = generate_X_Y(passed_dataset, 'Outcome')
		
		X_train_raw = X_train
		X_train = self.transformations.transform(X_train)
		
		predictions = self.model.predict(X_train)
			# prepare model predictions to be fed to RuleXAI, remember to change numerical predictions to labels (in this example it is simply converting predictions to a string)
		model_predictions = pd.DataFrame(predictions.astype(str), columns=[y_train.name], index = y_train.index)

		# use Explainer to explain model output
		explainer =  Explainer(X = X_train, model_predictions = model_predictions, type = "classification")
		explainer.explain(X_org=X_train_raw)
		rules = explainer.get_rules()
		rules_df = pd.DataFrame({'Rules': rules})
		print(rules_df)

		return (rules_df, explainer)
		

	def run_dice(self, passed_dataset=None, mode='genetic'):
		'''
		Generate counterfactuals 
		Can pass conditions here too 
		mode can be genetic / random
		'''	
		#dataset = dataset.drop('Sex', axis=1)

		backend = 'sklearn'
		m = dice_ml.Model(model=self.model, backend=backend)
		m.transformer.func = 'ohe-min-max'
		#m.transformer = transformations

		#can automate this somehow - so that even pipeline can use it
		d = dice_ml.Data(dataframe= self.dataset, 
					continuous_features=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'],
					categorical_features=['Sex'],
					outcome_name='Outcome')

		#you can specify ranges in counterfactuals - which is also nice! - https://github.com/interpretml/DiCE/blob/main/docs/source/notebooks/DiCE_model_agnostic_CFs.ipynb
		#set some instances for sampling

		print('# where outcome = 1 ',len(self.dataset[self.dataset['Outcome'] == 1.0]), '# where outcome = 0 ',len(self.dataset[self.dataset['Outcome'] == 0.0]))
		
		def random_sample_from_dataset(X, y):
			selection_range = np.array(random.sample(range(len(X)), 5))

			query_instances = X.loc[selection_range]
			y_queries = y.loc[selection_range]

			return (query_instances, y_queries)

		(query_instances, y_queries) = ([], [])
		#Always limit the instances you need to find counterfactuals for
		if (passed_dataset is None):
			#sample from dataset
			(query_instances, y_queries) = random_sample_from_dataset(self.X, self.Y)
		elif (len(passed_dataset) == len(self.dataset)):
			#again sample from dataset
			(query_instances, y_queries) =random_sample_from_dataset(self.X, self.Y)
		else:
			#use passed dataset and choose 5 randomly
			(X, y) = generate_X_Y(passed_dataset, 'Outcome')

			if len(X) > 10:
				(query_instances, y_queries) = random_sample_from_dataset(X, y)
			else:
				(query_instances, y_queries) = (X, y)

		exp = dice_ml.Dice(d, m, method='random')
		dice_exp = exp.generate_counterfactuals(query_instances, 
															total_CFs=1, 
															desired_class="opposite",
															random_seed=9,
															#ignoring the categorical features 
															features_to_vary= [col.replace('num__', '') for col in metaexplainer_utils.find_list_difference(self.transformations.get_feature_names_out(), ['cat__Sex_Female'])],
															verbose=False)
		
		#dice_exp.visualize_as_dataframe(show_only_changes=True)
		counterfactuals_objs = dice_exp.cf_examples_list
		queries = [counterfactual.test_instance_df for counterfactual in counterfactuals_objs]
		counterfactuals = [counterfactual.final_cfs_df for counterfactual in counterfactuals_objs]
		
		def find_changes(orig_df, counterfactual_df):
			changed_df = {}

			for column in orig_df.columns:
				if metaexplainer_utils.is_cat(str(orig_df[column].dtype)):
					changed_df[column] = counterfactual_df.iloc[0][column]
				else:
					if column == 'Outcome':
						changed_df[column] = counterfactual_df.iloc[0][column]
					else:
						changed_df[column] = round(counterfactual_df.iloc[0][column] - orig_df.iloc[0][column], 2)
			
			return pd.DataFrame([changed_df])

		result_counterfactual = {}
		result_counterfactual['Changed'] = pd.concat([find_changes(queries[orig_index], counterfactuals['Counterfactuals'][orig_index]) for orig_index in range(0, len(counterfactuals))])
		result_counterfactual['Queries'] = pd.concat(queries)
		result_counterfactual['Counterfactuals'] = pd.concat(counterfactuals)

		for res_i in range(0, len(result_counterfactual['Changed'])):
			print('Orig \n', result_counterfactual['Queries'][res_i])
			print('Changed \n',result_counterfactual['Changed'][res_i] )
			print('-----')

		return (result_counterfactual, dice_exp)

	def run_shap(self, passed_dataset=None, single_instance=False):
		'''
		Based on: 
		https://aix360.readthedocs.io/en/latest/lbbe.html#shap-explainers
		https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html
		'''
		if passed_dataset is None:
			X_test = self.X
		else:
			(X_test, y_test) = generate_X_Y(passed_dataset, 'Outcome')

		X_test = self.transformations.transform(X_test)

		shapexplainer = KernelExplainer(self.model.predict_proba, X_test, feature_names=self.transformations.get_feature_names_out()) 

		def generate_fnames_shap(shap_values, cols):
			vals= np.abs(shap_values.values).mean(0)
			feature_importance = pd.DataFrame(list(zip(cols,vals)),columns=['col_name','feature_importance_vals'])
			#print(feature_importance.head())
			feature_importance.sort_values(by=['feature_importance_vals'], key=lambda col: col.map(lambda x: x[1]), ascending=False, inplace=True)
			return feature_importance

		if single_instance:
			print(self.dataset.iloc[0,:])
			shap_values = shapexplainer.explain_instance(X_test[0])
			print(shap_values)
			return (shap_values, shapexplainer)
		else:
			#print(model.classes_, 'Column names', X_test.columns)
			sampled_dist = shap.sample(X_test,10)
			shap_values = shapexplainer.explainer(sampled_dist)
			
			feature_importances = generate_fnames_shap(shap_values, self.transformations.get_feature_names_out())
			print(feature_importances, type(feature_importances))
			return (feature_importances, shapexplainer)
			
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

	#Won't work unless tried from run_delegate
	
	tabular_explainer = TabularExplainers(domain_name)

	(results, shap_exp) = tabular_explainer.run_shap(single_instance=False)

	
	# tabular_explainer.run_protodash()

	#tabular_explainer.run_dice()

	#tabular_explainer.run_rulexai()
