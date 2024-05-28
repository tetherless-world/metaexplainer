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

# Importing shap KernelExplainer (aix360 style)
from aix360.algorithms.shap import KernelExplainer
from rulexai.explainer import Explainer

# the following import is required for access to shap plotting functions and datasets
import shap

from aix360.algorithms.protodash import ProtodashExplainer
import dice_ml

from metaexplainercode.delegate.parse_machine_interpretation import replace_feature_string_with_col_names

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
	def __init__(self, domain_name) -> None:
		self.domain_name = domain_name

		(self.model, self.transformations, self.results) = get_domain_model(domain_name)
		self.dataset = metaexplainer_utils.load_dataset(domain_name)
		(self.X, self.Y) = generate_X_Y(self.dataset, 'Outcome')

	def run_protodash(self, passed_dataset=None):
		'''
		Protodash helps find representative cases in the data 
		'''
		# convert pandas dataframe to numpy
		if not passed_dataset is None:
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
		print(inc_prototypes)

		

	def run_brcg(self, passed_dataset=None):
		'''
		Derive rules for prediction - https://github.com/adaa-polsl/RuleXAI
		'''
		y_train = self.Y

		if passed_dataset is None:
			X_train = self.X
		else:
			(X_train, y_train) = generate_X_Y(passed_dataset, 'Outcome')
		
		X_train = self.transformations.transform(X_train)
		
		predictions = self.model.predict(X_train)
			# prepare model predictions to be fed to RuleXAI, remember to change numerical predictions to labels (in this example it is simply converting predictions to a string)
		model_predictions = pd.DataFrame(predictions.astype(str), columns=[y_train.name], index = y_train.index)

		# use Explainer to explain model output
		explainer =  Explainer(X = X_train, model_predictions = model_predictions, type = "classification")
		explainer.explain(X_org=self.X)
		rules = explainer.get_rules()
		print(rules)

		print(explainer.condition_importances_)
		

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
		

		#query_instances = dataset.drop(columns="Outcome")[selection_range[0]: selection_range[1]]
		if passed_dataset is None:
			selection_range = (140, 143)
			query_instances = self.X[selection_range[0]: selection_range[1]]
			y_queries = self.Y[selection_range[0]: selection_range[1]]
		else:
			(query_instances, y_queries) = generate_X_Y(passed_dataset, 'Outcome')
		
		print('Query \n', query_instances)
		
		print('Outcomes \n', y_queries)

		exp_genetic = dice_ml.Dice(d, m, method='random')
		dice_exp_genetic = exp_genetic.generate_counterfactuals(query_instances, 
															total_CFs=2, 
															desired_class="opposite",
															random_seed=9,
															#ignoring the categorical features 
															features_to_vary= [col.replace('num__', '') for col in metaexplainer_utils.find_list_difference(self.transformations.get_feature_names_out(), ['cat__Sex_Female'])],
															verbose=False)
		
		dice_exp_genetic.visualize_as_dataframe(show_only_changes=True)

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
			print(X_test.iloc[0,:])
			shap_values = shapexplainer.explain_instance(X_test.iloc[0,:])
			print(shap_values)
		else:
			#print(model.classes_, 'Column names', X_test.columns)
			sampled_dist = shap.sample(X_test,10)
			shap_values = shapexplainer.explainer(sampled_dist)
			
			feature_importances = generate_fnames_shap(shap_values, self.transformations.get_feature_names_out())
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
	
	tabular_explainer = TabularExplainers(domain_name)

	#tabular_explainer.run_shap()
		
	#tabular_explainer.run_protodash()

	#tabular_explainer.run_dice()

	tabular_explainer.run_brcg()
