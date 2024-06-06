import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import re

import sys
sys.path.append('../')
from metaexplainercode import codeconstants

from metaexplainercode import metaexplainer_utils
from metaexplainercode.delegate.train_models.run_model_tabular import *

# Importing shap KernelExplainer (aix360 style)
from aix360.algorithms.shap import KernelExplainer
from rulexai.explainer import Explainer

from aix360.metrics import faithfulness_metric, monotonicity_metric

# the following import is required for access to shap plotting functions and datasets
import shap

from aix360.algorithms.protodash import ProtodashExplainer
import dice_ml

from scipy.spatial.distance import pdist, squareform

import random

class EvaluateExplainer():
	def __init__(self, dataset, model, transformations) -> None:
		self.model = model
		self.dataset = dataset
		self.transformations = transformations
		(self.X, self.Y) = generate_X_Y(self.dataset, 'Outcome')

	

	def evaluate_average_rule_length_and_fidelity(self, explainer, passed_dataset=None,results=None):
		rules = list(results['Rules'])
		len_rules = len(rules)
		rule_lens = []

		if len_rules == 0:
			return [{'Metric': 'Average rule length', 'Value': 0}, {'Metric': 'Fidelity', 'Value': 0}]

		def extract_lower_upper_bounds(input_str):
			split_by_AND = input_str.split(' AND ')
			features_and_ranges = []

			for sub_string in split_by_AND:
				# Regular expression to extract features and their range restrictions
				pattern = r"(\w+) = (<[^,]+), ([^>]+)\)"

				# Find all matches
				matches = re.findall(pattern, sub_string)

				# Extracted features and their ranges
				for match in matches:
					feature, lower_bound, upper_bound = match
					features_and_ranges.append({
						'feature': feature,
						'lower_bound': lower_bound.strip('<'),
						'upper_bound': upper_bound.strip(')')
					})
				
			return features_and_ranges
	
		def check_restrictions(row, restrictions):
			# Function to evaluate if a row matches the restrictions

			for restriction in restrictions:
				feature = restriction['feature']
				lower_bound = restriction['lower_bound']
				upper_bound = restriction['upper_bound']
				value = row[feature]
				if not (float(lower_bound) < value < float(upper_bound)):
					return False
			return True

		fidelity = 0

		for rule in rules:
			antecedent_cond = rule.split('THEN')
			label = antecedent_cond[1].split(' = ')[1].replace('{', '').replace('}', '')

			rule_len = rule.count('AND') + 1

			rule_lens.append(rule_len)
			
			#feature restriction checker
			features_and_ranges = extract_lower_upper_bounds(antecedent_cond[0])

			# if rule_len > 1:
			# 	print(features_and_ranges)
			
			passed_dataset['matches_restrictions'] = passed_dataset.apply(check_restrictions, axis=1, restrictions=features_and_ranges)
			matched_rows = passed_dataset[passed_dataset['matches_restrictions'] == True]

			if len(matched_rows) > 0:
				print(matched_rows)
				y_pred = self.model.predict(self.transformations.transform(generate_X_Y(matched_rows.drop(['matches_restrictions'], axis=1), 'Outcome')[0]))
				#print('Debugging: predictions on matched rules', y_pred, 'Label ', int(label))
				len_agree = [1 for pred in y_pred if pred == int(label)]
				fidelity += sum(len_agree)/len(passed_dataset)

			#print('Debugging', 'pre', antecedents, 'label', label)
		
		return [{'Metric': 'Average rule length', 'Value': sum(rule_lens)/len_rules}, {'Metric': 'Fidelity', 'Value': fidelity/len_rules}]

	
	def evaluate_non_representativeness(self, explainer, passed_dataset=None,results=None):
		X, y = generate_X_Y(results, 'Outcome')
		n_e = len(results)

		transformations = transform_data(passed_dataset, [], ['Outcome']) 

		pipeline = Pipeline([('transformer', transformations), ('estimator', self.model)])
		losses = cross_val_score(pipeline, X = X, y = y, scoring = "neg_mean_squared_error", 
									  cv = 2)
		
		losses = list(map(abs, losses))

		return [{'Metric': 'Non representativeness', 'Value': (sum(losses))/(2*n_e)}]
	
	def evaluate_diversity(self, explainer, passed_dataset=None,results=None):
		n_e = len(results)
		cols = results.columns
		non_cat_columns = [col for col in cols if not metaexplainer_utils.is_cat(results[col].dtype)]
		numeric_subset = results[non_cat_columns]

		distances = pdist(numeric_subset.values, metric='euclidean')
		dist_matrix = squareform(distances)
		#print(dist_matrix, '\n', len(dist_matrix))

		sum_diversity = sum([(sum(distance_between_row)/(2*n_e)) for distance_between_row in dist_matrix])
		print('Diversity ', sum_diversity)

		return [{'Metric': 'Diversity', 'Value': sum_diversity}]
		
	def evaluate_monotonicity_and_faithfulness(self, explainer,passed_dataset=None,results=None):
		'''
		Generate faithfulness and monotonicity scores - both use the same methods; so don't run it again
		'''
		ncases = [i for i in range(0, 10)]

		if passed_dataset is None:
			X_test = self.transformations.transform(self.X)
		else:
			(X, y) = generate_X_Y(passed_dataset, 'Outcome')
			X_test = self.transformations.transform(X)

		if (passed_dataset is not None) and len(passed_dataset) > 10:
			#choose random cases 
			ncases = random.choices(range(0, len(passed_dataset)), k=10)
		
			
		fait = np.zeros(len(ncases))
		mon = np.zeros(len(ncases))
		ctr = 0

		for i in ncases:
			predicted_class = self.model.predict(X_test[i].reshape(1,-1))[0]
			exp = explainer.explain_instance(X_test[i])
			#extracting weights for features 
			le = exp[..., predicted_class]
			#print(le)
			
			x = X_test[i]
			coefs = np.zeros(x.shape[0])
			
			for v in range(0, len(le)):
				coefs[v] = le[v]

			base = np.zeros(x.shape[0])

		
			fait[ctr] = faithfulness_metric(self.model, X_test[i], coefs, base)
			mon[ctr] = monotonicity_metric(self.model, X_test[i], coefs, base)
			ctr+=1

		return [{'Metric': 'Faithfulness', 'Value': np.mean(fait)},
		  {'Metric': 'Monotonicity', 'Value': np.mean(mon)}]
		