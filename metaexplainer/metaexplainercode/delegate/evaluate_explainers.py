import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

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

import random

class EvaluateExplainer():
	def __init__(self, dataset, model, transformations) -> None:
		self.model = model
		self.dataset = dataset
		self.transformations = transformations
		(self.X, self.Y) = generate_X_Y(self.dataset, 'Outcome')

	def return_evaluation_metric(explainer_method):
		pass
		
	def evaluate_non_representativeness(self, explainer, passed_dataset=None):
		pass
	
	def evaluate_diversity(self, explainer, samples):
		pass		

	def evaluate_monotonicity_and_faithfulness(self, explainer, passed_dataset=None):
		'''
		Generate faithfulness and monotonicity scores - both use the same methods; so don't run it again
		'''
		ncases = [i for i in range(0, 10)]

		if passed_dataset is None:
			X_test = self.transformations.transform(self.X)
		else:
			(X, y) = generate_X_Y(passed_dataset, 'Outcome')
			X_test = self.transformations.transform(X)

		if (not (passed_dataset == None)) and len(passed_dataset) > 10:
			#choose random cases 
			ncases = random.choices(range(0, len(passed_dataset)), 10)
		
			
		fait = np.zeros(len(ncases))
		mon = np.zeros(len(ncases))

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

		
			fait[i] = faithfulness_metric(self.model, X_test[i], coefs, base)
			mon[i] = monotonicity_metric(self.model, X_test[i], coefs, base)

		return {'Faithfulness': np.mean(fait), 'Monotonicity': np.mean(mon)}
		