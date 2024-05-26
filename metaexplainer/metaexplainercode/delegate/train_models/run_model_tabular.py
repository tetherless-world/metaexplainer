import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # for data visualization
import matplotlib.pyplot as plt # to plot charts
from collections import Counter
import os

# Modeling
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split
from sklearn.metrics import classification_report

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import sys
sys.path.append('../')

from metaexplainercode import metaexplainer_utils


def is_cat(column_dtype):
	'''
	In this approach string columns also get 1-hot encoded - but that is ok
	'''
	categorical_dtypes = ['object', 'category', 'bool']

	if column_dtype in categorical_dtypes:
		return True
	else:
		return False   


def transform_data(df, columns_to_ignore, outcome_columns):
	'''
	generate a normalized dataframe for training
	'''
	column_list = df.columns
	column_list = metaexplainer_utils.find_list_difference(column_list, columns_to_ignore)
	column_list = metaexplainer_utils.find_list_difference(column_list, outcome_columns)

	categorical_cols = []

	for col in column_list:
		if is_cat(str(df[col].dtype)):
			categorical_cols.append(col)
	
	numerical =  metaexplainer_utils.find_list_difference(column_list, categorical_cols)

	# We create the preprocessing pipelines for both numeric and categorical data.
	numeric_transformer = Pipeline(steps=[
		('scaler', StandardScaler())])

	categorical_transformer = Pipeline(steps=[
		('onehot', OneHotEncoder(handle_unknown='ignore'))])

	transformations = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical),
        ('cat', categorical_transformer, categorical_cols)])
	
	print('Numerical columns ', numerical)
	print('Categorical columns ', categorical_cols)
	
	return transformations

def get_models():
	'''
	Define here what models you want to run - maybe this can be parameterized
	First number in list is stored on the blockchain
	'''
	random_state = 30

	models = {
		1: ['Logistic Regression', LogisticRegression(random_state = random_state, solver='liblinear')],
		2: ['Decision Tree',DecisionTreeClassifier(random_state = random_state)],
		3: ['Random Forest', RandomForestClassifier(random_state = random_state)],
	}
	return models

def get_model(model_no):
	'''
	Hospital choose to run a particular model
	'''
	random_state = 30

	models = get_models()

	if model_no >= 1 and model_no <= 3:
		return {model_no: models[model_no]}
	else:
		#invalid model_no and then in any case the max will be picked in report crunching
		return models

def generate_X_Y(df, outcome_column):
	features = df.drop([outcome_column], axis=1)
	labels = df[outcome_column]
	return (features, labels)

#Separate train dataset and test dataset
def generate_train_test_split(transformedDF, outcome_column, test_size):
	(features, labels) = generate_X_Y(transformedDF, outcome_column)
	x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=7)
	return x_train, x_test, y_train, y_test


def evaluate_model(models, transformations, x_train, y_train):
	"""
	Takes a list of models and returns chart of cross validation scores using mean accuracy
	"""
	
	# Cross validate model with Kfold stratified cross val
	kfold = StratifiedKFold(n_splits = 10)
	
	result = []
	for mod_num in models.keys():
		model = models[mod_num]
		pipeline = Pipeline([('transformer', transformations), ('estimator', model[1])])
		model_cv = cross_val_score(pipeline, X = x_train, y = y_train, scoring = "accuracy", 
									  cv = kfold, n_jobs=4)
		result.append([model_cv, mod_num])

	cv_means = []
	cv_std = []
	cv_model = []
	for cv_result in result:
		#print(cv_result)
		cv_means.append(cv_result[0].mean())
		cv_std.append(cv_result[0].std())
		cv_model.append(cv_result[1])

	result_df = pd.DataFrame({
		"CrossValMeans":cv_means,
		"CrossValerrors": cv_std,
		"Models":cv_model
	})

	# Generate chart
	bar = sns.barplot(x = "CrossValMeans", y = "Models", data = result_df, orient = "h")
	bar.set_xlabel("Mean Accuracy")
	bar.set_title("Cross validation scores")
	return result_df


def fit_and_predict_model(mod_num, transformations, model, x_train, y_train, x_test, y_test, save_model=False):
	'''
	Function to train a model, test it and report F1, precision and recall on the test predictions
	'''
	pipeline = Pipeline([('transformer', transformations), ('estimator', model[1])])
	model_name = model[0]

	if not save_model:
		pipeline.fit(x_train, y_train)
		y_pred_model = pipeline.predict(x_test)
		class_report = classification_report(y_test, y_pred_model, output_dict=True)
	else:
		pipeline.fit(x_train, y_train)
		y_pred_model = pipeline.predict(x_train)
		model_name = model[0]['model']
		class_report = classification_report(y_train, y_pred_model, output_dict=True)
	
	class_report = pd.DataFrame(class_report).T
	class_report = class_report.set_axis(class_report.columns, axis=1).rename_axis('dimensions',axis=0)
	class_report.reset_index(inplace=True)
	
	clean_class_report_edited = clean_class_report(class_report, len(x_train), len(x_test), mod_num, model_name)
	return (model[1], clean_class_report_edited)

def clean_class_report(class_report, num_x_train, num_x_test, mod_num, mod_name):
	'''
	Just get the classification report dictionary idea is to parse into output the blockchain can store
	E.g., train_data, test_data, overall_precision, sensitivity, specificity, overall_f1
	'''
	class_report_edited = {'model number': mod_num,
	'model': mod_name,
							'train size': num_x_train, 
							'test size': num_x_test,
							'test precision': round(class_report[class_report['dimensions'] == 'weighted avg']['precision'].item(), 2),
							'test recall': round(class_report[class_report['dimensions'] == 'weighted avg']['recall'].item(), 2),							
							'test f1': round(class_report[class_report['dimensions'] == 'weighted avg']['f1-score'].item(), 2),
							'test sensitivity (recall - yes diabetes)': round(class_report[class_report['dimensions'] == '1']['recall'].item(), 2),
							'test speficity (recall - no diabetes)': round(class_report[class_report['dimensions'] == '0']['recall'].item(), 2)
							}

	return class_report_edited


def get_best_model(models_output, pick_other_node_model):
	'''
	
	models_output contains the name of the model not the number 
	'''

	if pick_other_node_model >= 1 and pick_other_node_model <= 3:
		return models_output[pick_other_node_model]
	else:
		#pick the max model 
		max_f1 = 0
		max_model = 0

		for model in models_output.keys():
			class_report_model = models_output[model][1]

			if class_report_model['test f1'] > max_f1:
				max_model = model
				max_f1 = class_report_model['test f1']
				
		if max_model != 0:
			return (models_output[max_model][1], models_output[max_model][0])
		else:
			return (models_output[1][1], models_output[1][0])
	


if __name__ == '__main__':
	pima_diabetes = metaexplainer_utils.load_dataset('Diabetes')

	models = get_models()

	x_train, x_test, y_train, y_test = generate_train_test_split(pima_diabetes, 'Outcome', 0.30)
	
	transformations = transform_data(pima_diabetes, [], ['Outcome'])

	evaluate_model(models, transformations, x_train, y_train)
	
	model_output = {}

	for mod_num in models.keys():
		model = models[mod_num]

		(model, mod_classification_report) = fit_and_predict_model(mod_num, transformations, model, x_train, y_train, x_test, y_test)

		#this is where you get the model 

		model_output[mod_num] = (model, mod_classification_report)

	#print(model_output)
	model_output_print = get_best_model(model_output, 0)
	print('Testing proba ', model_output_print[1].predict_proba)
	print(model_output_print[0])


