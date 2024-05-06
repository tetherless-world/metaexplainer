from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np
import pandas as pd

import sys
sys.path.append('../')
from metaexplainercode import codeconstants


def find_cosine_similarity(s1, s2):
	'''
	return score for sklearn's cosine similarity
	'''
	comparisons = (s1, s2)
	#print(comparisons)

	tfidf_vectorizer = TfidfVectorizer()
	tfidf_matrix = tfidf_vectorizer.fit_transform(comparisons)

	result_cos = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
	return result_cos[0][1]

def print_list(list_n):
	'''
	Print contents of list on separate lines
	'''
	for val in list_n:
		print(val)

def write_list(list_n, file_name):
	with open(file_name, 'w') as f:
		for line in list_n:
			f.write(f"{line.encode('utf-8')}\n")


def read_list_from_file(file_name):
	f = open(file_name, 'r', encoding='utf-8')
	lines = f.read().splitlines()
	return lines

def generate_confusion_matrix_and_visualize(y_pred, y_true, labels, output_file_path): 
	cm = confusion_matrix(y_true, y_pred, labels=labels)
	cm_array_df = pd.DataFrame(cm, index=labels, columns=labels)
	cm_heatmap = sns.heatmap(cm_array_df, annot=True, annot_kws={"size": 12})
	#based on https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
	fig = cm_heatmap.get_figure()
	fig.savefig(codeconstants.OUTPUT_FOLDER + '/' + output_file_path, bbox_inches='tight')
	fig.show()
	
