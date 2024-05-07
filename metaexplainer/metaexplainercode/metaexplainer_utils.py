from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score


#visualization
import matplotlib.pyplot as plt
import seaborn as sns

#data stack
import numpy as np
import pandas as pd

import torch
import collections 

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

def compute_f1(reference_strs, result_strs):
	'''
	Assume you have a list of strings and return F1s for each of those 
	Inspired by: https://cohere.com/blog/evaluating-llm-outputs
	'''
	

	gold_toks = sum([sent.split(' ') for sent in reference_strs], [])
	print('Sample ', gold_toks[:20])
	pred_toks = sum([sent_res.split(' ') for sent_res in result_strs], [])
	print('Sample pred ', pred_toks[:20])

	common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
	num_same = sum(common.values())

	if len(gold_toks) == 0 or len(pred_toks) == 0:
		#if either is no answer, then F1 is 1 if they agree, 0 otherwise
		return int(gold_toks == pred_toks)
	if num_same == 0:
		return 0
	precision = 1.0 * num_same/len(pred_toks)
	recall = 1.0*num_same/len(gold_toks)
	f1 = (2*precision*recall)/(precision + recall)
	return (f1, precision, recall)


def compute_exact_match(reference_strs, result_strs):
	'''
	Get exact match accuracy acrosss two strings - stricter than F1 but this is useful to assess correctness
	'''
	
