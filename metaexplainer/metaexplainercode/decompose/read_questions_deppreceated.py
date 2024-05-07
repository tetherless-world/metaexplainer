import re
import nltk
import random

import sys
sys.path.append('../')
from metaexplainercode import codeconstants

def read_utterances_file(utterances_path):
	with open(codeconstants.QUESTIONS_FOLDER + '/' + utterances_path) as f:
		conts = f.read()
		conts = conts.split('\n\n')
		questions_operations = [{'question': cont.split('\n')[0], 'talk_to_model_operations': cont.split('\n')[1]}
		for cont in conts]
		return questions_operations
	return []

def print_sample(utterances_pairs):
	'''
	Randomly print top-5 questions from utterance pairs
	'''
	rands = random.sample(range(0, len(utterances_pairs)), 5)

	for i in rands:
		print('Question ', utterances_pairs[i]['question'])
		print('Operations ', utterances_pairs[i]['talk_to_model_operations'])
		print('-----')

if __name__=='__main__':
	german_questions = read_utterances_file('german_test_suite.txt')
	diabetes_questions = read_utterances_file('diabetes_test_suite.txt')
	compas_questions = read_utterances_file('compas_test_suite.txt')

	print('Number of questions \n ',
		' Diabetes ', len(diabetes_questions), '\n',
		' COMPAS ', len(compas_questions), '\n',
		' German credit ', len(compas_questions))

	print('Diabetes samples')
	print_sample(diabetes_questions)
	print('*****'*5)

	print('Compas samples ')
	print_sample(compas_questions)
	print('*****'*5)

	print('German credit samples')
	print_sample(german_questions)
	print('*****'*5)

