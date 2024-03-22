import re
import nltk

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

if __name__=='__main__':
	german_questions = read_utterances_file('german_test_suite.txt')
	diabetes_questions = read_utterances_file('diabetes_test_suite.txt')
	compas_questions = read_utterances_file('compas_test_suite.txt')

	print('Number of questions \n ',
		' Diabetes ', len(diabetes_questions), '\n',
		' COMPAS ', len(compas_questions), '\n',
		' German credit ', len(compas_questions))
