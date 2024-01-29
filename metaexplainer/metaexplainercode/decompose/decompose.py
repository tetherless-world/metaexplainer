#A start at code for decomposing a question into question head and entity part
from rdflib import Graph
import rdflib
import ontospy
import pandas as pd

import os
import re

import sys
sys.path.append('../')
from metaexplainercode import codeconstants

def get_children_of_class(ont_class):
	'''
	Return a list of all children of a class
	'''
	children_list = ont_class.children()
	return children_list

def get_class_term(loaded_ont, class_term, search_index):
	'''
	Return class found if there is more than 1 match, allow the user to pick
	'''
	class_list = loaded_ont.get_class(class_term)
	class_at_index = class_list[search_index]
	return class_at_index


def get_property_value(class_obj, URIRef):
	'''
	Trying to get a value for a given property on a class
	For property pass the entire URI including namespace and property name 
	'''
	value_uri = class_obj.getValuesForProperty(rdflib.term.URIRef(URIRef))

	#retrieve string value of question if present
	if len(value_uri) > 0:
		value_uri = value_uri[0].toPython()
	else:
		value_uri = ''

	return value_uri


def get_example_question(class_obj):
	'''
	Get example questions listed against each explanation class
	They are stored as skos:example so just retrieve value for that
	'''
	skos_example_URI = 'http://www.w3.org/2004/02/skos/core#example'
	label_URI = 'http://www.w3.org/2000/01/rdf-schema#label'

	class_obj._buildGraph()
	child_uri = class_obj.uri
	quest = get_property_value(class_obj, skos_example_URI)
	label = get_property_value(class_obj, label_URI)	

	return (label, quest)

def extract_quoted_string(questText):
	'''
	Find all quoted quesitons 
	Pattern from: https://www.reddit.com/r/regex/comments/u3ao9g/trouble_capturing_multiple_string_matches_with/
	'''
	quoted = re.compile(r'[\"\“]((?:[^\"])+)\"')
	all_quoted_strings = []

	for value in quoted.findall(questText):
		if value != '':
			all_quoted_strings.append(value)
	return all_quoted_strings

def find_similar_question(question, questions_list):
	'''
	Find similar question based on the user input 
	'''
	questions_parsed = {exp_quest['question']: exp_quest['explanation']  for exp_quest in questions_list}
	print(questions_parsed)
	

def get_class_content(ont_class):
	'''
	Return class details including properties and property values
	'''
	pass
	
if __name__=="__main__":
	#Trying to load EO and inspect it; this works better than others
	eo_model = ontospy.Ontospy("https://purl.org/heals/eo",verbose=True)
	#eo_model.printClassTree()
	explanation_class = get_class_term(eo_model, "explanation", -1)
	children_list_exp = get_children_of_class(explanation_class)
	print('Explanations Extracted', children_list_exp)

	if not os.path.exists(codeconstants.OUTPUT_FOLDER + '/prototypical_questions_explanations_eo.csv'):
		print("Example questions for each child are")
		questions_children = []

		for child in children_list_exp:
			(child_uri, value_uri) = get_example_question(child)
			quests = extract_quoted_string(value_uri)

			for quest in quests:
				questions_children.append({'explanation': child_uri, 'question': quest})

		questions_children = pd.DataFrame(questions_children)

		if not os.path.exists(codeconstants.OUTPUT_FOLDER):
	   		os.makedirs(codeconstants.OUTPUT_FOLDER)

		questions_children.to_csv(codeconstants.OUTPUT_FOLDER + '/prototypical_questions_explanations_eo.csv')
	else:
		print('Proto file found not extracting questions again!')
		questions_children = pd.read_csv(codeconstants.OUTPUT_FOLDER + '/prototypical_questions_explanations_eo.csv')
		questions_children = questions_children.to_dict('records')

	
	find_similar_question('Why Semgluatide over Metformin?', questions_children)		

	

