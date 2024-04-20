#A start at code for decomposing a question into question head and entity part
from rdflib import Graph
import rdflib
import ontospy
import pandas as pd

#distance metrics imports

from Levenshtein import distance
from Levenshtein import jaro_winkler



import os
import re


import sys
sys.path.append('../')
from metaexplainercode import codeconstants

from metaexplainercode import metaexplainer_utils


def write_generated_questions(domain):
	'''
	Read question files and generate an excel sheet with one sheet for each explanation type
	'''	
	domain_dir_path = codeconstants.DECOMPOSE_QUESTIONS_FOLDER + '/' + domain
	domain_dirs = [dir_domain.name for dir_domain in os.scandir(domain_dir_path) if dir_domain.is_dir()]
	questions_csv = ''

	for explanation_type in domain_dirs:
		explanation_type_path = domain_dir_path + '/' + explanation_type
		explanation_dfs = []

		for gpt_file in os.listdir(explanation_type_path):
			'''
			Add all together and get a sense of columns
			'''
			gpt_file_path = open(explanation_type_path + '/' + gpt_file)
			gpt_file_cont = gpt_file_path.read()
			gpt_file_split = [quest.split('\n') for quest in gpt_file_cont.split('\n\n')]
			gpt_dict_records = []

			for quest_record in gpt_file_split:
				gpt_dict_record = {}
				for record_cont in quest_record:
					record_cont = record_cont.strip()
					record_cont = re.sub(r'[0-9]+.\s' ,'', record_cont)
					record_cont = re.sub(r'-\s','', record_cont)

					if ':' in record_cont:
						record_splits = record_cont.split(':')
						gpt_dict_record[record_splits[0]] = record_splits[1]
					else:
						gpt_dict_record['Question'] = record_cont

				gpt_dict_records.append(gpt_dict_record)


			gpt_dict_record_df = pd.DataFrame(gpt_dict_records)
			explanation_dfs.append([gpt_file, gpt_dict_record_df])

		#create a validation excel file for each explanation type
		val_file = domain_dir_path + '/' + explanation_type + '_validation.xlsx'
		with pd.ExcelWriter(val_file, engine='xlsxwriter') as writer:
			for df_entry in explanation_dfs:
				df_entry[1].to_excel(writer, sheet_name= df_entry[0])
			print('Finished writing validation file for ', val_file)
		#break
	'''
	Normalize each 
	'''


def get_children_of_class(ont_class):
	'''
	Return a list of all children of a class
	'''
	children_list = ont_class.descendants()
	return children_list

def get_class_term(loaded_ont, class_term, search_index):
	'''
	Return class found if there is more than 1 match, allow the user to pick
	'''
	class_list = loaded_ont.get_class(class_term)
	#class_parents = [class_o.parents() for class_o in class_list]
	#print('Parents ', class_parents)
	
	class_at_index = [class_matched for class_matched in class_list if get_property_value(class_matched, 'http://www.w3.org/2000/01/rdf-schema#label') == 'Explanation'][0]
	#print('All exps ', class_list[-1])
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
	#generating lookup with question and explanation, so that the most similar explanations can be returned
	scores_expl = []

	for exp_quest in questions_list:
		ref_question = exp_quest['question']
		expl_type = exp_quest['explanation']

		comparisons = (ref_question, question)
		#print(comparisons)

		result_cos = metaexplainer_utils.find_cosine_similarity(ref_question, question)
		leven_score = distance(ref_question, question)
		jaro_winkler_score = jaro_winkler(ref_question, question)

		scores_expl.append({'explanation type': expl_type, 'question': ref_question, 'levenshtein_score': leven_score, 'jaro_winkler_score': jaro_winkler_score, 'cosine_similarity': result_cos})
		#next add these to arrays and pick highest

	scores_expl = sorted(scores_expl, key=lambda x: x['cosine_similarity'], reverse=True)
	print('User question ', question)
	metaexplainer_utils.print_list(scores_expl)
	return {'User question': question, 'Comparison scores': scores_expl}
	

def get_class_content(ont_class):
	'''
	Return class details including properties and property values
	'''
	pass
	
if __name__=="__main__":
	if not os.path.exists(codeconstants.OUTPUT_FOLDER + '/prototypical_questions_explanations_eo.csv'):
		#Trying to load EO and inspect it; this works better than others
		eo_model = ontospy.Ontospy("https://purl.org/heals/eo",verbose=True)
		#eo_model.printClassTree()
		explanation_class = get_class_term(eo_model, "explanation", -1)
		children_list_exp = get_children_of_class(explanation_class)
		print('Explanations Extracted', children_list_exp, '\n # of explanations ', len(children_list_exp))

		print("Example questions for each child are")
		questions_children = []

		for child in children_list_exp:
			(exp_label, question) = get_example_question(child)
			print('Explanation ', exp_label, 'Questions ', question)
			quests = extract_quoted_string(question)

			for quest in quests:
				questions_children.append({'explanation': exp_label, 'question': quest})

		questions_children_pd = pd.DataFrame(questions_children)

		if not os.path.exists(codeconstants.OUTPUT_FOLDER):
			os.makedirs(codeconstants.OUTPUT_FOLDER)

		questions_children_pd.to_csv(codeconstants.OUTPUT_FOLDER + '/prototypical_questions_explanations_eo.csv')
	else:
		print('Proto file found not extracting questions again!')
		questions_children = pd.read_csv(codeconstants.OUTPUT_FOLDER + '/prototypical_questions_explanations_eo.csv')
		questions_children = questions_children.to_dict('records')

	
	#find_similar_question('Why is the patient`s A1C important for their Diabetes?', questions_children)		

	write_generated_questions('Diabetes')

