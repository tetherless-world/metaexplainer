from rdflib import Graph
import rdflib
import ontospy
import pandas as pd

import sys
sys.path.append('../')
from metaexplainercode import codeconstants

def load_eo():
	#eo_model = ontospy.Ontospy("https://purl.org/heals/eo",verbose=True)
	eo_model = ontospy.Ontospy("https://raw.githubusercontent.com/tetherless-world/explanation-ontology/master/Ontologies/v3/explanation-ontology.owl", verbose=True)
	#eo_model = ontospy.Ontospy(codeconstants.ONTOLOGY_FOLDER + 'explanation-ontology.owl', verbose=True)
	return eo_model

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

def get_label_class(class_obj):
	return get_property_value(class_obj, 'http://www.w3.org/2000/01/rdf-schema#label')

def get_class_term(loaded_ont, label, search_index):
	'''
	Return class found if there is more than 1 match, allow the user to pick
	'''
	label_edited = label.replace(' ', '')

	class_list = loaded_ont.get_class(label_edited)
	#print(class_list)
	#class_parents = [class_o.parents() for class_o in class_list]
	#print('Parents ', class_parents)
	
	class_at_index = [class_matched for class_matched in class_list if get_property_value(class_matched, 'http://www.w3.org/2000/01/rdf-schema#label') == label][0]
	#print('All exps ', class_list[-1])
	return class_at_index

def get_children_of_class(ont_class):
	'''
	Return a list of all children of a class
	'''
	children_list = ont_class.descendants()
	return children_list

def get_instances_of_class(ont_model, ont_class_label):
	'''
	Returns all instances of class and their labels
	'''
	ont_class = get_class_term(ont_model, ont_class_label, -1)	

	if not ont_class is None:
		#ont_class_graph = ont_class.rdflib_graph

		instance_query = "prefix rdfs:<http://www.w3.org/2000/01/rdf-schema#> " \
		"prefix owl:<http://www.w3.org/2002/07/owl#> "\
		"prefix ep: <http://linkedu.eu/dedalo/explanationPattern.owl#> " \
		"select ?instance where {" \
		"?instance a ?class . " \
		"?class rdfs:label \"" + ont_class_label + "\" . }" 

		instances = ont_model.query(instance_query)

		return instances

	return []
	