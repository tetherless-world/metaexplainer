'''
Read the AI methods supported for each explanation type in EO
Need to edit EO to include AI methods.
'''
from rdflib import Graph
import rdflib
import ontospy

import sys
sys.path.append('../')
from metaexplainercode import codeconstants

from metaexplainercode import metaexplainer_utils
from metaexplainercode import ontology_utils

def create_explanation_type_graph():
	#base it off of" https://stackoverflow.com/questions/16829351/is-there-a-hello-world-example-for-sparql-with-rdflib
	pass


if __name__ == '__main__':
	eo_model = ontology_utils.load_eo()

	data_explanation = ontology_utils.get_class_term(eo_model, 'Data Explanation', -1)

	print(dir(data_explanation))
	#need to call buildGraph here 
	data_explanation_graph = data_explanation.rdflib_graph
	
	sample_query = "prefix rdfs:<http://www.w3.org/2000/01/rdf-schema#> " \
"prefix owl:<http://www.w3.org/2002/07/owl#> " \
"prefix ep: <http://linkedu.eu/dedalo/explanationPattern.owl#> " \
"prefix prov: <http://www.w3.org/ns/prov#> select ?label ?exp where { ?exp a owl:Class . ?exp rdfs:label ?label . } "
	x = data_explanation_graph.query(sample_query)
	print(list(x))