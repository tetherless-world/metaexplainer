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

def run_query_on_explanation_graph(explanation_type_label, ont_model):
	explanation = ontology_utils.get_class_term(eo_model, explanation_type_label, -1)
	#base it off of" https://stackoverflow.com/questions/16829351/is-there-a-hello-world-example-for-sparql-with-rdflib
	explanation_method_query = "prefix rdfs:<http://www.w3.org/2000/01/rdf-schema#> " \
	"prefix owl:<http://www.w3.org/2002/07/owl#> "\
	"prefix ep: <http://linkedu.eu/dedalo/explanationPattern.owl#> " \
	"prefix prov: <http://www.w3.org/ns/prov#> " \
	"select DISTINCT ?taskObject where {" \
	"?class (rdfs:subClassOf|owl:equivalentClass)/owl:onProperty ep:isBasedOn ." \
	"?class (rdfs:subClassOf|owl:equivalentClass)/owl:someValuesFrom ?object ." \
	"?object owl:intersectionOf ?collections ." \
	"?collections rdf:rest*/rdf:first ?comps ." \
	"?comps rdf:type owl:Restriction ." \
	"?comps owl:onProperty ?property ." \
	"?comps owl:someValuesFrom ?taskObject ." \
	"?class rdfs:label \"" + explanation_type_label + "\" . }" 

	explanation_graph = explanation.rdflib_graph
	method_results = explanation_graph.query(explanation_method_query)
	
	for result in method_results:
		class_label = metaexplainer_utils.get_multi_word_phrase_from_capitalized_string(result[0].split('#')[1])
		print(class_label)
		print(ontology_utils.get_instances_of_class(ont_model, class_label))

	return method_results


if __name__ == '__main__':
	eo_model = ontology_utils.load_eo()

	methods = run_query_on_explanation_graph('Data Explanation', eo_model)

	print(list(methods))

