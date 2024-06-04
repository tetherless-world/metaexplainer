'''
Read the AI methods supported for each explanation type in EO
Need to edit EO to include AI methods.
'''
from rdflib import Graph
import rdflib
import ontospy

import pandas as pd

import sys
sys.path.append('../')
from metaexplainercode import codeconstants

from jinja2 import Template

from metaexplainercode import metaexplainer_utils
from metaexplainercode import ontology_utils

def run_query_on_explanation_graph(explanation_type_label, ont_model):
	explanation = ontology_utils.get_class_term(ont_model, explanation_type_label, -1)
	#since you aready have class term you don't need to pass the label
	#print('Explanation object ', explanation)

	#base it off of" https://stackoverflow.com/questions/16829351/is-there-a-hello-world-example-for-sparql-with-rdflib
	explanation_method_query = Template(open(codeconstants.QUERIES_FOLDER + '/fetch_explainers.html', 'r').read()).render(explanation_type_label=explanation_type_label)
	#print('Debugging', explanation_method_query)
	explanation_graph = explanation.rdflib_graph
	method_results = explanation_graph.query(explanation_method_query)
	#print('Debugging ', list(method_results))
	methods_instances = []
	
	for result in method_results:
		class_label = ontology_utils.get_label_from_URI(result[0])
		
		#print(class_label)
		instances_m = ontology_utils.get_instances_of_class(ont_model, class_label)

		if len(instances_m) > 0:
			all_instances = [ontology_utils.get_label_from_URI(instance[0], split=False) for instance in instances_m]

			for instance in all_instances:
				methods_instances.append({'Explanation Type': explanation_type_label,'Methods': class_label, 'Instances': instance})

	return methods_instances


if __name__ == '__main__':
	eo_model = ontology_utils.load_eo()

	loaded_explanations = metaexplainer_utils.load_selected_explanation_types()
	ep_list = []

	for explanation in loaded_explanations:
		print('Explanation ', explanation)
		explanation_methods_instances = run_query_on_explanation_graph(explanation, eo_model)
		ep_list += explanation_methods_instances
		print('Methods - Instances', explanation_methods_instances)
		print('------')
	
	pd.DataFrame(ep_list).to_csv(codeconstants.DELEGATE_FOLDER + '/explanation_type_methods.csv')
