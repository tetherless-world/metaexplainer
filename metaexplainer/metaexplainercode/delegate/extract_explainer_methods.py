'''
Read the AI methods supported for each explanation type in EO
Need to edit EO to include AI methods.
'''
from rdflib import Graph
import rdflib
import ontospy

import copy

import pandas as pd

import sys
sys.path.append('../')
from metaexplainercode import codeconstants

from jinja2 import Template

from metaexplainercode import metaexplainer_utils
from metaexplainercode import ontology_utils

def run_modality_query_on_method_graph(explainer_method_label, ont_model):
	explainer_method = ontology_utils.get_class_term(ont_model, explainer_method_label, -1)
	explainer_graph = explainer_method.rdflib_graph
	modality_query = Template(open(codeconstants.QUERIES_FOLDER + '/fetch_explanation_modality_explainer.html', 'r').read()).render(explainer_method_label=explainer_method_label)
	#print('Debug',modality_query)
	modality_results = explainer_graph.query(modality_query)
	#print(modality_results)
	modality_edited = []
	modality_metrics_perm = []

	for modality in modality_results:
		modality_label = ontology_utils.get_label_from_URI(modality[0])
		modality_term = ontology_utils.get_class_term(ont_model, modality_label, -1)

		if len([parent for parent in modality_term.parents() if 'ExplanationModality' in str(parent)]) > 0:
			modality_edited.append(modality_label)
			modality_graph = modality_term.rdflib_graph

			metrics_query = Template(open(codeconstants.QUERIES_FOLDER + '/fetch_explanation_modality_explainer.html', 'r').read()).render(explanation_modality_label=modality_label)
			metrics_results = modality_graph.query(metrics_query)
			#print(modality_graph)
			#print(metrics_results)

			for metric in metrics_results:
				modality_metrics_perm.append({'Modality': modality_label,'Metric': ontology_utils.get_label_from_URI(metric[0])})

	#print('Debugging ', list(modality_edited))
	return (modality_edited, modality_metrics_perm)

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
	metrics_instances = []
	
	for result in method_results:
		class_label = ontology_utils.get_label_from_URI(result[0])
		(modalities, metrics) = run_modality_query_on_method_graph(class_label, ont_model)
		
		metrics_instances += metrics
		modality = ''

		if len(modalities) > 0:
			modality = modalities[0]
		
		#print(class_label)
		instances_m = ontology_utils.get_instances_of_class(ont_model, class_label)

		if len(instances_m) > 0:
			all_instances = [ontology_utils.get_label_from_URI(instance[0], split=False) for instance in instances_m]

			for instance in all_instances:
				methods_instances.append({'Explanation Type': explanation_type_label,'Methods': class_label,'Modality': modality, 'Instances': instance})

	return (methods_instances, metrics_instances)


if __name__ == '__main__':
	eo_model = ontology_utils.load_eo()

	loaded_explanations = metaexplainer_utils.load_selected_explanation_types()
	ep_list = []
	metrics_list = []

	for explanation in loaded_explanations:
		print('Explanation ', explanation)
		(explanation_methods_instances, metrics_instances) = run_query_on_explanation_graph(explanation, eo_model)
		ep_list += explanation_methods_instances
		metrics_list += metrics_instances

		print('Methods - Instances', explanation_methods_instances)
		print('------')
	
	ep_df = pd.DataFrame(ep_list)
	ep_df.drop_duplicates(inplace=True)

	metrics_df = pd.DataFrame(metrics_list)
	metrics_df.drop_duplicates(inplace=True) 

	ep_df.to_csv(codeconstants.DELEGATE_FOLDER + '/explanation_type_methods.csv')
	metrics_df.to_csv(codeconstants.DELEGATE_FOLDER + '/explanation_modality_metrics.csv')
