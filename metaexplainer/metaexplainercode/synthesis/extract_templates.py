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

def get_definition_for_explanation_type(eo_model, ep_type):
    ep_term = ontology_utils.get_class_term(eo_model, ep_type, -1)
    definition = ontology_utils.get_property_value(ep_term, 'http://www.w3.org/2004/02/skos/core#definition')
    return definition

if __name__=='__main__':
    eo_model = ontology_utils.load_eo()

    et_methods = pd.read_csv(codeconstants.DELEGATE_FOLDER + '/explanation_type_methods.csv')

    ets = list(et_methods['Explanation Type'].unique())
    et_def = []

    for et in ets:
        et_def.append({'Explanation type': et, 'Definition': get_definition_for_explanation_type(eo_model, et)})
    
    metaexplainer_utils.create_folder(codeconstants.SYNTHESIS_FOLDER)

    print(et_def)

    pd.DataFrame(et_def).to_csv(codeconstants.SYNTHESIS_FOLDER + 'explanation_type_definition.csv')