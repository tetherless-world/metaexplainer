'''
Main entry-point for delegate; so given a machine interpretation; the steps are:
- Call parse machine interpretation (parse_machine_interpretation)
- Get the corresponding explainer method for explanation type (extract_explainer_methods)
- Run the explainer method (run_explainers)
'''

def retrieve_sample_decompose_passes(mode='fine-tuned'):
    '''
    Read from delegate output folder, if not running the method in real-time
    '''
    pass

def get_corresponding_explainer():
    '''
    This could be just reading from delegate output folder 
    '''

def run_explainer(feature_groups, actions, explainer_method):
    '''
    Call corresponding explainer with feature group filters and actions 
    '''

if __name__=='__main__':
    parse = retrieve_sample_decompose_passes()
    explainer_method = get_corresponding_explainer()
    method_results = run_explainer(parse['feature_groups'], parse['actions'], explainer_method)