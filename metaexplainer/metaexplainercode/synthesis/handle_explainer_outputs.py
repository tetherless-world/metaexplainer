'''
There are different formats of outputs, to make it easier for an LLM.
- Identify what exactly it needs to explain. e.g., for 
counterfactual - changes? 
contrastive - feature importances 
rationale - rules 
data and case-based - summary of cases
'''

class ParseExplainerOutput:
    def __init__(self, modality, explanation_type) -> None:
        self.modality = modality
        self.explanation_type = explanation_type

    def parse_representative_samples(self, output_frame):
        return output_frame

    def parse_rules(self, output_frame):
        return output_frame
    
    def parse_feature_importances(self, output_frame):
        output_frame['col_name'] = output_frame['col_name'].apply(lambda x: x.replace('num__', ''))
        output_frame['feature_importance_vals'] = output_frame['feature_importance_vals'].apply(lambda x: eval(x.replace(' ', ','))[0])
        return output_frame
    
    def parse_counterfactuals(output_frame):
        return output_frame



