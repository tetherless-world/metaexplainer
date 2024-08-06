'''
Generate user evaluation files with details from RAG and delegate outputs 
'''

import sys
sys.path.append('../')

import pandas as pd

from metaexplainercode import metaexplainer_utils
from metaexplainercode import codeconstants

if __name__=='__main__':
    #pick random set of 10 
    synthesis_results_folder = codeconstants.SYNTHESIS_FOLDER

    read_folders = metaexplainer_utils.read_delegate_explainer_outputs(mode='generated', stage='synthesis')
    print(len(read_folders))

    rand_questions_15 = metaexplainer_utils.get_random_samples_in_list(read_folders, 15)

    for folder in rand_questions_15:
        #need question, explanation, explanation type, metrics
        record_folder = {}
        quest_details = pd.read_csv(folder + '/Record.csv')
        record_folder['Question'] = quest_details['Question'][0]
        record_folder['Predicted Explanation Type'] = quest_details['Explanation type'][0]

        #there can be multiple rows in explanations - what do you do here? Show each differently or as one?
        explanation_details = pd.read_csv(folder + '/Explanations.csv')
        record_folder['Explanation of Matched Subset'] = explanation_details['Subset']
        record_folder['Explanation of Explainer Outputs'] = explanation_details['Explanation']
        
        sub_folders = metaexplainer_utils.get_subfolders_in_folder(folder)
        metrics = []

        for sub_folder in sub_folders:
            metrics.append(pd.read_csv(sub_folder + '/Metrics.csv'))
        
        #average along the metrics 