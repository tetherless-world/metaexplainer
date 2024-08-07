'''
Generate user evaluation files with details from RAG and delegate outputs 
'''

import sys
sys.path.append('../')

import pandas as pd
import time

from metaexplainercode import metaexplainer_utils
from metaexplainercode import codeconstants

if __name__=='__main__':
    #pick random set of 10 
    synthesis_results_folder = codeconstants.SYNTHESIS_FOLDER

    read_folders = metaexplainer_utils.read_delegate_explainer_outputs(mode='generated', stage='synthesis')
    print(len(read_folders))

    rand_questions_15 = metaexplainer_utils.get_random_samples_in_list(list(read_folders.keys()), 15)
    eval_records = []

    for folder in rand_questions_15:
        #need question, explanation, explanation type, metrics
        record_folder = {}
        quest_details = pd.read_csv(folder + '/Record.csv')
        record_folder['Question'] = quest_details['Question'][0]
        record_folder['Predicted Explanation Type'] = quest_details['Explanation type'][0]
        record_folder['Feature groups in Question'] = quest_details['Feature groups']

        #there can be multiple rows in explanations - what do you do here? Show each differently or as one?
        explanation_details = pd.read_csv(folder + '/Explanations.csv')
        record_folder['Explanation of Matched Subset'] = ''
        record_folder['Explanation of Explainer Outputs'] = ''

        ctr = 0

        for index, row in explanation_details.iterrows():
            ctr += 1
            record_folder['Explanation of Matched Subset'] += str(ctr) + ').' + row['Subset']
            record_folder['Explanation of Explainer Outputs'] += str(ctr) + ').' + row['Explanation']
        
        sub_folders = metaexplainer_utils.get_subfolders_in_folder(folder)
        metrics = []

        for sub_folder in sub_folders:
            metrics.append(pd.read_csv(sub_folder + '/Metrics.csv'))
            
        
        record_folder['Metrics'] = metaexplainer_utils.drop_unnamed_cols(pd.concat(metrics)).groupby(['Metric'], as_index=False).mean()
        eval_records.append(record_folder)
        #average along the metrics 
    
    print('Generated eval file for ', len(rand_questions_15), 'randomly choosen results.')
    metaexplainer_utils.create_folder(codeconstants.SYNTHESIS_FOLDER + '/eval_files/')
    pd.DataFrame(eval_records).to_excel(codeconstants.SYNTHESIS_FOLDER + '/eval_files/evaluation_set_' + str(int(time.time())) + '.xlsx')