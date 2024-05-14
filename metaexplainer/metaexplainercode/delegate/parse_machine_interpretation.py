import pandas as pd
import random
import re
import os

import sys
sys.path.append('../')
from metaexplainercode import codeconstants

from metaexplainercode import metaexplainer_utils

'''
Parse machine-interpretations and delegate to the specific explanation type - model classes and also to the particular functions
'''

def read_interpretations_from_file(domain_name, mode='fine-tune'):
    '''
    The interpretations from GPT directly are stored in output_files/decompose/<domain_name>/finetune_questions.csv
    We will mainly need to parse the Explanation type and Machine interpretation columns
    '''
    interpretations_records = pd.read_csv(codeconstants.DECOMPOSE_QUESTIONS_FOLDER + '/' + domain_name + '/finetune_questions.csv')

    if mode == 'generated':
        interpretations_records = pd.DataFrame(metaexplainer_utils.process_decompose_llm_result('llama-3-8b-charis-explanation', 'Diabetes', 'test', output_mode='list'))


    #the sample record will be removed once there is a way to either read from output file or fine-tuned data
    return interpretations_records

def retrieve_random_record(questions_dataset):
    '''
    Use a randint to retrieve record and parse it
    '''
    rand_int = random.randint(0, len(questions_dataset) - 1)
    record = dict(questions_dataset.iloc[rand_int])
    return record

def parse_machine_interpretation(record):
    '''
    The goal is to return:
    - Keywords
    - Filter groups
    The explanation type already tells you what explainer to run
    '''
    machine_interpretation = str(record['Machine interpretation'])
    actions = re.findall(r'([\w]+)\(', machine_interpretation)
    parantheses_groups = re.findall(r'\(([^()]+)\)', machine_interpretation)

    len_actions = len(actions)
    len_groups = len(parantheses_groups)

    if len_actions == len_groups + 1:
        actions = actions[1:]
        len_actions = len(actions)

    combined = []
    if len_actions == len_groups:
        combined = [actions[i].strip() + ' <> ' + parantheses_groups[i].strip() for i in range(0, len_groups)]

    #print('Length of action and paranthesis groups are ', len(actions), len(parantheses_groups))

    return {'Actions': actions, 'Groups': parantheses_groups, 'Combined': combined}

def get_explanation_type(record):
    '''
    Simple function to retrieve explanation type
    '''
    explan_type = record['Explanation type'].strip()
    
    return {'Explanation type': explan_type}


if __name__=='__main__':
    domain_name = 'Diabetes'
    interpretations_records = read_interpretations_from_file(domain_name, mode='generated')
    column_names = metaexplainer_utils.load_column_names(domain_name)
    print(column_names)

    delegate_folder = codeconstants.DELEGATE_FOLDER 

    if not os.path.isdir(delegate_folder):
        os.mkdir(delegate_folder)

    output_txt = ''

    for i in range(0, len(interpretations_records)):
        #sample_record = retrieve_random_record(interpretations_records)
        sample_record = dict(interpretations_records.iloc[i])
        #print(sample_record)
        
        output_txt += 'Question: ' + str(sample_record['Question']) + '\n' + 'Machine interpretation: ' + str(sample_record['Machine interpretation']) + '\n'

        parsed_mi = parse_machine_interpretation(sample_record)
        explanation_type = get_explanation_type(sample_record)

        parsed_mi.update(explanation_type)

        for parsed in parsed_mi['Combined']:
            output_txt += 'Parsed: ' + parsed + '\n'

        output_txt += 'Explanation type: ' + str(parsed_mi['Explanation type']) + '\n---------\n'
    
    with open(codeconstants.DELEGATE_FOLDER + '/' + domain_name + '_parsed_delegate_instructions.txt', 'w') as f:
        f.write(output_txt)
    




