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

def read_interpretations_from_file(domain_name, mode='fine-tune', data_split='test'):
    '''
    The interpretations from GPT directly are stored in output_files/decompose/<domain_name>/finetune_questions.csv
    We will mainly need to parse the Explanation type and Machine interpretation columns
    '''
    interpretations_records = pd.read_csv(codeconstants.DECOMPOSE_QUESTIONS_FOLDER + '/' + domain_name + '/finetune_questions.csv')

    if mode == 'generated':
        interpretations_records = pd.DataFrame(metaexplainer_utils.process_decompose_llm_result('llama-3-8b-charis-explanation', 'Diabetes',data_split, output_mode='list'))


    #the sample record will be removed once there is a way to either read from output file or fine-tuned data
    return interpretations_records

def retrieve_random_record(questions_dataset):
    '''
    Use a randint to retrieve record and parse it
    '''
    rand_int = random.randint(0, len(questions_dataset) - 1)
    record = dict(questions_dataset.iloc[rand_int])
    return record

def extract_feature_value_pairs(feature_val_string, column_names):
    '''
    Return a dictionary of feature: value pairs from a feature_val_string
    '''

    features_vals = feature_val_string.strip().split(',')
    #print(features_vals)
    features_dict = {}
    vals = []
    last_added_feature = ''

    for feature_or_val in features_vals:
        feature_or_val = feature_or_val.strip()

        if metaexplainer_utils.is_valid_number(feature_or_val):
            if (last_added_feature != '') and (features_dict[last_added_feature] == ''):
                (if_label, replacement_label) = metaexplainer_utils.check_if_label(last_added_feature, column_names)

                if if_label:
                    #need to add check for feature - else add this as unnamed
                    features_dict[replacement_label] = feature_or_val
                    del features_dict[last_added_feature]
                else:
                    features_dict[last_added_feature] = feature_or_val
            else:
                vals.append(feature_or_val)
        elif '=' in feature_or_val:
            feature_val = feature_or_val.split('=')
            features_dict[feature_val[0].strip()] = feature_val[1].strip()
        elif feature_or_val != '':
            features_dict[feature_or_val] = ''
            last_added_feature = feature_or_val

            if len(vals) > 0:
                features_dict[feature_or_val] = vals.pop()
    
    if len(vals) > 0:
        features_dict['Unnamed'] = vals.pop()
    
    return features_dict


def parse_machine_interpretation(record, column_names):
    '''
    The goal is to return:
    - Keywords
    - Filter groups
    The explanation type already tells you what explainer to run
    '''
    machine_interpretation = str(record['Machine interpretation'])

    actions = re.findall(r'([\w]+)\s?\(', machine_interpretation)
    parantheses_groups = re.findall(r'\(([^()]+)\)', machine_interpretation)

    if record['Question'] == 'What broader information about the current situation prompted the suggestion of this recommendation for a 55-year-old male with a BMI of 27 and a Diabetes Pedigree Function of 0.18?':
        print(parantheses_groups)

    len_actions = len(actions)
    len_groups = len(parantheses_groups)

    if len_actions == len_groups + 1:
        actions = actions[1:]
        len_actions = len(actions)

    '''
    Could be that action is in column name, in that case -> need to extract value right adjacent to it 
    Else leave action as is 
    If there are combined features in groups extract those and set filter vals -> (feature, range_pair)
    '''

    feature_groups_all = []
    

    for feature_group in parantheses_groups:
        feature_groups = extract_feature_value_pairs(feature_group.strip(), column_names)
        feature_groups_all.append(feature_groups)

    replaced_actions = []

    for action_i in range(len(actions)):
        action = actions[action_i]
        (if_label, replacement_label) = metaexplainer_utils.check_if_label(action, column_names)

        if if_label:
            if (action_i < len(feature_groups_all)):
                if 'Unnamed' in feature_groups_all[action_i].keys():
                    feature_groups_all[action_i][replacement_label] = feature_groups_all[action_i]['Unnamed']

                    replaced_actions.append(action)

                    del feature_groups_all[action_i]['Unnamed']
                elif 'x' in feature_groups_all[action_i].keys():
                    feature_groups_all[action_i][replacement_label] = feature_groups_all[action_i]['x']

                    replaced_actions.append(action)

                    del feature_groups_all[action_i]['x']
                    #print(feature_groups_all[action_i])
    
    actions = set(actions) - set(replaced_actions)

    #print('Length of action and paranthesis groups are ', len(actions), len(parantheses_groups))

    return {'Actions': actions, 'Groups': parantheses_groups, 'Alternate': feature_groups_all}

def get_explanation_type(record):
    '''
    Simple function to retrieve explanation type
    '''
    explan_type = record['Explanation type'].strip()
    
    return {'Explanation type': explan_type}


if __name__=='__main__':
    domain_name = 'Diabetes'
    

    mode = 'generated' # 'fine-tuned'

    #only makes sense if the mode is generated and not fine-tuned 

    data_split = 'test'

    interpretations_records = read_interpretations_from_file(domain_name, mode=mode, data_split=data_split)
    column_names = metaexplainer_utils.load_column_names(domain_name)
    
    print(column_names)

    delegate_folder = codeconstants.DELEGATE_FOLDER 

    if not os.path.isdir(delegate_folder):
        os.mkdir(delegate_folder)

    output_txt = ''

    for i in range(0, len(interpretations_records)):
        #This whole below part should be a function that takes as input an interpretation - which is {'Question': ,'Explanation type': , 'Machine interpretation': }
        #sample_record = retrieve_random_record(interpretations_records)
        sample_record = dict(interpretations_records.iloc[i])
        #print(sample_record)
        
        output_txt += 'Question : ' + str(sample_record['Question']) + '\n' + 'Machine interpretation : ' + str(sample_record['Machine interpretation']) + '\n'

        parsed_mi = parse_machine_interpretation(sample_record, column_names)
        explanation_type = get_explanation_type(sample_record)

        parsed_mi.update(explanation_type)

        for action in parsed_mi['Actions']:
            output_txt += 'Action : ' + action + '\n'

        for parsed_alts in parsed_mi['Alternate']:
            output_txt += 'Intermediate : ' + str(parsed_alts) + '\n'

        output_txt += 'Explanation type : ' + str(parsed_mi['Explanation type']) + '\n---------\n'
    
    output_file_name = codeconstants.DELEGATE_FOLDER + '/' + domain_name + '_parsed_' + mode + '_delegate_instructions.txt'

    if mode == 'generated':
        output_file_name = codeconstants.DELEGATE_FOLDER + '/' + domain_name + '_parsed_' + mode + '_' + data_split + '_delegate_instructions.txt'
        

    with open(output_file_name, 'w') as f:
        f.write(output_txt)
    




