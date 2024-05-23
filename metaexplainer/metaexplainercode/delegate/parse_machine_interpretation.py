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


def replace_unnamed_columns(field_key_i, dictionary_replace, replacement_label):
    '''
    Used by the next parse function to replace actions with column names 
    '''
    replacement_terms = ['Unnamed', 'x', 'patient']
    replace_match = list(set(dictionary_replace[field_key_i].keys()).intersection(replacement_terms))
    
    if len(replace_match) > 0:
        dictionary_replace[field_key_i][replacement_label] = dictionary_replace[field_key_i][replace_match[0]]
        del dictionary_replace[field_key_i][replace_match[0]]


def get_explanation_type(record):
    '''
    Simple function to retrieve explanation type
    '''
    explan_type = record['Explanation type'].strip()
    
    return explan_type

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

    #keep removing vals from the actions and parantheses groups and then add any missing values to it
    mi_without_action_features = machine_interpretation

    printer = False

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
        mi_without_action_features.replace(feature_group, '')
        feature_groups_all.append(feature_groups)

    replaced_actions = []
    skipped = {}

    for action_i in range(len(actions)):
        action = actions[action_i]
        mi_without_action_features.replace(action, '')
        
        (if_label, replacement_label) = metaexplainer_utils.check_if_label(action, column_names)

        if printer:
            print(action, if_label, replacement_label)

        if if_label:
            replaced_actions.append(action)

            if (action_i < len(feature_groups_all)):
                replace_unnamed_columns(action_i, feature_groups_all, replacement_label)
            else:
                skipped[action] = replacement_label

    if '=' in mi_without_action_features:
        remaining_feature_groups = extract_feature_value_pairs(mi_without_action_features.strip(), column_names)
        feature_groups_all.append(remaining_feature_groups)
    
    if len(skipped) > 0:
        skipped_i = 0

        for skipped_action in skipped.keys():
            replace_unnamed_columns(skipped_i, feature_groups_all, skipped[skipped_action])
            skipped_i += 1

    actions = list(set(actions) - set(replaced_actions))

    record_mi = record['Machine interpretation']
    record_question = record['Question']

    (edited_cols, expanded_cols, acronym_cols) = metaexplainer_utils.generate_acronyms_possibilities(column_names)

    #if there are no recognized feature groups, then extract them from the machine interpretation / question
    if len(feature_groups_all) == 0:
        search_string = record_mi

        if search_string == '':
            search_string = record_question

        if search_string != '':
            #record_mi might contain label!

            matched_col = list(filter(lambda x: x.lower() in search_string.lower(), edited_cols.keys()))

            if len(matched_col) > 0:
                feature_groups_all.append({edited_cols[matched_col[0]]: ''})
            else:
                matched_col = list(filter(lambda x: x.lower() in search_string.lower(), expanded_cols.keys()))

                if len(matched_col) > 0:
                    feature_groups_all.append({expanded_cols[matched_col[0]]: ''})
                else:
                    matched_col = list(filter(lambda x: x.lower() in search_string.lower(), acronym_cols.keys()))

                    if len(matched_col) > 0:
                        feature_groups_all.append({acronym_cols[matched_col[0]]: ''})


    #print('Length of action and paranthesis groups are ', len(actions), len(parantheses_groups))

    return {'Question': record['Question'],
            'Machine interpretation': record['Machine interpretation'],
        'Actions': actions, 
        'Groups': parantheses_groups, 
        'Feature groups': feature_groups_all, 
        'Explanation type': get_explanation_type(record)}

def print_record(record):
    '''
    Returns a string version of the parsed record
    '''
    record_txt = 'Question : ' + str(record['Question']) + '\n' + 'Machine interpretation : ' + str(record['Machine interpretation']) + '\n'

    for action in record['Actions']:
        record_txt += 'Action : ' + action + '\n'

    for parsed_alts in record['Feature groups']:
        record_txt += 'Feature groups : ' + str(parsed_alts) + '\n'

    record_txt += 'Explanation type : ' + str(record['Explanation type']) + '\n---------\n'
    
    return record_txt

def report_usability(record):
    '''
    records are unusable if they:
    - have an unrecognized / no explanation type
    - machine interpretation / action /  is empty 
    '''
    valid_explanations = metaexplainer_utils.load_selected_explanation_types()

    if len(record['Feature groups']) == []:
        #print('I enter for actions and feature groups')
        return False
    
    if (record['Explanation type'] == '') or not(record['Explanation type'] in valid_explanations):
        #print('I enter for explanations', record['Explanation type'], record['Explanation type'] in valid_explanations)
        return False

    return True

def generate_output_file_name(domain_name, mode, data_split, records_type):
    write_folder = codeconstants.DELEGATE_FOLDER 
    output_file_name = ''

    if records_type == 'unusable':
        write_folder += '/' + 'unusable'

        os.makedirs(write_folder, exist_ok=True)

    output_file_name = write_folder + '/' + domain_name + '_parsed_' + mode + '_delegate_instructions.txt'

    if mode == 'generated':
        output_file_name = write_folder + '/' + domain_name + '_parsed_' + mode + '_' + data_split + '_delegate_instructions.txt'
    
    return output_file_name

if __name__=='__main__':
    domain_name = 'Diabetes'
    

    mode = 'generated' # 'fine-tuned' / 'generated'

    #only makes sense if the mode is generated and not fine-tuned 

    data_split = 'test'

    interpretations_records = read_interpretations_from_file(domain_name, mode=mode, data_split=data_split)
    column_names = metaexplainer_utils.load_column_names(domain_name)
    
    print(column_names)
    print('Generating delegate parses for ', mode, ' data from ', data_split, 'split')

    delegate_folder = codeconstants.DELEGATE_FOLDER 

    os.makedirs(delegate_folder, exist_ok=True)

    usable_output_txt = ''
    unusable_output_txt = ''

    usable_records = []

    for i in range(0, len(interpretations_records)):
        #This whole below part should be a function that takes as input an interpretation - which is {'Question': ,'Explanation type': , 'Machine interpretation': }
        #sample_record = retrieve_random_record(interpretations_records)
        sample_record = dict(interpretations_records.iloc[i])
        #print(sample_record)
        
        parsed_mi = parse_machine_interpretation(sample_record, column_names)
        record_output_txt = print_record(parsed_mi)


        if report_usability(parsed_mi):
            usable_output_txt += record_output_txt
            usable_records.append(parsed_mi)
        else:
            unusable_output_txt += record_output_txt
    
    usable_output_file_name = generate_output_file_name(domain_name, mode, data_split, 'usable')        

    with open(usable_output_file_name, 'w') as f:
        f.write(usable_output_txt)
    
    unusable_output_file_name = generate_output_file_name(domain_name, mode, data_split, 'unusable')        

    with open(unusable_output_file_name, 'w') as f:
        f.write(unusable_output_txt)
    
    
    print('Usable records ', len(usable_records))
    




