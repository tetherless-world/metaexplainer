import scipy
import matplotlib as plt

import os
import pandas as pd

import sys
sys.path.append('../')
import copy

from metaexplainercode import codeconstants

from metaexplainercode import metaexplainer_utils

def read_records_df(record_path):
    df = pd.read_csv(record_path, skiprows=1, header=None).T   # Read csv, and transpose
    df.columns = df.iloc[0]                                 # Set new column names
    df.drop(0,inplace=True)
    return df

def evaluate_parses(domain_name, mode='generated'):
    decompose_parses = metaexplainer_utils.load_delegate_parses(domain_name, mode=mode)

    print('Reporting parse stats for mode ', mode)

    print('Usable passes ', len(decompose_parses))

    print('Number of explanation types ', decompose_parses['Explanation type'].value_counts())

    unusable_parses = metaexplainer_utils.load_delegate_parses(domain_name, mode=mode,usable=False)

    print('Length of unusable passes ', len(unusable_parses))



def plot_explainer_results(mode='generated'):
    delegate_results_folder = codeconstants.DELEGATE_FOLDER + '/results/'

    if mode == 'fine-tuned':
        delegate_results_folder = codeconstants.DELEGATE_FOLDER + '/fine-tuned_results/'

    overall_evals = []

    results_dict_format = {'Explanation Type': '', 'Explainer Method': '', 'Metric': '', 'Value': 0}

    for dir_ep in os.listdir(delegate_results_folder):
        dir_ep_path = delegate_results_folder + '/' + dir_ep
        record_dets = read_records_df(dir_ep_path + '/record.csv')
        #print(record_dets.columns)
        
        sub_dirs = os.listdir(dir_ep_path)

        for subset_dir in sub_dirs:
            subset_dir_path = dir_ep_path + '/' + subset_dir

            if os.path.isdir(subset_dir_path):
                evaluations = pd.read_csv(subset_dir_path + '/Evaluations.csv')

                for index, eval_vals in evaluations.iterrows():
                    editable_record = copy.deepcopy(results_dict_format)
                    editable_record['Explanation Type'] = record_dets.iloc[0]['Explanation type']
                    editable_record['Explainer Method'] = dir_ep.split('_')[2]
                    editable_record['Metric'] = eval_vals['Metric']
                    editable_record['Value'] = eval_vals['Value']
                    overall_evals.append(editable_record)
    
    overall_evals_df = pd.DataFrame(overall_evals)

    grouped_mean = overall_evals_df.groupby('Metric')['Value'].mean().reset_index()

    # Renaming the columns for better readability
    grouped_mean.columns = ['Metric', 'mean_value']

    print("\nMean by group:")
    print(grouped_mean)
    
    overall_evals_df.to_csv(codeconstants.DELEGATE_FOLDER + '/' + mode + '_explainer_evaluations.csv')

if __name__=='__main__':
    '''
    Read evaluation files from different folders and add to corresponding: explanation type, explainer method, metric, value dataframe 
    '''
    domain_name = 'Diabetes'
    
    plot_explainer_results(mode='fine-tuned')

    #evaluate_parses(domain_name, mode='generated')

   # evaluate_parses(domain_name, mode='fine-tuned')
    




