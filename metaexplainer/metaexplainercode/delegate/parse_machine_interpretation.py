import pandas as pd

import sys
sys.path.append('../')
from metaexplainercode import codeconstants

from metaexplainercode import metaexplainer_utils

'''
Parse machine-interpretations and delegate to the specific explanation type - model classes and also to the particular functions
'''

def read_interpretations_from_file(domain_name):
    '''
    The interpretations from GPT directly are stored in output_files/decompose/<domain_name>/finetune_questions.csv
    We will mainly need to parse the Explanation type and Machine interpretation columns
    '''
    interpretations_file = pd.read_csv(codeconstants.DECOMPOSE_QUESTIONS_FOLDER + '/' + domain_name + '/finetune_questions.csv')

    sample_record = dict(interpretations_file.iloc[0])

    print(sample_record)


if __name__=='__main__':
    read_interpretations_from_file('Diabetes')



