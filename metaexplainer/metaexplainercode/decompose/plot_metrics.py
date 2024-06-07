import scipy
import matplotlib as plt

import os
import pandas as pd

import sys
sys.path.append('../')
import copy

from metaexplainercode import codeconstants

from metaexplainercode import metaexplainer_utils


if __name__=='__main__':
    domain_name = 'Diabetes'
    gpt_parses = pd.read_csv(codeconstants.DECOMPOSE_QUESTIONS_FOLDER + '/' + domain_name + '/finetune_questions.csv')
    print(gpt_parses['Explanation type'].value_counts())
    