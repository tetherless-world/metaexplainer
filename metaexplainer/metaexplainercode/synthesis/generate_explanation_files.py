'''
Generate user evaluation files with details from RAG and delegate outputs 
'''

import sys
sys.path.append('../')

from metaexplainercode import metaexplainer_utils
from metaexplainercode import codeconstants

if __name__=='__main__':
    #pick random set of 10 
    synthesis_results_folder = codeconstants.SYNTHESIS_FOLDER

    read_folders = metaexplainer_utils.read_delegate_explainer_outputs(mode='generated', stage='synthesis')

    for folder in read_folders:
        print(folder)
        break