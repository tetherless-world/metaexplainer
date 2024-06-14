from ragas import evaluate

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

'''
Construct the question, contexts and answer fields for each result frame so that they can be evaluated.
'''

def construct_eval_record(explanations, question, subsets, result_set):
    '''
    read a result folder and generate the record 
    '''