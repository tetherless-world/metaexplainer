import os

#A file to save constant file variables that are used across code files in this directory

DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), r'../../data/datasets/'))

OUTPUT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), r'../../data/output_files/'))
DECOMPOSE_QUESTIONS_FOLDER = OUTPUT_FOLDER + '/decompose_questions/'
XREF_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), r'../../data/xref/'))


QUESTIONS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), r'../../data/questions/guideline_questions.txt'))
QUESTIONS_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), r'../../data/questions/'))




# Constants used in the guideline_extractionenhancedfile
BASE_URL = 'http://care.diabetesjournals.org'
TOC_PATH = '/content/40/Supplement_1'
ADA_CPG_EXTENSION = '../../data/output_files/ADA'

#change this for each year 
GUIDELINE_OUTPUT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/output_files/ADA2021Guidelines.json'))
CHAPT_EXTENSION = os.path.abspath(os.path.join(os.path.dirname(__file__),'../../data/output_files/ADA2021GuidelinesChapter'))
CITATION_FILE  = os.path.abspath(os.path.join(os.path.dirname(__file__),'../../data/output_files/ADA2021CitationMap.csv'))
PLACEHOLDER_URL = 'https://care.diabetesjournals.org'



#Constants used in pubmed citation lookup
PUBMED_PKL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),'../../data/output_files/pubmedtype_2021_cits.pkl'))
PUBMED_CHAPTER_PKL_PATH = '../../data/output_files/chaptercit_2021_annotated.pkl'

#Constants used in the citation parser file
CITATION_EXCEL_FILE  = os.path.abspath(os.path.join(os.path.dirname(__file__), r'../../data/output_files/CitationPublicationType2021.xlsx'))
CITATION_UNIQUECITATIONS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), r'../../data/output_files/2021Chapter9Chapter10UniqueRCTS.xlsx'))
RECOMMENDATIONS_EXCEL_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), r"../../data/output_files/GuidelineRecommendations.xlsx"))
SENTENCE_EXCEL_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), r"../../data/output_files/GuidelineEvidenceSentences.xlsx"))
MOST_CITATIONS_EXCEL = os.path.abspath(os.path.join(os.path.dirname(__file__), r"../../data/output_files/MostCitedCitations.xlsx"))

#Constants used for cohort data loading
T2DM_FEATURES_ONSET_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), r'/data/cohorts/v1/t2dm_features_upto_onset.csv'))
CCS_CODES_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), r'/data/cohorts/v1/ccs.csv'))
T2DM_DRUG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), r'/data/shared/v1/t2dm_drug_ckd_w360_cv0_proto.csv'))
T2DM_LAB_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), r'/data/shared/v1/t2dm_lab_ckd_w360_cv0_proto.csv'))
DRUG_CODES_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), r'/data/shared/XREF/XREF_THERCLS.csv'))
LAB_CODES_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), r'/data/shared/XREF/XREF_LABVALUES_RESLTCAT.csv'))
#change sample_patient to pid on server
SAMPLE_PATIENT_DRUG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), r'/data/shared/v1/sample_patient_drug_ckd_w360_v0.csv'))
SAMPLE_PATIENT_LAB_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), r'/data/shared/v1/sample_patient_drug_ckd_w360_v0.csv'))

MULTI_CCS_CODES_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), r'/data/lookup/dxmlabel-13.csv'))
MULTI_CCS_CODES_LOOKUP_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), r'/data/lookup/ccs-multilookup.csv'))


PROTOTYPICAL_PATIENTS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), r'/data/cohorts/v1/prototypical_patients.csv'))



DEMO_QUESTIONS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), r'../../data/questions/demo_questions.csv'))
DEMO_JSON_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), r'../../data/ui_test_riskprediction.json'))
DEMO_LONGER_TEMPLATE_JSON_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), r'../../data/ui_test_riskprediction_featurelonger_template.json'))

DEMO_LONGER_JSON_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), r'../../data/ui_test_riskprediction_featurelonger.json'))


