from openai import OpenAI
client = OpenAI()

import sys
sys.path.append('../')
from metaexplainercode import codeconstants

if __name__=="__main__":

  conts = ''

  for i in range(0, 3):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      # {"role": "system",
      #  "content": "You are an explanation assistant skilled at understanding the needs of users and the questions they could ask for which they require explanations. Anticipate what their needs might be based on the user's requirements and generate questions."},
       {"role": "system",
       "content": "Always answer in English."},
      {"role": "user",
       "content": "Generate 100 questions for contrastive explanations for the Diabetes domain. Contrastive questions are of the form, 'Why this and not that' or 'Why this feature versus this feature'. " \
       "Your features could be age, diabetes pedigree function, sex and BMI and the prediction is whether the patient has diabetes or not. Their values range from ages: 18 - 70, diabetes pedigree function: 0.2 - 1, sex: Male or female and BMI: 18 - 30." \
       " For each question, also generate a machine interpretation which includes a predicate logic interpretation of the question, features and also identify the target variable and if there is a low / high likelihood."}
       #recognizing in the question - all instances, all features - their filters and the values that are being compared, all datasets and the target variable and whether it is a low or high likelihood.
    ],
    temperature=0
    #above part of prompt 2 can be used across question types
  )

    output = completion.choices[0].message
    conts += output.content + '\n \n'

    print("Run ", str(i))

  with open(codeconstants.OUTPUT_FOLDER + '/gpt-questions.txt', 'w') as f:
    f.write(conts)