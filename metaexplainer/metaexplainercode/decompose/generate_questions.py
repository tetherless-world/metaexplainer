from openai import OpenAI
client = OpenAI()

import sys
sys.path.append('../')
from metaexplainercode import codeconstants


completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    # {"role": "system",
    #  "content": "You are an explanation assistant skilled at understanding the needs of users and the questions they could ask for which they require explanations. Anticipate what their needs might be based on the user's requirements and generate questions."},
     {"role": "system",
     "content": "Always answer in English."},
    {"role": "user",
     "content": "Generate fifty questions for contrastive explanations for the Diabetes domain. Contrastive questions are of the form, 'Why this and not that' or 'Why this feature versus this feature'. " \
     "Your features could be age, diabetes pedigree function, sex and BMI and the prediction is whether the patient has diabetes or not. Their values range from ages: 18 - 70, diabetes pedigree function: 0.2 - 1, sex: Male or female and BMI: 18 - 30." \
     "For each question, also generate a machine interpretation recognizing in the question -  instances and/or feature and their filters and the values that are being compared and/or datasets and the target variable and whether it is a low or high likelihood."}
  ]
)

output = completion.choices[0].message
print(output)

with open(codeconstants.OUTPUT_FOLDER + '/gpt-questions.txt', 'w') as f:
  f.write(output.content)