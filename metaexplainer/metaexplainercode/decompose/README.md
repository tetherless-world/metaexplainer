Decompose takes a user question and finds explanation types that could address it.

The explanation types are identified by a combination of the operations below:
- Identify which explanations would match the question heads
- What are the types of entities being asked about in the question? -> columns, samples, feature labels, literature, etc. (yet to be implemented)

The main files are:
- pre_loader_post_processor.py
- generate_questions.py -> Uses GPT API to generate training samples whose files need to be validated
- fine_tune.py -> fine-tune HuggingFace generative LLMs on the GPT samples to generate machine interpretation of question that can then be used by the next step: Delegate. For example,
- {"Question": "Why did the model predict that a 45-year-old female with a BMI of 25 and a Diabetes Pedigree Function of 0.3 has Diabetes?",
- "Machine Interpretation": "Why did DiabetesPredictionModel decide that Patient(age=45, sex=Female, BMI=25, DiabetesPedigreeFunction=0.3) has Diabetes?",
- "Explanation Type": "Rationale Explanation",
- "Action": "Provide rationale for the prediction",
-  "Target variable": "Diabetes prediction (High likelihood)"}
