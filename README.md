# Augmenting Transformer models with eye-tracking attention masks

This is the code for the project:

- Running main.py with different parameters to test different models augmented by different eye-tracking attention masks
- The data folder contained the datasets for both experiments along with eye-tracking data
- The util.py file contained the necessary pre-processing functions to run question-answering models, and the evaluation function generating all the evaluating indexes
- The EyeMask.py contained the function necessary to build eye-tracking attention masks for either BERT and ALBERT model, and the function for attention visualization

## Publication
Eye-Tracking Features Masking Transformer Attention in Question-Answering Tasks (Zhang & Hollenstein, LREC-COLING 2024) https://aclanthology.org/2024.lrec-main.619/

## Additional models used in the study
* The model used for generating question-answering pairs can be referred to this repository: https://github.com/ramsrigouthamg/Questgen.ai
* The model used for eye-tracking data prediction can be referred to this repository: https://github.com/SPOClab-ca/cmcl-shared-task
