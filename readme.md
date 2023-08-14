# Usage
To reproduce the result, first config the hyperparameter in ./main/model_config.py then run ./main/model_training.py, it will involved model training and testing, which will eventually print the metric.

- hyper-parameter setting (in ./main/model_config.py)
  - Baseline
    - LESA: dataset_name, test_dataset_name = 'LESA'
    - CheckThatLab2022 1b: dataset_name, test_dataset_name = 'CLEF2022_1b'
  - Rewrite -> Model
    - LESA: dataset_name, test_dataset_name = 'LESA_explain_by_GPT'
    - CheckThatLab2022 1b: dataset_name, test_dataset_name = 'CLEF2022_1b_normalize_by_GPT'
  - Split to Sentence-Level -> Model
    - LESA: 
    - CheckThatLab2022 1b:
  - Rewrite -> Split to Sentence-Level -> Model:
    - LESA:
    - CheckThatLab2022 1b:
  - Training With Sentence-Level Dataset:  

# Code Structure & Description
- /main: all the code related to the using of BERT-Classifier model, from data preprocessing to predicting.
  - /predictions: the tsv file which contain the prediction and original label, for analyzing the result.
  - /weights: the fine-tuned weights use in model_pred.py.
  - /wrong_predictions_idx: the wrong predicted idx in different run, for analyzing the result.
  - chatGPT_api.py: wrap gpt api in a class, for calling chatGPT to rewrite the tweets.
  - data_loader.py: load dataset into the class PyTorch needed.
  - data_preprocess_methods.py: functions that perform the data-preprocessing proposed in this paper.
  - model_config.py: config all the hyper-parameter.
  - model_pred.py: use the saved weight to predict on target dataset.
  - model_training.py: the main script which begin a training-testing run.
  - srl_predictor.py: wrap AllenNLP srl API into class.
  - TweetNormalizer.py: the class that perform rule-based special symbols and url eliminating. 
- /others: others codes used for reformatting, cleaning or analyzing the data
  - augment_dataset.py: augment the article-level dataset with sentence-level dataset.
  - calculate_avg_length.py: calculating the avg words of each data.
  - calculate_special_symbol.py: calculating the average special symbols among claim and non-claim, respectively. For Data analyzing.
  - combine_rewrite_origin.py: augment the original dataset with the data rewriting by GPT.
  - combine_sentence_level_dataset.py: combine the sentence-level datasets into one.
  - compare_dataset.py: compare the dataset of CLEF CheckThatLab in 2021 and 2022.
  - GPT_zeroshot_response_conversion.py: convert the natural language answer 'yes', 'no' into 1, 0 and calculate the evaluation metric.
  - reformat_sentence_level_dataset.py: reformat sentence-level dataset into the format same as article-level dataset.
  
  
# Dataset Introduction
- /dataset: the original dataset.
  - /CheckThatLab2022-1b: released in  
  - /LESA: released in 
  - /other-sentence-level
    - /MT, /OC, /PE, /VG, /WD, /WTP: released in 
    - /ClaimBuster: released in 
    - /NewsClaims: released in 
- /preprocess_datasets: the dataset after undergoing the preprocessing proposed in this paper.
  - preprocess_datasets_GPT: the dataset after undergoing the rewrite of GPT.
  - preprocess_datasets_sentence: the dataset after undergoing split tweets into sentence-level using rule-based method.
  - preprocess_datasets_SRL: the dataset after undergoing split tweets into sentence-level using SRL.