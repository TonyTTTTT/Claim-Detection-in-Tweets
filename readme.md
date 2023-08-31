# Usage
To reproduce the result, first setting the hyperparameter in ./main/model_config.py then run ./main/model_training.py, it will involved model training and testing, which will eventually print the metric.

### hyper-parameter setting (in ./main/model_config.py)
Each setting represents a scenario in Table 6.8 on page 57 of my paper. Please just change the parameter been mentioned.
- Baseline
  - CheckThatLab2022 1b: ```dataset_name = 'CLEF2022_1b', test_dataset_name = 'CLEF2022_1b'```
  - LESA: ```dataset_name = 'LESA', test_dataset_name = 'LESA'```
- Rewrite -> Model
  - CheckThatLab2022 1b: ```dataset_name= 'CLEF2022_1b_normalize_by_GPT', test_dataset_name = 'CLEF2022_1b_normalize_by_GPT'```
  - LESA: ```dataset_name = 'LESA_explain_by_GPT, test_dataset_name = 'LESA_explain_by_GPT'```
- Split to Sentence-Level -> Model: ```present in same run of Baseline``` 
- Rewrite -> Split to Sentence-Level -> Model: ```present in same run of Rewrite -> Model```
- Training With Sentence-Level Dataset: ```do_balancing = True, delete_dbquote = True```
  - Split to Sentence-Level -> Model
    - CheckThatLab2022 1b: ```dataset_name = 'sentence_level_CB_OC', test_dataset_name = 'CLEF2022_1b'```
    - LESA: dataset_name = ```'LESA_CB', test_dataset_name = 'LESA'```
  - Rewrite -> Model
    - CheckThatLab2022 1b: ```dataset_name = 'sentence_level_CB_OC', test_dataset_name = 'CLEF2022_1b_normalize_by_GPT'```
    - LESA: ```dataset_name = 'LESA_CB', test_dataset_name = 'LESA_explain_by_GPT'```


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
  - /CheckThatLab2022-1b: released in this [gitlab link](https://gitlab.com/checkthat_lab/clef2022-checkthat-lab/clef2022-checkthat-lab/-/tree/main/task1/data/subtasks-english) 
  - /LESA: released in this [github link](https://github.com/LCS2-IIITD/LESA-EACL-2021/tree/main/data)
  - /other-sentence-level
    - /MT, /OC, /PE, /VG, /WD, /WTP: released in this [github link](https://github.com/LCS2-IIITD/LESA-EACL-2021/tree/main/data)
    - /ClaimBuster: released in this [link](https://zenodo.org/record/3836810#.YwSJzHZByUl)
    - /NewsClaims: released in this [link](https://drive.google.com/file/d/1jlQ0kQLS0kLbrXIC1fh6oT2HsWppx5QT/view)
- /preprocess_datasets: the dataset after undergoing the preprocessing proposed in this paper.
  - preprocess_datasets_GPT: the dataset after undergoing the rewrite of GPT.
  - preprocess_datasets_sentence: the dataset after undergoing split tweets into sentence-level using rule-based method.
  - preprocess_datasets_SRL: the dataset after undergoing split tweets into sentence-level using SRL.