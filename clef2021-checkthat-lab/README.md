# CLEF2021--CheckThat-Lab
This repository contains the _dataset_, _format checker, scorer and baselines_ for the [CLEF2021-CheckThat! Lab](https://sites.google.com/view/clef2021-checkthat). 
The shared task focuses on detecting check-worthy claims, previously fact-checked claims, and fake news. 
The CLEF2021-CheckThat! Lab has three main tasks each offered in variaty of languages:
  - [Task 1: Check-Worthiness Estimation](task1)
    - Subtask 1A: Check-Worthiness of Tweets
      - **Arabic**
      - **Bulgarian**
      - **English**
      - **Spanish**
      - **Turkish**
    - Subtask 1B: Check-Worthiness of Debates/Speeches
      - **English**
  - [Task 2: Detecting Previously Fact-Checked Claims](task2)
    - Subtask 2A: Detect Previously Fact-Checked Claims in Tweets
      - **Arabic**
      - **English**
    - Subtask 2B: Detect Previously Fact-Checked Claims in Political Debates/Speeches
      - **English**
  - [Task 3: Fake News Detection](task3)
    - Subtask 3A: Multi-Class Fake News Detection of News Articles
      - **English**
    - Subtask 3B: Topical Domain Classification of News Articles
      - **English**

# Leaderboard 

## Task 1
Kindly find the leaderboard released in this google sheet, [link](https://tinyurl.com/kfmawuke). you can find in the tab labeled "Task 1".

## Task 2
Kindly find the leaderboard released in this google sheet, [link](https://tinyurl.com/kfmawuke). you can find in the tab labeled "Task 2".


# Submission sites
## Task 1

Submission Guidelines:
- Make sure that you create one account for each team, and submit it through one account only. 
- The last file submitted to the leaderboard will be considered as the final submission. 
- The output file has to have a `.tsv` extension; otherwise, you will get an error on the leaderboard.
- You need to write a small description of the model submitted in a `.txt` file. 
- You have to zip the tsv, `zip submission.zip path_to_tsv_pred_file_1.tsv path_to_tsv_pred_file_2.tsv ... path_to_tsv_pred_file_n.tsv model_description.txt` and submit it through the codalab page. 

All leaderboard for dev and test data can be found here, https://competitions.codalab.org/competitions/30853. 

**NOTE**: The leaderboard for the Spanish test data is found in a separate leaderboard, https://competitions.codalab.org/competitions/31262.

## Task 2

Submission Guidelines:
- Make sure that you create one account for each team, and submit it through one account only. 
- The last file submitted to the leaderboard will be considered as the final submission. 
- The output file has to have a `.tsv` extension; otherwise, you will get an error on the leaderboard.
- You need to write a small description of the model submitted in a `.txt` file. 
- You have to zip the tsv, `zip submission.zip path_to_tsv_file.tsv model_description.txt` and submit it through the codalab page. 

All leaderboards for dev and test data can be found here, https://competitions.codalab.org/competitions/30949. 

# Licensing

These datasets are free for general research use.

# Credits

Lab Organizers:

Task website: https://sites.google.com/view/clef2021-checkthat

Contact:   clef-factcheck@googlegroups.com

# Citation

You can find the overview paper on the CLEF2021-CheckThat! Lab in the paper, "The CLEF-2021 CheckThat! Lab on Detecting Check-Worthy Claims, Previously Fact-Checked Claims, and Fake News".

```
@InProceedings{​​​​​​​CheckThat:ECIR2021,
  author    = {​​​​​​​Preslav Nakov and
               Da San Martino, Giovanni and
               Tamer Elsayed and
               Alberto Barr{​​​​​​​\'{​​​​​​​o}​​​​​​​}​​​​​​​n{​​​​​​​-}​​​​​​​Cede{​​​​​​​\~{​​​​​​​n}​​​​​​​}​​​​​​​o and
               Rub\'{​​​​​​​e}​​​​​​​n M\'{​​​​​​​i}​​​​​​​guez and
               Shaden Shaar and
               Firoj Alam and
               Fatima Haouari and
               Maram Hasanain and
               Nikolay Babulkov and
               Alex Nikolov and
               Shahi, Gautam Kishore and
               Struß, Julia Maria and
               Thomas Mandl}​​​​​​​,
  title     = {​​​​​​​The {​​​​​​​CLEF}​​​​​​​-2021 {​​​​​​​CheckThat}​​​​​​​! Lab on Detecting Check-Worthy Claims, Previously Fact-Checked Claims, and Fake News}​​​​​​​,
    booktitle = {​​​​​​​Proceedings of the 43rd European Conference on Information Retrieval}​​​​​​​,
    series = {​​​​​​​ECIR~'21}​​​​​​​,
    pages = {​​​​​​​639--649}​​​​​​​,
    address   = {​​​​​​​Lucca, Italy}​​​​​​​,
    month     = {​​​​​​​March}​​​​​​​,
    year      = {​​​​​​​2021}​​​​​​​,
    url = {​​​​​​​https://link.springer.com/chapter/10.1007/978-3-030-72240-1_75}​​​​​​​​
}​​​​​​​

```
