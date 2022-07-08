# Task 1: Check-Worthiness Estimation

This repository contains the _dataset_, _format checker, scorer and baselines_ for the [CLEF2021-CheckThat! Task 1](https://sites.google.com/view/clef2021-checkthat/tasks/task-1-check-worthiness-estimation). 
The task consists in ranking a stream of tweets according to their check-worthiness. 

````
FCPD corpus for the CLEF-2021 LAB on "Detecting Check-Worthy Claims, Previously Fact-Checked Claims, and Fake News"
````

This file contains the basic information regarding the CLEF2021-CheckThat! Task 1
on check-worthiness on tweets provided for the CLEF2021-CheckThat! Lab
on "Automatic Detecting Check-Worthy Claims, Previously Fact-Checked Claims, and Fake News".
The current version are listed bellow corresponds to the release of the training and dev data sets.



__Table of contents:__
- [Evaluation Results](#evaluation-results)
- [List of Versions](#list-of-versions)
- [Contents of the Task 1 Directory](#contents-of-the-repository)
- [Input Data Format](#input-data-format)
	- [Subtask 1A: Check-Worthiness of Tweets](#subtask-1a-check-worthiness-of-tweets)
	- [Subtask 1B: Check-Worthiness of Debates/Speeches](#subtask-1b-check-worthiness-of-debatesspeeches)
- [Output Data Format](#output-data-format)
	- [Subtask 1A: Check-Worthiness of Tweets](#subtask-1a-check-worthiness-of-tweets-1)
	- [Subtask 1B: Check-Worthiness of Debates/Speeches](#subtask-1b-check-worthiness-of-debatesspeeches-1)
- [Format Checkers](#format-checkers)
	- [Subtask 1A: Check-Worthiness of Tweets](#subtask-1a-check-worthiness-of-tweets-2)
	- [Subtask 1B: Check-Worthiness of Debates/Speeches](#subtask-1b-check-worthiness-of-debatesspeeches-2)
- [Scorers](#scorers)
	- [Subtask 1A: Check-Worthiness of Tweets](#subtask-1a-check-worthiness-of-tweets-3)
	- [Subtask 1B: Check-Worthiness of Debates/Speeches](#subtask-1b-check-worthiness-of-debatesspeeches-3)
- [Evaluation Metrics](#evaluation-metrics)
- [Baselines](#baselines)
	- [Subtask 1A: Check-Worthiness of Tweets](#subtask-1a-check-worthiness-of-tweets-4)
	- [Subtask 1B: Check-Worthiness of Debates/Speeches](#subtask-1b-check-worthiness-of-debatesspeeches-4)
- [Credits](#credits)

## Evaluation Results

Kindly find the leaderboard released in this google sheet, [link](https://tinyurl.com/kfmawuke). you can find in the tab labeled "Task 1".

Submission Guidelines:
- Make sure that you create one account for each team, and submit it through one account only. 
- The last file submitted to the leaderboard will be considered as the final submission. 
- The output file has to have a `.tsv` extension; otherwise, you will get an error on the leaderboard.
- You need to write a small description of the model submitted in a `.txt` file. 
- You have to zip the tsv, `zip submission.zip path_to_tsv_pred_file_1.tsv path_to_tsv_pred_file_2.tsv ... path_to_tsv_pred_file_n.tsv model_description.txt` and submit it through the codalab page. 

All leaderboard for dev and test data can be found here, https://competitions.codalab.org/competitions/30853. 

*NOTE*: The leaderboard for the Spanish test data is found in a separate leaderboard, https://competitions.codalab.org/competitions/31262.


## List of Versions

* __subtask-1a--bulgarian-v1.0 [2021/02/23]__ - Training/Dev data for subtask-1a--Bukgarian released. Containing 2594 tweets for train and 372 tweets for dev.
* __subtask-1b--english-v1.0 [2021/02/07]__ - Training/Dev data for subtask-1b--English released. Containing 50 debates/speeches for train and 10 debates/speeches for dev.
* __subtask-1a--english-v1.0 [2021/02/07]__ - Training/Dev data for subtask-1a--English released. Containing 822 tweets for train and 140 tweets for dev.
* __subtask-1a--spanish-v1.0 [2021/02/04]__ - Training/Dev data for subtask-1a--Spanish released. Containing 2496 tweets for train and 1248 tweets for dev.
* __subtask-1a--turkish-v1.0 [2021/03/04]__ - Training/Dev data for subtask-1a--Turkish released. Containing 1899 tweets for train and 388 tweets for dev.

## Contents of the Task 1 Directory
We provide the following files:

* Main folder: [data](./data)
  * Subfolder: [subtask-1A--english](./data/subtask-1a--english)
  	* [data.zip](./data/subtask-1a--english/v1.zip) </br>
  	Contains train data released with the tweets used, the tweets JSON objects, and the labels assigned. 
  	Also it contains dev data released with the tweets used, the tweets JSON objects, and the labels assigned.
  * Subfolder: [subtask-1A--bulgarian](./data/subtask-1a--bulgarian)
  	* [data.zip](./data/subtask-1a--bulgarian/v1.zip) </br>
  	Contains train data released with the tweets used, the tweets JSON objects, and the labels assigned. 
  	Also it contains dev data released with the tweets used, the tweets JSON objects, and the labels assigned. 
  * Subfolder: [subtask-1A--spanish](./data/subtask-1a--english)
  	* [data.zip](https://github.com/Newtral-Tech/clef2021-checkthat/tree/main/data) </br>
  	Contains train data released with the tweets used, the tweets JSON objects, and the labels assigned. 
  	Also it contains dev data released with the tweets used, the tweets JSON objects, and the labels assigned. </br>
  * Subfolder: [subtask-1A--turkish](./data/subtask-1a--turkish)
  	* [data.zip](./data/subtask-1a--turkish/v1.zip) </br>
  	Contains train data released with the tweets used and the labels assigned. 
  	Also it contains dev data released with the tweets used and the labels assigned. </br>

	**NOTE:** The data for the spanish language is released in a seperate directory. Kindly find the link [here](https://github.com/Newtral-Tech/clef2021-checkthat/tree/main/data). 
  * Subfolder: [subtask-1B--english](./data/subtask-1b--english)
  	* [data.zip](./data/subtask-1b--english/v1.zip) </br>
  	Contains train data released the debates/speeches used, and the labels assigned. 
  	Also it contains dev data released the debates/speeches used, and the labels assigned. 
* Main folder: [baselines](./baselines)<br/>
	Contains scripts provided for baseline models of the tasks
* Main folder: [formet_checker](./format_checker)<br/>
	Contains scripts provided to check format of the submission file
* Main folder: [scorer](./scorer)<br/>
	Contains scripts provided to score output of the model when provided with label (i.e., dev set).

* [README.md](./README.md) <br/>
	This file!


## Input Data Format

### Subtask 1A: Check-Worthiness of Tweets
All languages (**Arabic**, **Bulgarian**, **English**, and **Spanish**) in subtask-1A have the same data format, which includes train and dev files.

For both **Train** and **Dev** datasets we provide 2 files each. We give a JSON file containing the tweet objects retrieved from the Twitter API, and a TSV file containing the tweets and it's annotations. 

The datasets are TAB separated text files. The text encoding is UTF-8. 
A row of the file has the following format:

> topic_id <TAB> tweet_id <TAB> tweet_url <TAB> tweet_text <TAB> claim <TAB> check_worthiness

Where: <br>
* topic_id: unique ID for the topic the tweet is about <br/>
* tweet_id: Tweet ID for a given tweets given by Twitter <br/>
* tweet_url: URL to the given tweet <br/>
* tweet_text: content of the tweet <br/>
* claim: 1 if the tweet is a claim; 0 otherwise <br/>
* check_worthiness: 1 if the tweet is worth fact checking; 0 otherwise <br/>

Example:
> covid-19	1235648554338791427	https://twitter.com/A6Asap/status/1235648554338791427	COVID-19 health advice⚠️ https://t.co/XsSAo52Smu	0	0 <br/>
> covid-19	1235287380292235264	https://twitter.com/ItsCeliaAu/status/1235287380292235264	There's not a single confirmed case of an Asian infected in NYC. Stop discriminating cause the virus definitely doesn't. #racist #coronavirus https://t.co/Wt1NPOuQdy	1	0 <br/>
> covid-19	1236020820947931136	https://twitter.com/ddale8/status/1236020820947931136	Epidemiologist Marc Lipsitch, director of Harvard's Center for Communicable Disease Dynamics: “In the US it is the opposite of contained.' https://t.co/IPAPagz4Vs	1	1 <br/>
> ... <br/>

Note that the gold labels for the task are the ones in the *check_worthiness* column 

### Subtask 1B: Check-Worthiness of Debates/Speeches
This task is only given in **English**. The input files are TAB-separated CSV files with four fields:

> line_number <TAB> speaker <TAB> text <TAB> label

Where: <br>
* line_number: the line number (starting from 1) <br/>
* speaker: the person speaking (a candidate, the moderator, or "SYSTEM"; the latter is used for the audience reaction) <br/>
* text: a sentence that the speaker said <br/>
* label: 1 if this sentence is to be fact-checked, and 0 otherwise 

The text encoding is UTF-8.

Example:

>  ... <br/>
>  65  TRUMP So we're losing our good jobs, so many of them. 0 <br/>
>  66  TRUMP When you look at what's happening in Mexico, a friend of mine who builds plants said it's the eighth wonder of the world. 0 <br/>
>  67  TRUMP They're building some of the biggest plants anywhere in the world, some of the most sophisticated, some of the best plants. 0 <br/>
>  68  TRUMP With the United States, as he said, not so much.  0 <br/>
>  69  TRUMP So Ford is leaving. 1 <br/> 
>  70  TRUMP You see that, their small car division leaving. 1 <br/>
>  71  TRUMP Thousands of jobs leaving Michigan, leaving Ohio. 1 <br/>
>  72  TRUMP They're all leaving.  0 <br/>
>  ...


## Output Data Format

### Subtask 1A: Check-Worthiness of Tweets
All languages (**Arabic**, **Bulgarian**, **English**, **Spanish** and **Turkish**) in subtask-1A have the same data format, which includes submission files.

For this task, the expected results file is a list of tweets with the estimated score for check-worthiness. Each row contains four TAB separated fields:

> topic_id <TAB> tweet_id <TAB> score <TAB> run_id 

Where: <br>
* topic_id: unique ID for the topic the tweet is about given in the test dataset file <br/>
* tweet_id: Tweet ID for a given tweets given by Twitter given in the test dataset file<br/>
* score: score given by the participant's model about whether a claim is worth fact checking or not <br/>
* run_id: string identifier used by participants. <br/>

Example:
> covid-19	1235648554338791427	0.39  Model_1<br/>
> covid-19	1235287380292235264	0.61  Model_1<br/>
> covid-19	1236020820947931136	0.76  Model_1<br/>
> ... <br/>

### Subtask 1B: Check-Worthiness of Debates/Speeches

For this subtask, the expected results file is a list of claims with the estimated score for check-worthiness. 
    Each row contains two tab-separated fields:
>line_number <TAB> score

Where _line_number_ is the number of the claim in the debate and _score_ is a number, indicating the priority of the claim for fact-checking. For example:
>1	0.9056 <br/>
>2	0.6862 <br/>
>3	0.7665 <br/>
>4	0.9046 <br/>
>5	0.2598 <br/>
>6	0.6357 <br/>
>7	0.9049 <br/>
>8	0.8721 <br/>
>9	0.5729 <br/>
>10	0.1693 <br/>
>11	0.4115 <br/>
> ...

Your result file **MUST contain scores for all lines** of the input file.
Otherwise the scorer will return an error and no score will be computed. 


## Format Checkers

#### Subtask 1A: Check-Worthiness of Tweets
The checker for the subtask is located in the [format_checker](./format_checker) module of the project.
To launch the baseline script you need to install packages dependencies found in [requirement.txt](./requirement.txt) using the following:
> pip3 install -r requirements.txt <br/>

The format checker verifies that your generated results files complies with the expected format.
To launch it run: 
> python3 format_checker/main.py --subtask=1a --pred-files-path=<path_to_result_file_1 path_to_result_file_2 ... path_to_result_file_n> <br/>

or 
> python3 format_checker/subtask_1a.py --pred-files-path=<path_to_result_file_1 path_to_result_file_2 ... path_to_result_file_n> <br/>

`--pred-files-path` take a single string that contains a space separated list of file paths. The lists may be of arbitrary positive length (so even a single file path is OK) but their lengths must match.

__<path_to_result_file_n>__ is the path to the corresponding file with participants' predictions, which must follow the format, described in the [Output Data Format](#subtask-1a-check-worthiness-of-debatesspeeches-1) section.

Note that the checker can not verify whether the prediction files you submit contain all lines / claims), because it does not have access to the corresponding gold file.

#### Subtask 1B: Check-Worthiness of Debates/Speeches

The checker for the subtask is located in the [format_checker](./format_checker) module of the project.
To launch the baseline script you need to install packages dependencies found in [requirement.txt](./requirement.txt) using the following:
> pip3 install -r requirements.txt <br/>

The format checker verifies that your generated results files complies with the expected format.
To launch it run: 
> python3 format_checker/main.py --subtask=1b --pred-files-path=<path_to_result_file_1 path_to_result_file_2 ... path_to_result_file_n> <br/>

or 
> python3 format_checker/subtask_1b.py --pred-files-path=<path_to_result_file_1 path_to_result_file_2 ... path_to_result_file_n> <br/>

`--pred-files-path` take a single string that contains a space separated list of file paths. The lists may be of arbitrary positive length (so even a single file path is OK) but their lengths must match.

__<path_to_result_file_n>__ is the path to the corresponding file with participants' predictions for debate __n__, which must follow the format, described in the [Output Data Format](#subtask-1b-check-worthiness-of-debatesspeeches-1) section.

Note that the checker can not verify whether the prediction files you submit contain all lines / claims), because it does not have access to the corresponding gold file.

## Scorers

### Subtask 1A: Check-Worthiness of Tweets
The scorer for the subtask is located in the [scorer](./scorer) module of the project.
To launch the script you need to install packages dependencies found in [requirement.txt](./requirement.txt) using the following:
> pip3 install -r requirements.txt <br/>

Launch the scorer for the subtask as follows:
> python3 scorer/subtask_1a.py --gold-file-path=<path_gold_file> --pred-file-path=<predictions_file> <br/>

The scorer invokes the format checker for the subtask to verify the output is properly shaped.
It also handles checking if the provided predictions file contains all lines/tweets from the gold one.

### Subtask 1B: Check-Worthiness of Debates/Speeches

The scorer for the subtask is located in the [scorer](./scorer) module of the project.
To launch the script you need to install packages dependencies found in [requirement.txt](./requirement.txt) using the following:
> pip3 install -r requirements.txt <br/>

Launch the scorer for the subtask as follows:
> python3 scorer/subtask_1b.py --gold-files-path=<path_gold_file_1 path_gold_file_2 ... path_gold_file_n> --pred-files-path=<prediction_file_1 prediction_file_2 ... prediction_file_n> <br/>

Both `--gold-files-path` and `--pred-files-path` take a single string that contains a space separated list of file paths. The lists may be of arbitrary positive length (so even a single file path is OK) but their lengths must match.

__<path_gold_file_n>__ is the path to the file containing the gold annotations for debate __n__ and __<predictions_file_n>__ is the path to the corresponding file with participants' predictions for debate __n__, which must follow the format, described in the [Output Data Format](#subtask-1b-check-worthiness-of-debatesspeeches-1) section.

The scorer invokes the format checker for the task to verify the output is properly shaped.
It also handles checking if the provided predictions file contains all lines / claims from the gold one.

## Evaluation Metrics
Both subtasks \{Subtask 1A: Check-Worthiness of Tweets, Subtask 1B: Check-Worthiness of Debates/Speeches\} of all languages use the same evaluation metric. 
The subtasks are evaluated as ranking tasks. We will use Mean Average Precision (MAP) as the official evaluation measure and we will report reciprocal rank, and P@k for k∈ {1,3,5,10,20,30} as well. 


## Baselines

### Subtask 1A: Check-Worthiness of Tweets
The [baselines](./baselines) module contains a random and a simple ngram baseline for the task.
To launch the baseline script you need to install packages dependencies found in [requirement.txt](./requirement.txt) using the following:
> pip3 install -r requirements.txt <br/>

To launch the baseline script run the following:
> python3 baselines/subtask_1a.py --train-file-path=<path_to_your_training_data> --test-file-path=<path_to_your_test_data_to_be_evaluated> --lang=<language_of_the_sutask_1a><br/>

Both baselines will be trained on the training tweets and the performance of the model was was evaluated on the dev tweets.
The MAP score of both baselines are as follows:<br/>
| Model | English | Arabic | Spanish | Bulgarian | 
| :---: | :---: | :---: | :---: | :---: |
| Random Baseline | 0.4795 |  | 0.0806 | 0.2045 |
| Ngram Baseline  | 0.5916 |  | 0.4122 | 0.4729 |

### Subtask 1B: Check-Worthiness of Debates/Speeches

The [baselines](./baselines) module contains a random and a simple ngram baseline for the task.
To launch the baseline script you need to install packages dependencies found in [requirement.txt](./requirement.txt) using the following:
> pip3 install -r requirements.txt <br/>

To launch the baseline script run the following:
> python3 baselines/subtask_1b.py --train-files-path=<path_to_train_file_1 path_to_train_file_2 ... path_to_train_file_n> --test-files-path=<path_to_test_file_1 path_to_test_file_2 ... path_to_test_file_n> <br/>

Both baselines will be trained on the training debates/speeches and the performance of the model was was evaluated on the dev debates/speeches.
The MAP score of both baselines are as follows:<br/>
| Model | English | 
| :---: | :---: | 
| Random Baseline | 0.0352 | 
| Ngram Baseline  | 0.0707 |


## Credits

Task 1 Organizers: TBA

Task website: https://sites.google.com/view/clef2021-checkthat/tasks/task-1-check-worthiness-estimation

Contact:   clef-factcheck@googlegroups.com
