# Task 3: Fake News Detection

This repository contains the _dataset_, _format checker, scorer and baselines_ for the [CLEF2022-CheckThat! Task 3](https://sites.google.com/view/clef2022-checkthat/tasks/task-3-fake-news-detection).
Given the text of a news article, determine whether the claims made in the article are true, partially true, false or other (e.g., claims in dispute) and also detect the topical domain of the article. This task will run in English and German.

This file contains the basic information regarding the CLEF2022-CheckThat! Task 3 on check-worthiness on tweets provided for the CLEF2022-CheckThat! Lab on "Fighting the COVID-19 Infodemic and Fake News Detection".

<!-- The current version (?) corresponds to the release of the first batch of the training data set.
The test set is released with the current version. -->


## Data-sharing Agreement

The data in the research collection provided for CheckThat! 2022 task 3 may only be used for research purposes. Portions of the data are copyrighted and have commercial value as data, so you must be careful to use it only for research purposes. Due to these restrictions, the CheckThat! task-3 collection is not open data. Please fill the Data Sharing Agreement at <a href="https://tinyurl.com/4yx2xs5u"> Data Sharing Agreement</a>.

**Data is available at**
 https://zenodo.org/record/5775511


__Table of contents:__
- [Evaluation Results](#evaluation-results)
- [List of Versions](#list-of-versions)
- [Contents of the Task 3 Directory](#contents-of-the-repository)
- [Input Data Format](#input-data-format)
	- [Task 3: Multi-Class Fake News Detection of News Articles](#Subtask-3A-Multi-Class-Fake-News-Detection-of-News-Articles)
- [Output Data Format](#output-data-format)
	- [Task 3: Multi-Class Fake News Detection of News Articles](#Multi-Class-Fake-News-Detection-of-News-Articles)
- [Format Checkers](#format-checkers)
	- [Task 3: Multi-Class Fake News Detection of News Articles](#Subtask-3A-Multi-Class-Fake-News-Detection-of-News-Articles-2)
- [Scorers](#scorers)
	- [Task 3: Multi-Class Fake News Detection of News Articles](#Subtask-3A-Multi-Class-Fake-News-Detection-of-News-Articles-3)
- [Evaluation Metrics](#evaluation-metrics)
- [Baselines](#baselines)
	- [Task 3: Multi-Class Fake News Detection of News Articles](#Subtask-3A-Multi-Class-Fake-News-Detection-of-News-Articles-4)
- [Credits](#credits)

## Evaluation Results

TBA

## List of Versions

- **Task 3--English-v1.0 [2022/24/03]** - data for task 3 is released.

## Contents of the Task 3 Directory
We provide the following files:

- Main folder: [data](./data)
  - subfolder: Task 3--english
- Main folder: [baseline](./baseline)<br/>
- 	Contains scripts provided for baseline models of the tasks
- Main folder: [evaluation](./format_checker)<br/>
- 	Contains scripts provided to evaluate submission file
- [README.md](./README.md) <br/>
- 	This file!



# Input Data Format

The data will be provided in the format of Id, title, text, rating, domain the description of column are as follows:

## Task 3
```
- public_id- Unique indetifier of the news article
- title- Title of the news article
- text- Text mentioned inside the news article
- our rating - class of the news article as false, partially false, true, other
```

# Multi-Class Fake News Detection of News Articles

Multi-class fake news detection of news articles (English): Sub-task A would be the detection of fake news designed as a four-class classification problem. The training data will be released in batches and will be roughly about 900 articles with the respective label. Given the text of a news article, determine whether the main claim made in the article is true, partially true, false, or other.

### Cross-Lingual Task

Along with the multi-class task for the English language, we have introduced a task for low resourced language. We will provide the data for dev and test in the German language. The idea of the task is to use the English data and the concept of transfer to build a classification model for the German language.

# Output Data Format

### Task 3: Multi-Class Fake News Detection of News Articles

We need the output file in the format of public_id, predicted_rating.


# Format Checkers

#### Task 3: Multi-Class Fake News Detection of News Articles

Task 3

public_id- Unique identifier of the news article
predicted_rating- predicted class
Sample File

```
public_id, predicted_rating
1, false
2, true
```

### Multi-Class Fake News Detection of News Articles

# Evaluation Metrics

This task is evaluated as a classification task. We will use the F1-macro measure for the ranking of teams. There is a limit of 5 runs (total and not per day), and only one person from a team is allowed to submit runs.

Submission Link: Coming Soon

Evaluation File task3/evaluation/CLEF_-_CheckThat__Task3ab_-_Evaluation.txt

# Baselines

### Task 3: Multi-Class Fake News Detection of News Articles

For this task, we have created a baseline system. The baseline system can be found at https://zenodo.org/record/6362498

## Scorers


## Credits

Task 3 Organizers:

- Thomas Mandl, University of Hildesheim, Germany
- Julia Maria Struß, University of Applied Sciences Potsdam, Germany
- Gautam Kishore Shahi, University of Duisburg-Essen, Germany

Task website: https://sites.google.com/view/clef2022-checkthat/tasks/task-3-fake-news-detection

Contact:   clef-factcheck@googlegroups.com
