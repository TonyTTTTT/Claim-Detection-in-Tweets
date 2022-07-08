# CheckThat! Task 1 Turkish Datasets

The original dataset for Turkish does not have a "claim" column while  datasets for other languages have it. This difference might cause problems when using evaluation script. Therefore, we provide two versions of our data. Files ending with "_noclaim.tsv" has 5 columns including topic_id, tweet_id, tweet_url, tweet_text, and check_worthiness.  The other files has an additional column named "claim". However, you should *ignore* the values in this column (all of them are -1).
