# -*- coding: utf-8 -*-
import re, os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from tkinter import Tk, filedialog
from IPython.display import Markdown

duplicates = ["1a46b040", "c1e124d5", "8209ca7b", "e2abfbe6", "39f5c37f",
              "47423bb6", "097c142a", "08bc59f4", "af3393ce", "a39d07df"]

"""## Gold Standard"""

root = Tk()
root.withdraw() #dont show base window
path_to_test_data = filedialog.askopenfile() #file selection dialog
print(path_to_test_data.name) #check, that the correct file has been selected

#depending on whether task a or task b testing data has been loaded
column_name_test = "our rating" if "3a" in path_to_test_data.name else "category"
column_name_system = "predicted_rating" if "3a" in path_to_test_data.name else "predicted_domain"

#read gold standard
#If you don't wan't to use a dialog, specify the path directly
test_data = pd.read_csv(path_to_test_data.name, encoding="UTF-8")
test_data.head() #quick check, if the data has been correctly read.

"""## Participants file"""

#read participant data 

#If you don't wan't to use a dialog, specify the path directly
path_to_system_predictions = filedialog.askopenfile() #file selection dialog
print(path_to_system_predictions.name) #check, if the correct file has been selected

if path_to_system_predictions.name.endswith("tsv"):
    system_predictions = pd.read_csv(path_to_system_predictions.name, encoding="utf-8", sep="\t") 
else:
    system_predictions = pd.read_csv(path_to_system_predictions.name, encoding="utf-8")

#Check, if the number of entries is correct
print(len(system_predictions))
system_predictions.head()

"""## Drop duplicates"""

#Only for 3A
system_predictions_no_duplicates = system_predictions[~system_predictions["public_id"].isin(duplicates)]
if(len(system_predictions) - len(system_predictions_no_duplicates)==10):
    print("Dropping duplicates was successful") 
    system_predictions = system_predictions_no_duplicates.reset_index(drop=True)
else:
    print("Dropping duplicates was NOT successful. Please mark for a later check.")
    print(f"system_predictions has {len(system_predictions)} entries and system_predictions_no_duplicates has {len(system_predictions_no_duplicates)} entries.")
    
print(f"system_predictions has {len(system_predictions)} entries")

"""## Check public ids"""

#check, if the id columns are identical
#there are some public ids, that are saved differently depending on the editor used to view the file 
#    test_data    system_predictions (example for public ids that might be problematic)
#    public_id    public_id
#86   080e0024       8E+025
#129  17334e19  1,7334E+023
#210  84231e18  8,4231E+022
#219  85100e77    8,51E+081
#271  04866616      4866616
#272  03963580      3963580
#363  02695016      2695016
#If you can verify that the ids are identical, than you can proceed with the evaluation. If not, please inspect further.
print(test_data["public_id"].isin(system_predictions["public_id"]).value_counts())
matched = pd.DataFrame(test_data["public_id"].eq(system_predictions["public_id"]))
if len(matched[matched["public_id"]==False]) == 0:
    display(Markdown("<span style='color: #ff0000'>everything is fine, please proceed with the evaluation.</span>".upper()))
else:
    try:
        print(pd.concat((
            test_data["public_id"].loc[
                test_data["public_id"].where(test_data["public_id"]==system_predictions["public_id"]).isna()], 
            system_predictions["public_id"].loc[
                system_predictions["public_id"].where(system_predictions["public_id"]==test_data["public_id"]).isna()]), axis =1))
    except ValueError as e:
        display(Markdown("<span style='color: #ff0000'>The number of entries in both files is different. An evaluation is not possible.</span>".upper())) 
        merged_dataframe = pd.merge(test_data["public_id"], system_predictions["public_id"], how="outer", indicator=True)
        print("entries missing in the participants file:\n", 
              merged_dataframe[merged_dataframe["_merge"]=="left_only"]["public_id"])
        print("\nentries missing in the systems file:\n", system_predictions["public_id"].loc[
        system_predictions["public_id"].isin(merged_dataframe[merged_dataframe["_merge"]=="right_only"]["public_id"])])

"""## Evaluation"""

try:
    classification_report_results = classification_report(test_data[column_name_test], system_predictions[column_name_system].str.lower().str.strip(), output_dict=True)
    eval_results = pd.DataFrame.from_dict(classification_report_results).transpose()
    participant_file = re.sub(".+/(\d+)/*[\w_-]*/([\w_-]+)(\.(tsv|csv))?", "\\1_\\2", path_to_system_predictions.name)

    eval_results.to_csv("classification_report_" + participant_file + ".csv")
    #pretty print
    print(classification_report(test_data[column_name_test], system_predictions[column_name_system].str.lower().str.strip(), digits=4))
except KeyError as e:
    print("The column names of the participant file don't match the required column names ('public_id, 'predicted_rating'):", 
          system_predictions.columns)

#Destroy the "invisible" root window
root.destroy()
