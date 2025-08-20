import numpy as np
import pandas as pd
import shutil
import os

AUTOMORPH_DATA = os.getenv("AUTOMORPH_DATA", "..")

result_Eyepacs = f"{AUTOMORPH_DATA}/Results/M1/results_ensemble.csv"

result_Eyepacs_ = pd.read_csv(result_Eyepacs)

Eyepacs_pre = result_Eyepacs_["Prediction"]
Eyepacs_bad_mean = result_Eyepacs_["softmax_bad"]
Eyepacs_usable_sd = result_Eyepacs_["usable_sd"]
name_list = result_Eyepacs_["Name"]

Eye_good = 0
Eye_bad = 1

for i in range(len(name_list)):
    if Eyepacs_pre[i] == 0:
        Eye_good += 1

        result_Eyepacs_.loc[i, "quality"] = "good"
    elif (Eyepacs_pre[i] == 1) and (Eyepacs_bad_mean[i] < 0.25):
        # elif (Eyepacs_pre[i]==1) and (Eyepacs_bad_mean[i]<0.25) and (Eyepacs_usable_sd[i]<0.1):
        Eye_good += 1
        result_Eyepacs_.loc[i, "quality"] = "good"
    else:
        Eye_bad += 1
        result_Eyepacs_.loc[i, "quality"] = "bad"


print("Gradable cases by EyePACS_QA is {} ".format(Eye_good))
print("Ungradable cases by EyePACS_QA is {} ".format(Eye_bad))

result_Eyepacs_.to_csv(result_Eyepacs, index=False)
