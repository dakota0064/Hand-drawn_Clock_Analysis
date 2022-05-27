from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from itertools import chain, combinations

''' Runs a Random Forest classifier using some subset of the features to predict some user-defined label.
    For example, can use this script to use all features to predict age, total MoCA score, etc.
    You will need to edit the main function as necessary to set your own feature subset and labels.

    (Caveat - the normalize function was defined by visually inspecting the results in 2.0_feature_breakdown.ipynb.
    The intention was to make each feature as monotonic as possible with respect to its' label value so that 
    meaningful correlation values could be obtained. In practice this will likely need to be redefined based on your own
    data IF YOU WANT CORRELATION VALUES. Since neither of the models are linear it shouldn't have a significant impact
    on predictive power.)

    data_file = the saved feature data csv file computed from 1.0_compute_clock_features.py
    save_file = filename to save the prediction results and normalized data

    junk_threshold = throws away any images/rows where LeftoverInk is greater than this value
                    (used as quality control for the automated feature calculation)
'''

# -------------------------------------------------------------------------------------------------------------------- #

data_file = "data/feature_data.csv"
save_file = "data/forest_moca.csv"

junk_threshold = 0.03

# -------------------------------------------------------------------------------------------------------------------- #

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

########################################################################################################################

def normalize(df):
    df["CenterDeviation"] = df["CenterDeviation"] / np.max(df["CenterDeviation"])
    df["HandsAngle"] = df["HandsAngle"] / np.max(df["HandsAngle"])
    df["HandsAngle"] = np.where(df["HandsAngle"] > 180, df["HandsAngle"] - 180, df["HandsAngle"])
    df["IntersectDistance"] = df["IntersectDistance"] / np.max(df["IntersectDistance"])
    df["NumComponents"] = df["NumComponents"] / np.max(df["NumComponents"])
    df["DigitAngleMean"] = df["DigitAngleMean"] / np.max(df["DigitAngleMean"])
    df["DigitAngleStd"] = df["DigitAngleStd"] / np.max(df["DigitAngleStd"])
    df["DigitAreaMean"] = df["DigitAreaMean"] / np.max(df["DigitAreaMean"])
    df["DigitAreaStd"] = df["DigitAreaStd"] / np.max(df["DigitAreaStd"])
    df["ExtraDigits"] = df["ExtraDigits"] / np.max(df["ExtraDigits"])
    df["MissingDigits"] = df["MissingDigits"] / np.max(df["MissingDigits"])

    # Normalization based on feature analysis, to make monotonic
    df["DensityRatio"] = np.abs(0.37612489 - df["DensityRatio"])
    df["LengthRatio"] = np.abs(0.57538314 - df["LengthRatio"])
    df["BBRatio"] = np.abs(0.48631634 - df["BBRatio"])
    df["ExtraDigits"] = np.abs(0.30434783 - df["ExtraDigits"])
    df["DigitRadiusMean"] = np.abs(0.77046706 - df["DigitRadiusMean"])

    # Fixing "mesa" shaped behavior with 0 values
    df["HandsAngle"] = np.where(df["HandsAngle"] < 0.15, 1.0, df["HandsAngle"])
    df["NumComponents"] = np.where(df["NumComponents"] == 0.0, 1.0, df["NumComponents"])
    df["DigitAngleMean"] = np.where(df["DigitAngleMean"] < 0.05, 1.0, df["DigitAngleMean"])
    df["DigitAreaMean"] = np.where(df["DigitAreaMean"] == 0.0, 1.0, df["DigitAreaMean"])
    df["DigitAreaStd"] = np.where(df["DigitAreaStd"] == 0.0, 1.0, df["DigitAreaStd"])

    return df

########################################################################################################################

def run_classifiers(df, columns, label_name):

    labels = df[[label_name]].to_numpy()
    split_point = int(len(labels) * .8)

    train_labels = labels[:split_point]
    test_labels = labels[split_point:]

    model = RandomForestRegressor()#class_weight="balanced")
    feature_set = columns
    #print(feature_set)

    data = df[list(feature_set)].to_numpy()
    train_data = data[:split_point]
    test_data = data[split_point:]

    model.fit(train_data, train_labels.ravel())

    predictions = model.predict(data)
    print(model.score(test_data, test_labels))
    df[label_name + "_pred"] = predictions

    return df


########################################################################################################################

if __name__ == '__main__':

    df = pd.read_csv(data_file)
    #df = df.sample(frac=1).reset_index(drop=True)
    #df.to_csv(data_file)
    df = normalize(df)

    # Columns to drop null values from
    drop_columns = ["Circularity", "RemovedPoints", "CenterDeviation",
               "HandsAngle", "DensityRatio", "BBRatio", "LengthRatio", "IntersectDistance", "NumComponents",
               "DigitRadiusMean", "DigitRadiusStd", "DigitAngleMean", "DigitAngleStd", "DigitAreaMean", "DigitAreaStd",
               "ExtraDigits", "MissingDigits", "LeftoverInk", "PenPressure",
               "ClockContour", "ClockHands", "ClockNumbers", "Score"]

    # Columns being used for prediction, don't include labels in here!
    columns = ["Circularity", "RemovedPoints", "CenterDeviation",
               "HandsAngle", "DensityRatio", "BBRatio", "LengthRatio", "IntersectDistance", "NumComponents",
               "DigitRadiusMean", "DigitRadiusStd", "DigitAngleMean", "DigitAngleStd", "DigitAreaMean", "DigitAreaStd",
               "ExtraDigits", "MissingDigits", "LeftoverInk", "PenPressure"]

    # Drop significantly uncircular contours, assume they are mistakes in our automated evaluation.
    df.drop(df[df["Circularity"] < 0.8].index, inplace=True)

    print(len(df))
    df.drop(df[df["LeftoverInk"] > junk_threshold].index, inplace=True)
    df.dropna(subset=drop_columns, inplace=True)

    print(len(df))
    results_df = run_classifiers(df, columns, "Score")
    results_df.to_csv(save_file)