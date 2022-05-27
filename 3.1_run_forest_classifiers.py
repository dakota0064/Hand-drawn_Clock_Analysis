from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
import numpy as np
import pandas as pd
import os
from itertools import chain, combinations
from utils.manual_feature_prediction_utils import apply_single_threshold, get_accuracy, get_confusion

''' Runs a Random Forest classifier for every possible combination of the full feature set.

    (Caveat - the normalize function was defined by visually inspecting the results in 2.0_feature_breakdown.ipynb.
    The intention was to make each feature as monotonic as possible with respect to its' label value so that 
    meaningful correlation values could be obtained. In practice this will likely need to be redefined based on your own
    data IF YOU WANT CORRELATION VALUES. Since neither of the models are linear it shouldn't have a significant impact
    on predictive power.)

    data_file = the saved feature data csv file computed from 1.0_compute_clock_features.py
    save_file = filename to save the normalized feature data

    junk_threshold = throws away any images/rows where LeftoverInk is greater than this value
                    (used as quality control for the automated feature calculation)
'''

# -------------------------------------------------------------------------------------------------------------------- #

data_file = "data/feature_data.csv"
save_file = "data/forest_normalized_feature_data.csv"

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

    rows_list = []
    # Test all feature combos
    feature_set = columns
    model = RandomForestClassifier(class_weight="balanced_subsample")

    data = df[list(feature_set)].to_numpy()
    train_data = data[:split_point]
    test_data = data[split_point:]

    model.fit(train_data, train_labels.ravel())

    predictions = model.predict(test_data)
    if feature_set == ["Circularity", "RemovedPoints", "CenterDeviation"]:
        df["Forest_Contour"] = model.predict_proba(data)[:, 1]

    if feature_set == ["HandsAngle", "DensityRatio", "BBRatio", "LengthRatio", "IntersectDistance", "NumComponents"]:
        df["Forest_Hands"] = model.predict_proba(data)[:, 1]

    if feature_set == ["DigitRadiusMean", "DigitRadiusStd", "DigitAngleMean", "DigitAngleStd", "DigitAreaMean", "DigitAreaStd",
               "ExtraDigits", "MissingDigits"]:
        df["Forest_Digits"] = model.predict_proba(data)[:, 1]

    dummy_thresh = 0
    best_f1 = 0
    best_thresh = 0

    for i in range(100):
        threshed_predictions = apply_single_threshold(predictions, dummy_thresh)
        f1 = f1_score(test_labels, threshed_predictions, pos_label=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = dummy_thresh

        dummy_thresh += 0.01

    threshed_predictions = apply_single_threshold(predictions, best_thresh)
    accuracy = get_accuracy(test_labels, threshed_predictions)
    _, _, _, fnr = get_confusion(test_labels, threshed_predictions)

    average_precision = average_precision_score(test_labels, predictions, pos_label=0)
    auc = roc_auc_score(test_labels, predictions)
    rows_list.append({"Features": str(feature_set),
                      "F1 Score": best_f1,
                      "Accuracy": accuracy,
                      "FNR": fnr,
                      "Average Precision": average_precision,
                      "AUC Score": auc,
                      "Threshold": best_thresh})

    # Random guessing
    predictions = np.random.randint(2, size=len(test_labels))
    dummy_thresh = 0
    best_f1 = 0
    best_thresh = 0

    for i in range(100):
        threshed_predictions = apply_single_threshold(predictions, dummy_thresh)
        f1 = f1_score(test_labels, threshed_predictions, pos_label=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = dummy_thresh

        dummy_thresh += 0.01

    threshed_predictions = apply_single_threshold(predictions, best_thresh)
    accuracy = get_accuracy(test_labels, threshed_predictions)
    _, _, _, fnr = get_confusion(test_labels, threshed_predictions)

    average_precision = average_precision_score(test_labels, predictions, pos_label=0)
    auc = roc_auc_score(test_labels, predictions)
    rows_list.append({"Features": "Random Guessing",
                      "F1 Score": best_f1,
                      "Accuracy": accuracy,
                      "FNR": fnr,
                      "Average Precision": average_precision,
                      "AUC Score": auc,
                      "Threshold": best_thresh})

    # Always guessing majority class
    predictions = np.ones((len(test_labels)))
    dummy_thresh = 0
    best_f1 = 0
    best_thresh = 0

    for i in range(100):
        threshed_predictions = apply_single_threshold(predictions, dummy_thresh)
        f1 = f1_score(test_labels, threshed_predictions, pos_label=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = dummy_thresh

        dummy_thresh += 0.01

    threshed_predictions = apply_single_threshold(predictions, best_thresh)
    accuracy = get_accuracy(test_labels, threshed_predictions)
    _, _, _, fnr = get_confusion(test_labels, threshed_predictions)

    average_precision = average_precision_score(test_labels, predictions, pos_label=0)
    auc = roc_auc_score(test_labels, predictions)
    rows_list.append({"Features": "Assuming Majority",
                      "F1 Score": best_f1,
                      "Accuracy": accuracy,
                      "FNR": fnr,
                      "Average Precision": average_precision,
                      "AUC Score": auc,
                      "Threshold": best_thresh})

    return pd.DataFrame(rows_list)


########################################################################################################################

if __name__ == '__main__':

    # Ensure the results directory exists
    if not os.path.isdir("results"):
        os.mkdir("results")

    if not os.path.isdir("results/tables"):
        os.mkdir("results/tables")

    df = pd.read_csv(data_file)
    df = normalize(df)
    print(len(df))
    df.drop(df[df["LeftoverInk"] > junk_threshold].index, inplace=True)
    df = df.dropna(subset=["ClockContour", "ClockHands", "ClockNumbers", "Circularity", "RemovedPoints", "CenterDeviation",
                      "HandsAngle", "DensityRatio", "BBRatio", "LengthRatio", "IntersectDistance", "NumComponents",
                      "DigitRadiusMean", "DigitRadiusStd", "DigitAngleMean", "DigitAngleStd", "DigitAreaMean",
                      "DigitAreaStd", "ExtraDigits", "MissingDigits"
                      ])
    # Drop significantly uncircular contours, assume they are mistakes.
    df.drop(df[df["Circularity"] < 0.8].index, inplace=True)

    print(len(df))

    columns = ["Circularity", "RemovedPoints", "CenterDeviation"]
    results_df = run_classifiers(df, columns, "ClockContour")
    results_df.to_csv("results/tables/forest_contour_results.csv")

    columns = ["HandsAngle", "DensityRatio", "BBRatio", "LengthRatio", "IntersectDistance", "NumComponents"]
    results_df = run_classifiers(df, columns, "ClockHands")
    results_df.to_csv("results/tables/forest_hands_results.csv")

    columns = ["DigitRadiusMean", "DigitRadiusStd", "DigitAngleMean", "DigitAngleStd", "DigitAreaMean", "DigitAreaStd",
               "ExtraDigits", "MissingDigits"]
    results_df = run_classifiers(df, columns, "ClockNumbers")
    results_df.to_csv("results/tables/forest_digit_results.csv")

    df.to_csv(save_file)