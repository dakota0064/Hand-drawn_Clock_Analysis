''' Utility functions for measuring the efficacy of features on various predictive processes '''

def get_accuracy(labels, predictions):
    correct = 0
    for i in range(len(labels)):
        if labels[i] == predictions[i]:
            correct += 1

    return correct / len(labels)

#-----------------------------------------------------------------------------------------------------------------------


def get_confusion(labels, predictions):
    fps = 0
    fns = 0
    tps = 0
    tns = 0

    for i in range(len(labels)):
        if labels[i] == 0 and predictions[i] == 0:
            tps += 1
        if labels[i] == 0 and predictions[i] == 1:
            fns += 1
        if labels[i] == 1 and predictions[i] == 1:
            tns += 1
        if labels[i] == 1 and predictions[i] == 0:
            fps += 1

    if tps + fns == 0:
        tpr = 0
        fnr = 0
    else:
        tpr = tps/(tps + fns)
        fnr = fns/(fns + tps)

    if fps + tns == 0:
        fpr = 0
        tnr = 0
    else:
        fpr = fps/(fps + tns)
        tnr = tns/(tns + fps)

    return tpr, fpr, tnr, fnr

#-----------------------------------------------------------------------------------------------------------------------


def get_total_confusion(labels, predictions):
    # type 1 is false positive, type 2 is false negative
    type1 = 0
    type2 = 0
    for i in range(len(labels)):
        if predictions[i] < labels[i]:
            type1 += 1
        if labels[i] < predictions[i]:
            type2 += 1

    return type1/len(labels), type2/len(labels)

#-----------------------------------------------------------------------------------------------------------------------


def apply_single_threshold(predictions, threshold):
    threshed = []
    for prediction in predictions:
        if prediction >= threshold:
            threshed.append(1)
        else:
            threshed.append(0)

    return threshed

#-----------------------------------------------------------------------------------------------------------------------


def apply_double_threshold(predictions, lower, upper):
    threshed = []
    for prediction in predictions:
        if lower <= prediction and prediction < upper:
            threshed.append(1)
        else:
            threshed.append(0)

    return threshed

#-----------------------------------------------------------------------------------------------------------------------