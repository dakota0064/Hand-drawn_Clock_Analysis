***How to use this pipeline to process and evaluate hand-drawn clock images:***

Install the necessary requirements using the following command - "pip install -r requirements.txt"

Create a directory called "data" and unzip the contents of sample_data.zip into it.
Run the scripts/notebooks in order, starting from 1.0. Scripts can be run with no parameters - any changes, such as data/directory structures, need to be done within the files themselves. Take note of the details listed below.

**0.0_extract_clocks_from_MoCA.py**
  
This script exists because our original dataset consisted of PDF files created from scanned paper copies of MoCA evaluations. The script reads the PDFs, locates the page with the clock drawing, isolates the clock, crops out the desired section and saves the new image using the same unique identifier as the PDF. This process was very sensitive even on our original scans (about ~5% failure rate), and will likely only serve as an assitive tool for new datasets. Expect to have to do manual cropping to prepare clock images for best results.


**1.0_compute_clock_features.py**

The starting point if you're using the pipeline on our sample subset of data. Matches cropped clock images with a csv file containing meta data (scan date, various MoCA score(s), age, gender, etc) and computes the full set of features for each image. Creates a new csv file with this feature data and an annotated image for each clock showing various feature dimensions and detected segments.


**2.0_feature_breakdown.ipynb**

An interactive notebook showing the relationship between various feature value percentiles and their corresponding prediction value. Ideally we want all all features to have a monotonic relationship with their prediction value so we can provide meaningful correlation values - however, many features are naturally bitonic or more complex. The normalize functions in all the step 3 scripts use values obtained from this notebook to attempt to force the deviant features to be monotonic.

These steps are only relevant if you're trying to recreate correlation values. Since neither of the models are linear making these adjustments has no practical effect on the prediction power. Comment out the calls to 'normalize' in 3.0 and 3.1 and run them as normal. Run 2.0 on the output files and determine which features need adjustment to be made monotonic. Change the values in the normalize functions, uncomment the calls to 'normalize', and run 3.0 and 3.1 again.

**3.0_run_SVM_classifiers.py**

Trains SVM classifiers on all subsets of the full feature list to predict the categorical scores and saves the results in 3 csv files. These csv files were visually inspected to find which feature subset produced the best results, and those subsets are now hardcoded to be written to a final csv file (denoted by the save_filename parameter) as the prediction values for that model.

**3.1_run_forest_classifiers.py**

Trains Random Forest classifiers on all subsets of the full feature list to predict the categorical scores and saves the results in 3 csv files. These csv files were visually inspected to find which feature subset produced the best results, and those subsets are now hardcoded to be written to a final csv file (denoted by the save_filename parameter) as the prediction values for that model.

**3.2_run_forest_alternative_labels.py**

Used to train a Random Forest classifier on some subset of the full feature list to predict a global feature, such as age or total MoCA score.

**4.0_display_results.ipynb**

An interactive notebook which displays the results from the normalized csv files computed in 3.0 and 3.1. 
