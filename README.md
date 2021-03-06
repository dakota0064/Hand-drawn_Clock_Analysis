# How to use this pipeline to process and evaluate hand-drawn clock images:

## Step 1 
Install Python and pip

- Download page: https://www.python.org/downloads/
  - (Python 3.8 used in development, should be compatible with newer versions)
- Installation tutorial: https://docs.python-guide.org/starting/installation/

You can also choose to install Python using Anaconda, a dedicated Python package/environment manager. This is recommended for users entirely new to Python.
- Installation tutorial and Getting Started links: https://docs.anaconda.com/anaconda/install/
- After creating and activating a new environment you will need to install pip within that environment:
  - ```conda install pip``` 

Note that installing and setting up a python environment is often the toughest part for new (and often experienced) users. The listed tutorials might not be the best fit depending on your system and level of experience, so googling alternatives is recommended if this proves to be an issue.

## Step 2
Download this repository and unzip to wherever you want this tool and its marked-up images and results to live. Make sure to maintain the existing directory structure so that the data/results are saved to the right locations and the utils scripts import properly.

## Step 3
Using the command line, cd into the newly unzipped Hand-drawn_Clock_Analysis directory and install the necessary requirements using the following command:

```pip install -r requirements.txt```

If you chose the Anaconda route many of the required packages will likely come installed with your distribution.

If this process fails for any reason you can manually install the missing packages using pip or conda. Note that any complaints about missing imports beginning with utils (i.e. ```utils.feature_utils```) refer to local files included in this repository. You do not need to install these - if you have maintained the existing directory structure they should work as is.

## Step 4
Within the Hand-drawn_Clock_Analysis directory create a new directory called "data" and unzip the contents of sample_data.zip into it.

## Step 5
Run the scripts/notebooks in order, starting from 1.0. Python scripts (ending with the extension .py) can be run with no parameters - i.e. using a command such as: 

```python 1.0_compute_clock_features.py```

or

```python3 1.0_compute_clock_features.py```

Whereas notebook files (ending with the extension .ipynb) should be opened using Jupyter Notebook. To run this, enter the following commands:
  - ```pip install jupyter``` (optional, should already be installed from Step 3)
  - ```jupyter notebook```
This should open a new window in your browser. From here navigate to the desired notebook file and double-click to open. You can run all cells by clicking the 'Kernel' tab and choosing 'Restart and Run All'.

Any changes, such as to desired data/directory structures, need to be made within the files themselves. This can be done with any basic text editor or a dedicated Python IDE such as PyCharm. The parts most likely to require changing are marked at the beginning of each file. 

Take note of the details listed below, especially regarding 2.0 and feature normalization.

# Pipeline Script Descriptions
## 0.0_extract_clocks_from_MoCA.py
  
This script exists because our original dataset consisted of PDF files created from scanned paper copies of MoCA evaluations. The script reads the PDFs, locates the page with the clock drawing, isolates the clock, crops out the desired section and saves the new image using the same unique identifier as the PDF.

This process was very sensitive even on our original scans (about ~5% failure rate), and will likely only serve as an assitive tool for new datasets. Expect to have to do manual cropping to prepare clock images for best results.


## 1.0_compute_clock_features.py

The starting point if you're using the pipeline on our sample subset of data. 
- Matches cropped clock images with a csv file containing meta data (scan date, various MoCA score(s), age, gender, etc) and computes the full set of features for each image. 
- Creates a new csv file with this feature data and an annotated image for each clock showing various feature dimensions and detected segments.


## 2.0_feature_breakdown.ipynb

An interactive notebook showing the relationship between various feature value percentiles and their corresponding prediction value. Ideally we want all all features to have a monotonic relationship with their prediction value so we can provide meaningful correlation values - however, many features are naturally bitonic or more complex. The normalize functions in all the step 3 scripts use values obtained from this notebook to attempt to force the deviant features to be monotonic.

These steps are only relevant if you're trying to recreate correlation values. Since neither of the models are linear making these adjustments has no practical effect on the prediction power. 
- Comment out the calls to 'normalize' in 3.0 and 3.1 and run them as normal. 
- Run 2.0 on the output files and determine which features need adjustment to be made monotonic. 
- Change the values in the normalize functions, uncomment the calls to 'normalize', and run 3.0 and 3.1 again.

## 3.0_run_SVM_classifiers.py

Trains SVM classifiers on all subsets of the full feature list to predict the categorical scores and saves the results in 3 csv files. These csv files were visually inspected to find which feature subset produced the best results, and those subsets are now hardcoded to be written to a final csv file (denoted by the save_filename parameter) as the prediction values for that model.

## 3.1_run_forest_classifiers.py

Trains Random Forest classifiers on all subsets of the full feature list to predict the categorical scores and saves the results in 3 csv files. These csv files were visually inspected to find which feature subset produced the best results, and those subsets are now hardcoded to be written to a final csv file (denoted by the save_filename parameter) as the prediction values for that model.

## 3.2_run_forest_alternative_labels.py

Used to train a Random Forest classifier on some subset of the full feature list to predict a global feature, such as age or total MoCA score.

## 4.0_display_results.ipynb

An interactive notebook which displays the results from the normalized csv files computed in 3.0 and 3.1. 
