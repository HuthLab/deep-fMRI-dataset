# deep-fMRI-dataset
Code accompanying data release of first 5 fMRI sessions (LeBel et al.)

To clone and use this dataset:
* git clone git@github.com:HuthLab/deep-fMRI-dataset.git

To download the preprocessed data:
* `python encoding/load_dataset.py -download_preprocess`

This function will create a `data` dir if it does not exist and will use datalad to download the data.

The basic function to fit a model is encoding.py

it takes a series of arguments such as subject id, feature space to use, list of stories to fit, etc. 
It will automatically use the preprocessed data from the location that get_data saves the data to. 
