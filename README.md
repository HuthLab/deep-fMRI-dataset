# deep-fMRI-dataset
Code accompanying data release of 5 fMRI sessions (LeBel et al.) that can be found at [openneuro](https://openneuro.org/datasets/ds003020)

### To install the toolbox

To clone and use this dataset:
```
$ git clone git@github.com:HuthLab/deep-fMRI-dataset.git
```
then to intiallize:
``` 
$ cd deep-fMRI-dataset
$ pip install .
```

### Downloading Data

This toolbox can automatically download the preprocessed data for you by using
```
$ python load_dataset.py -download_preprocess
```


This function will create a `data` dir if it does not exist and will use datalad to download the preprocessed data for fitting semantic
encoding models as well as the feature spaces needed. It will download ~20gb of data. 



To download the rest of the data separately you can use:

```
$ datalad clone https://github.com/OpenNeuroDatasets/ds003020.git

$ datalad get ds003020
```

### Fitting Models

The basic function to fit a model is encoding.py

it takes a series of arguments such as subject id, feature space to use, list of stories to fit, etc. 
It will automatically use the preprocessed data from the location that get_data saves the data to. 

To run a semantic model:

```
$ python encoding.py --subject <subject_code> --feature <feature_type>
```

The other optional parameters that encoding.py takes such as sessions, ndelays, single_alpha allow the user to change the amount of data and regularization aspects of the linear regression used. 

This function will then save model performance metrics and model weights as numpy arrays. 
