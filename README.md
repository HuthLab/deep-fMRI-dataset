python# deep-fMRI-dataset
Code accompanying data release of natural language listening data from 5 fMRI sessions for each of 8 subjects (LeBel et al.) that can be found at [openneuro](https://openneuro.org/datasets/ds003020).

### To install the toolbox

To clone and use this dataset:
```
$ git clone git@github.com:HuthLab/deep-fMRI-dataset.git
```
then to intialize:
``` 
$ cd deep-fMRI-dataset
$ pip install .
```

### Downloading Data

First datalad needs to be installed which can be done with:
`sudo apt get install datalad`

Then, to automatically download the preprocessed data
```
$ cd encoding
$ python load_dataset.py -download_preprocess
```

This function will create a `data` dir if it does not exist and will use [datalad](https://github.com/datalad/datalad) to download the preprocessed data as well as feature spaces needed for fitting [semantic encoding models](https://www.nature.com/articles/nature17637). It will download ~20gb of data. Alternately, you can supply a different download location using the `--location DATA_DIR` flag. If you choose to change the default location of the data, make sure to update the `config.py` file with the new location.

To download the raw data you can use:

```
$ datalad clone https://github.com/OpenNeuroDatasets/ds003020.git

$ datalad get ds003020
```

### Fitting Models

The basic functionality for fitting encoding models can be found the script `encoding.py`, which takes a series of arguments such as subject id, feature space to use, list of training stimuli, etc. 
It will automatically use the preprocessed data from the location that get_data saves the data to. 

To fit a semantic encoding model (`eng1000`) for one subject (`UTS03`) and test it on held-out data:

```
$ python encoding/encoding.py --subject UTS03 --feature eng1000
```

The other optional parameters that encoding.py takes such as sessions, ndelays, single_alpha allow the user to change the amount of data and regularization aspects of the linear regression used. 

This function will then save model performance metrics and model weights as numpy arrays. 

### Voxelwise Encoding Model Tutorials

For more information about fitting voxelwise encoding models:
- This [repo](https://github.com/HuthLab/speechmodeltutorial) has a tutorial for fitting semantic encoding models
- Additionally this [repo](https://github.com/gallantlab/voxelwise_tutorials) has a wide selection of tutorials to fit encoding models
