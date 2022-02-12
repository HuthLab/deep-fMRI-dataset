import os
import pathlib


def get_data(download_preprocess=True):
    """ Get data from openneuro
    this will make a data directory in the main directory of the repo called data if it doesn't already exist.

    Args:
        download_preprocess (bool): download data to disk. If true downloads all preprocessed data.

    """
    
    current_path = pathlib.Path(__file__).parent.resolve()
    main_dir = pathlib.Path(__file__).parent.parent.resolve()
    data_dir = os.path.join(main_dir, 'data')

    os.chdir(main_dir)
    dir_exist = os.path.isdir(data_dir)
    if dir_exist:
        os.chdir(data_dir)
        os.system('datalad clone git@github.com:OpenNeuroDatasets/ds003020.git')
        if download_preprocess == True:
            os.chdir('ds003020')
            os.system('datalad get derivative/preprocessed_data')
    else:                      
        os.system('mkdir data')
        data_dir = os.path.join(main_dir, 'data')
        os.chdir(data_dir)
        os.system('datalad clone git@github.com:OpenNeuroDatasets/ds003020.git')
        if download_preprocess == True:
            os.chdir('ds003020')
            os.system('datalad get derivative/preprocessed_data')