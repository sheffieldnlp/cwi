# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 13:47:02 2018

@author: pmfin
"""

from pathlib import Path
import pandas as pd
import pickle

def read_dataset(file_path):
    with open(file_path, encoding="utf-8") as file:
                fieldnames = ['hit_id', 'sentence', 'start_offset', 'end_offset', 'target_word', 'native_annots',
                              'nonnative_annots', 'native_complex', 'nonnative_complex', 'gold_label', 'gold_prob']

                dataset = pd.read_csv(file, names=fieldnames, sep="\t")
    return dataset

def getDatasetFromH5(h5filepath):
    dataset = pd.read_hdf(h5filepath, key='/table')
    return dataset

def getAllH5DatasetPaths(directory):
    filepaths = [x for x in list(Path(directory).iterdir()) if x.suffix == '.h5']
    return filepaths

def joinDatasets(h5_dataset_list):
    train_dataset = pd.DataFrame()
    test_dataset = pd.DataFrame()
    dev_dataset = pd.DataFrame()
    for pathstr in h5_dataset_list:
        
        path = Path(pathstr)
        sub_dataset = getDatasetFromH5(path)
        
        split_group = path.stem.split('_')[1]
        
        if split_group == 'Train':
            train_dataset = train_dataset.append(sub_dataset)
        elif split_group == 'Test':
            test_dataset = test_dataset.append(sub_dataset)
        elif split_group == 'Dev':
            dev_dataset = dev_dataset.append(sub_dataset)
        else:
            print("Something went wrong! All datasets should be of the form: file_Dev.h5, file_Test.h5 or file_Train.h5")
        
    return train_dataset, test_dataset, dev_dataset

def getCrosslingualSplit(full_dataset, test_language):
    test_lang_set = full_dataset.loc[full_dataset['language'] == test_language]
    train_lang_set = full_dataset.loc[full_dataset['language'] != test_language]
    return train_lang_set, test_lang_set

def getAllCrosslingualSplits(train_dataset, test_dataset, dev_dataset, language_list):
    
    # This function returns a 2-level dictionary in the form:
    # split[language_use_as_test][train/dev/test] = dataset
    
    split = {}
    subsplits = ['train', 'dev', 'test']
    
    # The language here is the language to ignore.
    for language in language_list:
        
        split[language] = {}
        
        for subsplit in subsplits:
            
            # This assumes that we are allowed to train on the training data of 
            # our test language (as well as the training data of our training languages)
            if subsplit == 'train':
                train_lang_set, test_lang_set = getCrosslingualSplit(train_dataset, language)
                final_set = train_lang_set.append(test_lang_set)
            elif subsplit == 'dev':
                _, test_lang_set = getCrosslingualSplit(dev_dataset, language)
                final_set = test_lang_set
            elif subsplit == 'test':
                _, test_lang_set = getCrosslingualSplit(test_dataset, language)
                final_set = test_lang_set
            else:
                print("Something went wrong! Subsplits should be train, dev and test only.")
            
            split[language][subsplit] = final_set
    return split

path_to_raw_data = 'data/raw'
destination_path_for_processed_files = 'data/processed/'

p = Path(path_to_raw_data)
dir_list = [x for x in p.iterdir() if x.is_dir()]
dir_names = [x.parts[-1] for x in dir_list]

for i in range(len(dir_list)):
    lang_dir = dir_list[i]
    lang_name = dir_names[i]
    p2 = Path(lang_dir)
    sub_filepaths = list(p2.iterdir())
    for sub_file_path in sub_filepaths:
        dataset = read_dataset(sub_file_path)
        dataset['language'] = lang_name
        sub_file_name = str(sub_file_path.parts[-1])[:-4]
        new_file_path = destination_path_for_processed_files + sub_file_name+ '.h5'
        p3 = Path(new_file_path)
        if not p3.exists():
            dataset.to_hdf(new_file_path, 'table', mode='w', append=True, complevel=3, complib='zlib')


split_filepath = Path(destination_path_for_processed_files + 'all_splits.pkl')
if not split_filepath.exists():
    all_datasets = getAllH5DatasetPaths(destination_path_for_processed_files)
    train_dataset, test_dataset, dev_dataset = joinDatasets(all_datasets)
    language_list = dir_names
    split = getAllCrosslingualSplits(train_dataset, test_dataset, dev_dataset, language_list)
    
    with open(split_filepath,'wb') as file:
        pickle.dump(split, file, protocol=pickle.HIGHEST_PROTOCOL)