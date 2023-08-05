# ASSET
## Overview
The repository includes the following three folders:
* The folder 'Code' includes the source code of ASSET and all CPDP baselines (DMDAJFR, BurakMHD, CTDP, 3SW-MSTL, and DBN-CP);
* The folder 'Datasets' includes the benchmark datasets used in our paper;
   * Note that each defect dataset is a .arff file, which can be loaded in the MATLAB by calling weka.jar.
* The folder 'Experiemnt Results' includes the detailed experimental results (e.g., performance on each of 200 comninations) of RQ1/2/3/4 and sensitivity analysis;

## Operating Environment: 
* If you want to load the datasets and run ASSET or the baselines, please first meke sure that you have installed MATLAB, [WEKA](https://www.cs.waikato.ac.nz/ml/weka/index.html), and Python in your PC.
* The experimental environment in our paper is: MATLAB R2018b, WEKA 3.8.5, Python 3.6. 

## Introduction
### The method to load dataset
1. Add the specified directory or jar file to the current dynamic Java path of MATLAB;
   E.g., javaaddpath('E:/Program Files/weka.jar');
2. Use the following code to load a .arff file:
    file1 = java.io.File('E:\Datasets\AEEEM\EQ.arff'); % The path of EQ.arff is 'E:\Datasets\AEEEM\';
    loader = weka.core.converters.ArffLoader;  % create an ArffLoader object
    loader.setFile(file1);  % using ArffLoader to load data in file .arff
    insts = loader.getDataSet; % get an Instances object
    insts.setClassIndex(insts.numAttributes()-1); %  set the index of class label
    [source,featureNamesSrc,sourceNDX,stringVals,relationName1] = weka2matlab(insts,[]); %{false,true}-->{0,1}, 'source' is a mat file (i.e., the dataset EQ);

### Deatiled experimental results of RQ1/2/3/4:

