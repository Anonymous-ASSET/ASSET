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
1. First add the specified directory or jar file to the current dynamic Java path of MATLAB;  
   E.g., javaaddpath('E:/Program Files/weka.jar');  
2. Use the following code to load a .arff file:  
    file1 = java.io.File('E:\Datasets\AEEEM\EQ.arff'); % The path of EQ.arff is 'E:\Datasets\AEEEM\';  
    loader = weka.core.converters.ArffLoader;  % create an ArffLoader object  
    loader.setFile(file1);  % using ArffLoader to load data in file .arff  
    insts = loader.getDataSet; % get an Instances object  
    insts.setClassIndex(insts.numAttributes()-1); %  set the index of class label  
    [source,featureNamesSrc,sourceNDX,stringVals,relationName1] = weka2matlab(insts,[]); %{false,true}-->{0,1}, 'source' is a mat file (i.e., the dataset EQ);  

### Deatiled experimental results of RQ1/2/3/4:
* Under the folder 'Experiemnt Result/RQ1,2,3/', there are two excel files named AUC_Statistic and MCC_Statistic. They include the experiemntal results of 200 combinations based on the benckmark datasets in terms of AUC and MCC, respectively.
* In each excel file, the first colum is the name of each combination (e.g., EQ_JDT, where EQ is the source dataset, JDT is the target dataset). The second column is performance of ASSET, and the rest is the performance of baselines.
* Note that the performance is in percentage (%).
* Under the folder 'Experiemnt Result/RQ4/', there are two subfolders named '10%' and '5%', which include the experiemntal results of 200 combinations when 10% taregt instances being used as the training target dataset and when 5% target instances being used as the training target dataset, respectively.


### The results of AUC for sensitivity analysis
* Owing to the space limitation, in the Section 'Sensitivity Analysis', we just present the experiemntal results in terms of MCC. In fact, the same conclusions as the result of MCC can be found in the result of AUC.
* The result of AUC when fixing alpha and best, and changing the value of N1 anf N2 is presented as follows:
  ![The effect of N1 and N2 on the performance of ASSET in terms of AUC](https://github.com/Anonymous-ASSET/ASSET/Experiment+Results/Sensitivity+Analysis/3DGrid_AUC_N1_N2.jpg)
* The result of AUC when fixing N1 anf N2, and changing the value of alpha and best is presented as follows:
  ![The effect of alpha and beta on the performance of ASSET in terms of AUC](https://github.com/Anonymous-ASSET/ASSET/Experiment+Results/Sensitivity+Analysis/3DGrid_AUC_alpha_beta.jpg)

   
